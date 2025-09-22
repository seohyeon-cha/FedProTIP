import torch.nn as nn
import torch.optim as optim
from progress.bar import Bar
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed, pil_loader
import time
import torch
import copy
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tqdm
import os

import numpy as np
from torchvision import transforms
from PIL import Image 

def get_one_hot(target, num_class, device):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class Client_base(object):
    def __init__(self, args, idx):
        self.args = args
        self.idx = idx
        self.device = args["device"][0]
        self.model = None
        self.old_model = None

        self.prev_trainset = None
        self.prev_transform = None
        self.trainset = None
        self.curr_transform = None 
        self.trainloader = None
        
        self.optimizer = None
        self.lr_schedule = False 
        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = 0

        self.memory_size = 2000
        if "domain" in self.args["dataset"] or "imagenet" in self.args["dataset"]:
            self.memory_size = 1000
        self.exemplar_set = []
        self.exemplar_label = []
        self.last_entropy = 0
        self.start = True
        self.signal = False
        self.new_task = False
        self.learned_classes = 0 
        self.thsh = 0.6
        self.proto_model = IncrementalNet(args, False).to(self.device)
        self.encode_model = None 
        self.learning_rate = self.args["lr"]
        
    def _update(self):
        for batch_idx, (_, images, labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            loss = self._compute_loss(_, images, labels)
            loss.backward()
            self.optimizer.step()

        return 
    
    def _finetune(self): # to update only part that corresponds to current class 
        for batch_idx, (_, images, labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            fake_targets = labels - self._known_classes 
            output = self.model(images)["logits"].to(self.device)
            loss = F.cross_entropy(output[:, self._known_classes :], fake_targets)
            loss.backward()
            self.optimizer.step()

        return 

    def local_training(self, task_id, model_old):
        # compute average entropy 
        self.model.eval()
        self.signal = False
        if self.new_task == True:
            self.signal = True 
        # self.signal = self.entropy_signal(self.trainloader)
        self.update_new_set() # update exemplar set 
        
        # initialize local model 
        self.model.train()
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args["lr"], 
                                       momentum=self.args["momentum"],weight_decay=self.args["weight_decay"])
        # pick old model
        if model_old[1] != None:
            if self.signal:
                self.old_model = model_old[1]
            else:
                self.old_model = model_old[0]
        else:
            if self.signal:
                self.old_model = model_old[0]

        
        if self.old_model != None:
            print('load best old model')
            self.old_model.to(self.device)
        
        for epoch in range(self.args["local_epochs"]):
            self._update() 
        
        # update proto_grad for choosing best old model
        if self.signal:
            proto_grad = self.prototype_mask()
        else:   
            proto_grad = None

        # return local model and proto_grad 
        return self.model.state_dict(), proto_grad

    def update_new_set(self):
        if self.signal and self._known_classes!=0: 
            m = int(self.memory_size / self._known_classes)
            self._reduce_exemplar_sets(m)
            for i in range(self._known_classes - self.args["increment"], self._known_classes): # change this part if class shuffled 
                idx = [j for j in range(len(self.prev_trainset)) if self.prev_trainset.labels[j] == i]
                if len(idx) > 0:
                    images = self.prev_trainset.images[idx]
                    self._construct_exemplar_set(images, m)
                    self.exemplar_label.append(i)

            self.mixTrainData()
        self.model.train()
        # mix trainset with exemplar and return new trainloader

    def entropy_signal(self, loader):
        self.model.eval()
        start_ent = True
        res = False

        for step, (indexs, imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                self.model = self.model.to(self.device)
                outputs = self.model(imgs)
            softmax_out = nn.Softmax(dim=1)(outputs["logits"])
            ent = entropy(softmax_out)

            if start_ent:
                all_ent = ent.float().cpu()
                all_label = labels.long().cpu()
                start_ent = False
            else:
                all_ent = torch.cat((all_ent, ent.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.long().cpu()), 0)

        overall_avg = torch.mean(all_ent).item()
        print(overall_avg)
        if overall_avg - self.last_entropy > 0.5: # 1.2로 바꾸기
            res = True
            print(res)
        self.last_entropy = overall_avg

        return res

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]

    def _construct_exemplar_set(self, images, m):
        class_mean_notrsf, class_mean, feature_extractor_output = self.compute_class_mean(images, self.prev_transform)
        exemplar = []
        now_class_mean = np.zeros((1, feature_extractor_output.shape[1]))
     
        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        self.exemplar_set.append(exemplar)
        
    def compute_class_mean(self, images, transform):
        self.model.eval()
        x, x_tt = self.Image_transform(images, transform)
        x = x_tt.to(self.device) if "domain" in self.args["dataset"] else x.to(self.device)
        # x_notrsf = x_notrsf.to(self.device)
        feature_extractor_output = F.normalize(self.model.convnet(x)['features'].detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return x, class_mean, feature_extractor_output

    def Image_transform(self, images, transform): # check transform 
        tt = transforms.Compose([
                        transforms.Resize((300, 300), interpolation=3),
                        transforms.RandomCrop((32, 32)), 
                        transforms.ToTensor()])
        
        if isinstance(images, str):
            data = transform(pil_loader(images)).unsqueeze(0)
            data_notrsf = tt(pil_loader(images)).unsqueeze(0)  
        elif not isinstance(images[0], np.ndarray):
            data = transform(pil_loader(images[0])).unsqueeze(0)
            data_notrsf = tt(pil_loader(images[0])).unsqueeze(0)
            for index in range(1, len(images)):
                data = torch.cat((data, transform(pil_loader(images[index])).unsqueeze(0)), dim=0)
                data_notrsf = torch.cat((data_notrsf, tt(pil_loader(images[index])).unsqueeze(0)), dim=0)
        else:
            data = transform(Image.fromarray(images[0])).unsqueeze(0)
            data_notrsf = tt(Image.fromarray(images[0])).unsqueeze(0)
            for index in range(1, len(images)):
                data = torch.cat((data, transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
                data_notrsf = torch.cat((data_notrsf, tt(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data, data_notrsf

    def mixTrainData(self): # change if class shuffle 
        datas, labels = [], []
        if len(self.exemplar_set)!=0 and self._known_classes != self._total_classes:
            datas = [exemplar for exemplar in self.exemplar_set if exemplar is not None]
            length = len(datas[0])
            labels = [np.full((length), label) for label in self.exemplar_label]

        for label in range(self._known_classes, self._total_classes):
            data = self.trainset.images[np.array(self.trainset.labels)==label]
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1,len(datas)):
            con_data = np.concatenate((con_data,datas[i]),axis=0)
            con_label = np.concatenate((con_label,labels[i]),axis=0)
        
        from utils.data_manager import DummyDataset
        from torch.utils.data import DataLoader
        if "domain" in self.args["dataset"] or "imagenet" in self.args["dataset"]:
            use_path = True 
        else:
            use_path = False
        self.mixed_trainset = DummyDataset(con_data, con_label, self.curr_transform, use_path)
        self.trainloader = DataLoader(self.mixed_trainset, 
                                        batch_size=self.args["batch_size"], shuffle=True, num_workers=1)
    
    

    def prototype_mask(self):
        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        iters = 50 if "domain" in self.args["dataset"] else 100 
        criterion = nn.CrossEntropyLoss().to(self.device)
        proto = []
        proto_labels = []
        proto_grad = []

        # get prototype images 
        for label in range(self._known_classes, self._total_classes):
            idx = [j for j in range(len(self.trainset)) if self.trainset.labels[j] == label]
            if len(idx) > 0:
                images = self.trainset.images[idx]
                _, class_mean, feature_extractor_output = self.compute_class_mean(images, self.curr_transform)
                dis = class_mean - feature_extractor_output
                dis = np.linalg.norm(dis, axis=1)
                pro_index = np.argmin(dis)
                proto.append(images[pro_index])
                proto_labels.append(label)

                # image_data = images[pro_index]
                # if isinstance(image_data, str):
                #     image_data, _ = self.Image_transform(image_data, self.curr_transform) #
                # if isinstance(image_data, torch.Tensor):
                #     image_data = image_data.squeeze(0).cpu().numpy()  

                #     if image_data.shape[0] == 3:  # For tensors with channels
                #         image_data = np.transpose(image_data, (1, 2, 0))  # Convert [C, H, W] to [H, W, C]
                #     if image_data.dtype in [np.float32, np.float64]:
                #         image_data = (image_data * 255).astype(np.uint8)
                # image = Image.fromarray(image_data)
                # image.save(f"./run/lga_perturb/orig_label{label}.png")
        
        if self.args["classIL"] == "True":
            self.proto_model.update_fc(self._cur_task, 0)
        else:
            if len(self.proto_model.fc) == 0:
                self.proto_model.update_fc(self._cur_task, 0)
                
        for i in range(len(proto_labels)):
            self.model.eval()
            data = proto[i]
            label = proto_labels[i]
            # data = Image.fromarray(data)
            
            if isinstance(data, str):
                data, data_tt = self.Image_transform(data, self.curr_transform)
                data = data_tt if "domain" in self.args["dataset"] else data 
                if isinstance(data, torch.Tensor):
                    data = data.squeeze(0).cpu().numpy()  
                    if data.ndim == 3:  # For tensors with channels
                        data = np.transpose(data, (1, 2, 0))  # Convert [C, H, W] to [H, W, C]
                    if data.dtype in [np.float32, np.float64]:
                        data = (data * 255).astype(np.uint8)

            data, label = tt(data), torch.Tensor([label]).long()
            data, label = data.to(self.device), label.to(self.device)
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
            target = get_one_hot(label, self._total_classes, self.device)

            # changed this! 
            proto_lr = 0.2
            opt = optim.SGD([data, ], lr=proto_lr, weight_decay=0.00001)
            
            proto_model_w = copy.deepcopy(self.model.state_dict())
            self.proto_model.load_state_dict(proto_model_w)
            self.proto_model = self.proto_model.to(self.device)

            # get perturbed images 
            for ep in range(iters):
                outputs = self.proto_model(data)["logits"]
                if self.args["classIL"] != "True":  # domain-IL
                    label = (label % self.args["increment"]).to(self.device)
                    loss_cls = F.cross_entropy(outputs, label)
                else:
                    loss_cls = F.binary_cross_entropy_with_logits(outputs, target)       
                opt.zero_grad()
                loss_cls.backward()
                opt.step()

            data = data.detach().clone().to(self.device).requires_grad_(False)
            self.encode_model = self.encode_model.to(self.device)
            outputs = self.encode_model(data)

            # image = np.asarray(tp(data.clone().squeeze(0).cpu()))
            # image = Image.fromarray(image)
            # image.save(f"./run/lga_perturb/label{i}.png")

            if self.args["classIL"] != "True":  # domain IL
                label = (label % self.args["increment"])

            loss_cls = criterion(outputs, label)
            dy_dx = torch.autograd.grad(loss_cls, self.encode_model.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            proto_grad.append(original_dy_dx)

        return proto_grad


    def _compute_loss(self, indexs, imgs, label):
        output = self.model(imgs)["logits"]
        target = get_one_hot(label, self._total_classes, self.device)
        
        if self.args["classIL"] != "True": 
            # --- Domain Incremental: no weighting ---
            label = (label % self.args["increment"]).to(self.device)
            target = get_one_hot(label, self.args["increment"], self.device)
            output, target = output.to(self.device), target.to(self.device)

            if self.old_model is None:  # first domain
                loss_cur = F.cross_entropy(output, label)
                return loss_cur
            else:
                loss_cur = F.cross_entropy(output, label)

                # KL distillation on the SAME class space with temperature
                with torch.no_grad():
                    old_logits = self.old_model(imgs)["logits"].to(self.device)
                T = 2.0
                alpha = 0.5
                p_old_T = F.softmax(old_logits / T, dim=1)
                log_p_new_T = F.log_softmax(output / T, dim=1)
                loss_kd = F.kl_div(log_p_new_T, p_old_T, reduction='batchmean') * (T * T)

                return (1.0 - alpha) * loss_cur + alpha * loss_kd


        output, target = output.to(self.device), target.to(self.device) 
        if self.old_model == None: # t=1 
            w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none')) # L_GC 
            # loss_cur = F.cross_entropy(output, label)

            return loss_cur
        else: 
            # weighted CE loss 
            w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none')) # L_GC
            # loss_cur = torch.mean(w * F.cross_entropy(output, label)) # L_GC


            # distillation loss 
            distill_target = target.clone()
            old_target = torch.sigmoid(self.old_model(imgs)["logits"])
            old_task_size = old_target.shape[1]
            distill_target[..., :old_task_size] = old_target
            loss_old = F.binary_cross_entropy_with_logits(output, distill_target) # L_RD 
            # loss_old = F.cross_entropy(output, distill_target)

            return loss_cur + loss_old

    def efficient_old_class_weight(self, output, label):
        # pred = torch.sigmoid(output)
        pred = torch.softmax(output, dim=1)
        N, C = pred.size(0), pred.size(1)
        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        target = get_one_hot(label, self._total_classes, self.device)
        g = torch.abs(pred.detach() - target)
        if self.task_id_old > 0:
           z=torch.div(self.task_id_old,self.task_id_old+1)
           z=g.clone().fill_(z)
           g=torch.pow(g,z)
        g = (g * class_mask).sum(1).view(-1, 1)

        if self._known_classes != 0:
            for i in range(self._known_classes):
                ids = torch.where(ids != i, ids, ids.clone().fill_(-1))
            index2 = torch.ne(ids, -1).float()
            if index2.sum() != 0:
                w2 = torch.div(g * index2, (g * index2).sum() / index2.sum())
            else:
                w2 = g.clone().fill_(0.)
            ids = label.view(-1, 1)
            for i in range(self.task_id_old):
                task = (i + 1) * self.args["increment"]
                classes = [n for n in range(self._known_classes) if n >=task-self.args["increment"] and n < task]
                for j in classes:
                    ids = torch.where(ids != j, ids, ids.clone().fill_(-1))
                index = torch.eq(ids, -1).float()
                if index.sum() != 0:
                    w = torch.div(g * index, (g * index).sum() / index.sum())
                else:
                    w = g.clone().fill_(0.)
                w2 += w
            w2=torch.clamp(w2, 0.5, 5)
            return w2

        else:
            w = g.clone().fill_(1.)
        return w
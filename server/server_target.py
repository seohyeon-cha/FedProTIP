import os, sys
sys.path.append('../')
from client.client_target import Client_target as Client
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from progress.bar import Bar
import numpy as np
from utils.toolkit import count_parameters
from utils.inc_net import IncrementalNet, Increment_ViT
from utils.data_manager import DataManager, partition_data, DatasetSplit, UnlabeledImageDataset, DataIter, average_weights, setup_seed
from models.target_gen import * 
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import math
import wandb 
import shutil

class Server_TARGET(object):
    def __init__(self,args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self.args = args
        self.seed = args["seed"]
        self.device = args["device"][0]
        self.wandb = args["wandb"]
        self.clients = []

        if self.args["dataset"] == "cifar100":
            # for target # should change this! into ImageNet setting 
            self.nums = 10000 
            self.syn_bs = 128 if self.args["img_size"] == 224 else 256
            self.syn_sample_bs = 128 if self.args["img_size"] == 224 else 256
            self.syn_round = 40 
            self.T = 20.0
            self.warmup = 10
            self.is_maml = 1
            self.tau = 1
            self.lr_g = 0.002
            self.lr_z = 0.01
            self.g_steps = 40  
            self.kd_steps = 400
            self.act = 0.0
            self.oh = 0.5 # weight for L_G (CE)
            self.adv = 1.0 # weight for L_G (div)
            self.bn = 10.0 # weight for L_G (BN)
            self.reset_l0 = 1
            self.reset_bn = 0
            self.bn_mmt = 0.9
        else:
            self.nums = 12800
            self.syn_bs = 64
            self.syn_sample_bs = 64
            self.g_steps = 100 if "domain" in self.args["dataset"] else 40
            self.is_maml = 0   
            self.kd_steps = 400     
            self.warmup = 20
            self.lr_g = 2e-4
            self.lr_z = 0.01   
            self.oh = 0.1  
            self.T = 5     
            self.act = 0.0 
            self.adv = 1.0 
            self.bn = 0.1  
            self.reset_l0 = 0 
            self.reset_bn = 0 
            self.bn_mmt = 0.9  
            self.syn_round = 200 if "domain" in self.args["dataset"] else 40
            self.tau = 1
            
        self.data_manager = DataManager(args["dataset"],args["class_shuffle"], args["seed"],args["increment"],args["increment"],args)
        pretrained = True if self.args["pretrained"]=="True" else False 
        if 'vit' in self.args["net"]:
            self.global_model = Increment_ViT(args, pretrained=pretrained).to(self.device)
        else:
            self.global_model = IncrementalNet(args, pretrained=pretrained).to(self.device)

        # create clients
        for idx in range(args["n_clients"]):
            self.clients.append(Client(args, idx))
            
        self.class_order = torch.tensor(self.data_manager.get_class_order()).to(self.device)
        self.topk = 5
        self.each_task = args["increment"]
        self.save_dir = args["save_dir"]
        self.test_dataset = []
        self.freeze_layers_flag = False 
        self.frozen_layer_names = []

    def freeze_layers(self):
        self.frozen_layer_names = []
        print("Freezing layers 1 and 2 of network.")
        if "vit" in self.args["net"]:
            for k, (name, param) in enumerate(self.global_model.vit.named_parameters()):
                if k < 4:
                    print(f'Freeze layer {k}: {name}')
                    param.requires_grad = False
                    self.frozen_layer_names.append(name)
                    
            for k, (name, param) in enumerate(self.global_model.vit.blocks.named_parameters()):
                if k < 12 * 6: # freeze 10 blocks 
                    print(f'Freeze layer {k}: {name}')
                    param.requires_grad = False
                    self.frozen_layer_names.append(name)
        else:
            for k, (name, param) in enumerate(self.global_model.convnet.named_parameters()):
                if k < 30: # block 1 (~14) / block 2 (~29) / block 3 (~44) / block 4 (~59)
                    print(f'Freeze layer {k}: {name}')
                    param.requires_grad = False
                    self.frozen_layer_names.append(name)

    def train(self):
        cnn_curve_gt_true = {"top1": [], "top5": []}
        cnn_curve_gt_false = {"top1": [], "top5": []}
        acc_t_t_gt_true, acc_t_t_gt_false = [], []
        acc_t_T_gt_true, acc_t_T_gt_false = [], []
        self.acc_matrix_gt_true = []
        self.acc_matrix_gt_false = []

        for task in range(self.data_manager.nb_tasks):
            self._cur_task += 1
            self._total_classes = self._known_classes + self.data_manager.get_task_size(self._cur_task) # previous classes + # classes of current task 

            if self.args["classIL"] != "True" and task > 0:
                pass
            else:
                self.global_model.update_fc(self._cur_task, 1) # increment fc neurons  

            print("Learning on {}-{}".format(self._known_classes, self._total_classes))
            train_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train",mode="train") 
            test_dataset = self.data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
            self.test_loader_target = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=1)
            # setup_seed(self.seed)
            self.test_dataset.append(self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test", mode="test"))
            setup_seed(self.seed)
            
            # Load synthetic data
            if self._cur_task != 0:
                self.syn_data_loader = self.get_syn_data_loader()

            # FL training
            user_groups,_ = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["n_clients"]) # divide samples indep. to classes 
            prog_bar = tqdm(range(self.args["epochs"]))

            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.global_model.parameters()), lr=self.args["lr"], 
                                       momentum=self.args["momentum"],weight_decay=self.args["weight_decay"])
            self.scheduler = None 

            for _, epoch in enumerate(prog_bar):
                local_weights = []
                m = max(int(self.args["frac"] * self.args["n_clients"]), 1)
                selected_clients = np.random.choice(range(self.args["n_clients"]), m, replace=False)
                for idx in selected_clients:
                    print("Fine-tuning on client: ", idx)
                    local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                                                    batch_size=self.args["batch_size"], shuffle=True, num_workers=1)
                    self.clients[idx].trainloader = local_train_loader
                    self.clients[idx]._known_classes = self._known_classes
                    self.clients[idx].model = copy.deepcopy(self.global_model)
                    self.clients[idx]._cur_task = self._cur_task
                    self.clients[idx]._known_classes = self._known_classes
                    self.clients[idx].scheduler = self.scheduler
                    if self._cur_task != 0:
                        self.clients[idx].old_model = self.old_model
                        self.clients[idx].syn_data_loader = self.syn_data_loader
                    w = self.clients[idx].local_training(self._cur_task) # local training 
                    local_weights.append(copy.deepcopy(w))
                    del w, local_train_loader
                    torch.cuda.empty_cache()

                # update global weights
                global_weights = average_weights(local_weights)
                self.global_model.load_state_dict(global_weights)
                
                if (epoch+1) % 10 == 0:
                    acc_at_round, _ = self.eval_task()
                    print("**************** test accuracy: ", acc_at_round["top1"])
                    if self.wandb:
                        wandb.log({
                            "Global round": (epoch+1) + task * self.args["epochs"],
                            "Mean task accuracy": acc_at_round["top1"] 
                        })

            # Freeze layers after learning the first task
            if task == 0 and not self.freeze_layers_flag and self.args["pretrained"] == "True":
                self.freeze_layers()
                self.freeze_layers_flag = True  # Ensure layers are frozen only once



            # after_task() in target original code 
            # for model distillation 
            if hasattr(self, "old_model"):
                del self.old_model
                torch.cuda.empty_cache()

            self.old_model = copy.deepcopy(self.global_model).to("cpu")
            self.old_model.freeze()

            if hasattr(self, "syn_data_loader"):
                del self.syn_data_loader
                torch.cuda.empty_cache()

            if self._cur_task + 1 != self.args["n_tasks"]:
                self.data_generation()
                if self._cur_task >= 1:
                    self.remove_syn_imgs()

            """
            Evaluation
            """
            # `gt=True` evaluation
            self.args["gt"] = "True"
            cnn_accy_gt_true, _ = self.eval_task()
            acc_t_t_gt_true.append(cnn_accy_gt_true["grouped"]["new"])
            self.acc_matrix_gt_true.append(cnn_accy_gt_true["grouped"]["record"])

            # `gt=False` evaluation
            self.args["gt"] = "False"
            cnn_accy_gt_false, _ = self.eval_task()
            acc_t_t_gt_false.append(cnn_accy_gt_false["grouped"]["new"])
            self.acc_matrix_gt_false.append(cnn_accy_gt_false["grouped"]["record"])

            self._known_classes = self._total_classes
            if self.wandb:
                wandb.log({
                    "Task": task + 1,
                    "Accuracy (GT1)": cnn_accy_gt_true["top1"],
                    "Accuracy (GT0)": cnn_accy_gt_false["top1"]
                })

            # Log accuracies
            cnn_curve_gt_true["top1"].append(cnn_accy_gt_true["top1"])
            cnn_curve_gt_false["top1"].append(cnn_accy_gt_false["top1"])
            print("(GT 1) Grouped: {}".format(cnn_accy_gt_true["grouped"]))
            print("(GT 0) Grouped: {}".format(cnn_accy_gt_false["grouped"]))
            print("(GT 1) CNN top1 curve: {}".format(cnn_curve_gt_true["top1"]))
            print("(GT 0) CNN top1 curve: {}".format(cnn_curve_gt_false["top1"]))

        # Compute forgetting
        print("Forgetting (gt=True):")
        self.compute_forgetting_from_matrix(self.acc_matrix_gt_true)
        print("Forgetting (gt=False):")
        self.compute_forgetting_from_matrix(self.acc_matrix_gt_false)

    def get_syn_data_loader(self):
        if self.args["dataset"] =="cifar100":
            dataset_size = 50000
        elif self.args["dataset"] == "imagenet-r":
            dataset_size = 24000
        elif self.args["dataset"] == "DomainNet":
            dataset_size = 50000
        elif self.args["dataset"] == "tiny_imagenet":
            dataset_size = 100000
        elif self.args["dataset"] == "imagenet100":
            dataset_size = 130000 
        elif self.args["dataset"] == "cubs":
            dataset_size = 5994
        elif 'domainnet' in self.args["dataset"]:
            dataset_size = 60000

        iters = math.ceil(dataset_size / (self.args["n_clients"]*self.args["n_tasks"]*self.args["batch_size"]))
        syn_bs = self.syn_bs
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task-1))
        print("iters{}, syn_bs:{}, data_dir: {}".format(iters, syn_bs, data_dir))

        # resize when loading (not generating 224)
        data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.args["img_size"] == 224 and self.args["dataset"] == "cifar100":
            self.train_trsf = transforms.Compose([
                    transforms.Resize(224), 
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**dict(data_normalize)),
            ])
        syn_dataset = UnlabeledImageDataset(data_dir, transform=self.train_trsf, nums=self.nums)
        syn_data_loader = DataLoader(
            syn_dataset, batch_size=syn_bs, shuffle=True, num_workers=4)
        return syn_data_loader

    def get_all_syn_data(self):
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))

        # resize when loading (not generating 224)
        data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.args["img_size"] == 224 and self.args["dataset"] == "cifar100":
            self.train_trsf = transforms.Compose([
                    transforms.Resize(224), 
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**dict(data_normalize)),
            ])

        syn_dataset = UnlabeledImageDataset(data_dir, transform=self.train_trsf, nums=self.nums)
        loader = DataLoader(
            syn_dataset, batch_size=self.syn_sample_bs, shuffle=True, num_workers=4)
        return loader

    def data_generation(self):
        
        nz = 256 # noise dimension 
        img_size = 32 if self.args["dataset"] == "cifar100" else self.args["img_size"]
        img_shape = (3, img_size, img_size)
        generator = Generator(self.args, nz=nz, ngf=64, img_size=img_size, nc=3).to(self.device) # convnet backbone

        # student model 
        student = copy.deepcopy(self.global_model)
        if self.args["net"] == 'alexnet':
            student.apply(alexnet_weight_init)
        else:
            student.apply(weight_init)

        tmp_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir) 
        
        if self.args["dataset"] == "cifar100":
            data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            normalizer = Normalizer(**dict(data_normalize)) # data_normalize 
            # if self.args["img_size"] == 224:
            #     self.train_trsf = transforms.Compose([transforms.Resize(224), 
            #             transforms.RandomResizedCrop(224),
            #             transforms.RandomHorizontalFlip(),
            #             transforms.ToTensor(),
            #             transforms.Normalize(**dict(data_normalize)),
            #     ])
            # else:
            self.train_trsf = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(**dict(data_normalize)),
                ])
        else:
            data_normalize = dict(mean=[0, 0, 0], std=[1, 1, 1])
            normalizer = Normalizer(**dict(data_normalize))
            self.train_trsf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
            
        synthesizer = GlobalSynthesizer(self.args, copy.deepcopy(self.global_model), student, generator,
                    nz=nz, num_classes=self._total_classes, img_size=img_shape, init_dataset=None,
                    save_dir=tmp_dir,
                    transform=self.train_trsf, normalizer=normalizer,
                    synthesis_batch_size=self.syn_bs, sample_batch_size=self.args["batch_size"],
                    iterations=self.g_steps, warmup=self.warmup, lr_g=self.lr_g, lr_z=self.lr_z,
                    adv=self.adv, bn=self.bn, oh=self.oh,
                    reset_l0=self.reset_l0, reset_bn=self.reset_bn,
                    bn_mmt=self.bn_mmt, is_maml=self.is_maml)
        
        criterion = KLDiv(T=self.T)
        optimizer = torch.optim.SGD(student.parameters(), lr=self.args["lr_MD"], weight_decay=0.0002,
                            momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)

        for it in range(self.syn_round + self.warmup):
            synthesizer.synthesize(self.device) # generate synthetic data
            if it >= self.warmup:
                self.kd_train(student, self.global_model, criterion, optimizer) # kd_steps
                test_acc = self._compute_accuracy(student, self.test_loader_target)
                print("Task {}, Data Generation, Epoch {}/{} =>  Student test_acc: {:.2f}".format(
                    self._cur_task, it + 1, self.syn_round + self.warmup, test_acc,))
                scheduler.step()

        del student
        print("For task {}, data generation completed! ".format(self._cur_task))  

    def remove_syn_imgs(self):
        folder_path = os.path.join(self.save_dir, "task_{}".format(self._cur_task-1))
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (folder_path, e))


    def kd_train(self, student, teacher, criterion, optimizer):
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data() 
        data_iter = DataIter(loader)
        for i in range(self.kd_steps):
            images = data_iter.next().to(self.device)
            with torch.no_grad():
                t_out = teacher(images)["logits"]
            s_out = student(images.detach())["logits"]
            loss_s = criterion(s_out, t_out.detach())
            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def eval_task(self):
        y_pred_all = [] 
        y_true_all = []
        self.task_acc = 0
        self.task_count = 0

        evaluate = self._evaluate_domain if self.args["classIL"] != "True" else self._evaluate
        for testset in self.test_dataset:
            self.test_loader = DataLoader(testset, batch_size=self.args["batch_size"], shuffle=True, num_workers=1)
            y_pred, y_true = self._eval_cnn(self.test_loader)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            if hasattr(self, "_class_means"):
                y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
                nme_accy = evaluate(y_pred, y_true)
            else:
                nme_accy = None

        y_pred_all, y_true_all = np.concatenate(y_pred_all), np.concatenate(y_true_all)
        cnn_accy = evaluate(y_pred_all, y_true_all)
        return cnn_accy, nme_accy
    
    def _eval_cnn(self, loader):
        self.global_model.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.global_model(inputs)["logits"]
                
            if self.args["gt"] == "True":
                for i in range(len(outputs)):
                    task_id = int(targets[i]//self.each_task)
                    outputs[i][0:task_id*self.each_task] = -float('inf')
                    outputs[i][(task_id+1)*self.each_task:] = -float('inf')
                    
                targets = targets 
                
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
        return np.concatenate(y_pred), np.concatenate(y_true)   

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, increment=self.each_task)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret
    
    def _evaluate_domain(self, y_pred, y_true):
        ret = {}
        domain = 'domain' in self.args["dataset"] 
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, increment=self.each_task, domain=domain)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]

        class_id = (y_true % self.each_task).astype(np.int64) if domain else y_true
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(class_id, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        return ret

    def _eval_nme(self, loader, class_means):
        self.global_model.eval()
        EPSILON = 1e-8
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true 

    def _extract_vectors(self, loader):
        self.global_model.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self.global_model, nn.DataParallel):
                _vectors = tensor2numpy(
                    self.global_model.module.extract_vector(_inputs.to(self.device))
                )
            else:
                _vectors = tensor2numpy(
                    self.global_model.extract_vector(_inputs.to(self.device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def compute_forgetting_from_matrix(self, acc_matrix):
        """
        acc_matrix: list of lists, acc_matrix[t][k] = accuracy on task k after training task t.
        (0-indexed tasks: t = 0..T-1, k = 0..t)
        """
        
        print(acc_matrix)
        T = len(acc_matrix)
        forgetting = []

        for k in range(T):  # for each task k
            acc_curve = [acc_matrix[t][k] for t in range(k, T)]  # from when task k appears until end
            if len(acc_curve) > 1:
                best_prev = max(acc_curve[:-1])  # exclude final accuracy
                last = acc_curve[-1]
                forgetting.append(best_prev - last)
            else:
                forgetting.append(0.0)  # no forgetting possible for last task

        avg_forgetting = sum(forgetting) / len(forgetting)
        print(f"FT (Forgetting): {avg_forgetting:.4f}")
        return avg_forgetting

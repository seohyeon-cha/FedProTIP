import torch.nn as nn
import torch
import copy
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from models import *
from torch.utils.data import DataLoader
import random
from utils import *


class proxyServer:
    def __init__(self, args, global_model, test_transform):
        super(proxyServer, self).__init__()

        self.args = args
        self.Iteration = 100 # 100 
        if "domain" in self.args["dataset"]:
            self.Iteration = 50
        self.model = global_model
        self.encode_model = None 

        self.test_transform = test_transform
        self.new_set = []
        self.new_set_label = []
        self.device =  self.args["device"][0]
        self.num_image = 20
        self.pool_grad = None
        self.best_model_1 = None
        self.best_model_2 = None
        self.best_perf = 0

        self.known_classes = 0

    def dataloader(self, pool_grad):
        self.pool_grad = pool_grad
        if len(pool_grad) != 0: # new task detected 
            self.reconstruction()
            print("New set label: ", self.new_set_label)
            monitor_dataset = self.getMonitorData(self.new_set, self.new_set_label)
            self.monitor_loader = DataLoader(dataset=monitor_dataset, 
                                        batch_size=self.args["batch_size"], shuffle=True, num_workers=1, drop_last=False)
            self.best_model_1 = self.best_model_2

        cur_perf = self.monitor()
        print(cur_perf)
        if cur_perf >= self.best_perf:
            self.best_perf = cur_perf
            self.best_model_2 = copy.deepcopy(self.model)

    def model_back(self):
        return [self.best_model_1, self.best_model_2]
    
    def monitor(self):
        self.model.eval()
        correct, total = 0, 0
        for step, (_, imgs, labels) in enumerate(self.monitor_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            fake_targets = labels - self._known_classes
            start_point = 0 if self.args["classIL"] != "True" else self._known_classes 
            with torch.no_grad():
                self.model = self.model.to(self.device)
                outputs = self.model(imgs)["logits"][:, start_point:]
            predicts = torch.max(outputs, dim=1)[1]
            if self.args["classIL"] != "True": 
                correct += (predicts.cpu() == fake_targets.cpu()).sum()
            else:
                correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        
        return accuracy

    def gradient2label(self):
        pool_label = []
        for w_single in self.pool_grad:
            pred = torch.argmin(torch.sum(w_single[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            pool_label.append(pred.item())

        return pool_label

    def reconstruction(self):
        self.new_set, self.new_set_label = [], []

        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        pool_label = self.gradient2label() # obtain GT label from pool_grad 
        pool_label = np.array(pool_label)
        n_classes = self.args["increment"] if self.args["classIL"] != "True" else self.args["n_classes"]
        class_ratio = np.zeros((1, n_classes))

        for i in pool_label:
            class_ratio[0, i] += 1
        print("Class Ratio in Pool:", class_ratio)

        for label_i in range(n_classes):
            if class_ratio[0, label_i] > 0:
                num_augmentation = self.num_image
                augmentation = []
                
                grad_index = np.where(pool_label == label_i)
                for j in range(len(grad_index[0])):
                    print('reconstruct_{}, {}-th'.format(label_i, j))
                    grad_truth_temp = self.pool_grad[grad_index[0][j]]

                    lr = 0.5 if "imagenet" in self.args["dataset"] else 0.1 
                    img_size = 224 if "imagenet" in self.args["dataset"] else 32 
                    dummy_data = torch.randn((1, 3, img_size, img_size)).to(self.device).requires_grad_(True)
                    label_pred = torch.Tensor([label_i]).long().to(self.device).requires_grad_(False)

                    optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                    scheduler = None 
                    if dummy_data.shape[-1] != 32:
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * self.Iteration), int(0.75 * self.Iteration)], gamma=0.1)
                    criterion = nn.CrossEntropyLoss().to(self.device)

                    recon_model = copy.deepcopy(self.encode_model)
                    recon_model = recon_model.to(self.device)

                    for iters in range(self.Iteration):
                        def closure():
                            optimizer.zero_grad()
                            pred = recon_model(dummy_data)
                            dummy_loss = criterion(pred, label_pred)
                            dummy_dy_dx = torch.autograd.grad(dummy_loss, recon_model.parameters(), create_graph=True) 

                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, grad_truth_temp):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        optimizer.step(closure)
                        current_loss = closure().item()
                        if scheduler is not None:
                            scheduler.step()

                        if iters == self.Iteration - 1:
                            print(current_loss)

                        if iters >= self.Iteration - self.num_image:
                            dummy_data_temp = np.asarray(tp(dummy_data.clone().squeeze(0).cpu()))
                            augmentation.append(dummy_data_temp)
                            del dummy_data_temp
                            torch.cuda.empty_cache()

                            # # Convert to image and save
                    # image = Image.fromarray(dummy_data_temp)
                    # image.save(f"./run/glfc_recon/label{label_i}-iter{iters}.png")

                self.new_set.append(augmentation)
                self.new_set_label.append(label_i)

    def getMonitorData(self, new_samples, new_labels):
        datas, labels = [], []
        monitorData, monitorLabel = [], []
        if len(new_samples) != 0 and len(new_labels) != 0:
            datas = [exemplar for exemplar in new_samples]
            for i in range(len(new_samples)):
                length = len(datas[i])
                labels.append(np.full((length), new_labels[i]))
        
        monitorData = datas[0]
        monitorLabel = labels[0]

        for i in range(1,len(datas)):
            monitorData = np.concatenate((monitorData,datas[i]),axis=0)
            monitorLabel = np.concatenate((monitorLabel,labels[i]),axis=0)
        
        from utils.data_manager import DummyDataset
        monitor_dataset = DummyDataset(monitorData, monitorLabel, self.test_transform)

        return monitor_dataset
        

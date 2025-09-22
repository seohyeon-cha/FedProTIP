import torch.nn as nn
import torch.optim as optim
from progress.bar import Bar
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import time
import torch
import copy
from torch.nn import functional as F
from tqdm import tqdm
import os

class Client_base(object):
    def __init__(self, args,idx):
        self.args = args
        self.idx = idx
        self.device = args["device"][0]
        self.model = None
        self.trainloader = None
        self.optimizer = None
        self._known_classes = 0
        self._cur_task = 0

    def _update(self):
        for batch_idx, (_, images, labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)["logits"].to(self.device)
            loss = F.cross_entropy(output, labels) 
            loss.backward()
            self.optimizer.step()
        return
    
    def _finetune(self):  
        for batch_idx, (_, images, labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            fake_targets = labels - self._known_classes
            output = self.model(images)["logits"].to(self.device)
            start_point = 0 if self.args["classIL"] != "True" else self._known_classes
            loss = F.cross_entropy(output[:, start_point:], fake_targets)
            loss.backward()
            self.optimizer.step()
        return

    def local_training(self, task_id):
        self.model.train()
        self.task_id = task_id
        if self.scheduler is None:
            lr = self.args["lr"]
        else:
            lr = self.scheduler.get_last_lr()[0]
        
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, 
                                    momentum=self.args["momentum"],weight_decay=self.args["weight_decay"])

        if self.args["optimizer"] == "adam":
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, 
                                        weight_decay=self.args["weight_decay"])

        for epoch in range(self.args["local_epochs"]):
            if task_id == 0:
                self._update()
            else:
                self._finetune()

        return self.model.state_dict()

        
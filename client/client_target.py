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

class Client_target(object):
    def __init__(self, args, idx):
        self.args = args
        self.idx = idx
        self.device = args["device"][0]
        self.model = None
        self.old_model = None
        self.trainloader = None
        self.optimizer = None
        self._known_classes = 0
        self._cur_task = 0
        self.kd = 25

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
        # global print_flag
        iter_loader = enumerate(zip((self.trainloader), (self.syn_data_loader)))
        self.total_local = 0.0
        self.total_syn = 0.0
        for batch_idx, ((_, images, labels), syn_input) in iter_loader:
            images, labels, syn_input = images.to(self.device), labels.to(self.device), syn_input.to(self.device)
            start_point = 0 if self.args["classIL"] != "True" else self._known_classes 
            fake_targets = labels - self._known_classes
            output = self.model(images)["logits"]
            # for new tasks
            if self.args["lode"] == "True":
                rho = 0.01 / self._cur_task
                loss_ce = self.lode_loss(output, labels, rho)
            else:
                loss_ce = F.cross_entropy(output[:, start_point :], fake_targets)
            
            s_out = self.model(syn_input)["logits"]
            with torch.no_grad():
                t_out = self.old_model(syn_input.detach())["logits"]
                self.total_syn += syn_input.shape[0]
                self.total_local += images.shape[0]
            
            end_point = s_out.shape[-1] if self.args["classIL"] != "True" else self._known_classes
            # for old task
            loss_kd = _KD_loss(
                s_out[:, :end_point],   # logits on previous tasks
                t_out.detach(),
                2,
            )
            loss = loss_ce + self.kd * loss_kd 
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def local_training(self, task_id):
        self.model.train()
        if self.scheduler is None:
            lr = self.args["lr"]
        else:
            lr = self.scheduler.get_last_lr()[0]
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, 
                                       momentum=self.args["momentum"],weight_decay=self.args["weight_decay"])
        for epoch in range(self.args["local_epochs"]):
            if self._cur_task == 0:
                self._update()
            else:
                self.old_model.to(self.device)
                self.old_model.eval()
                self._finetune()

        return self.model.state_dict()


    def lode_loss(self, output, labels, rho):
        
        total_classes = output.shape[1]
        old_classes = list(range(self._known_classes))
        new_classes = list(range(self._known_classes, total_classes))
        fake_targets = labels - self._known_classes
        
        # --- intra (new-only) CE  ---
        # mask = torch.zeros_like(output)
        # mask[:, new_classes] = 1
        # logits_new = output - (1 - mask) * 1e9
        loss_intra = F.cross_entropy(output[:, self._known_classes:], fake_targets)

        # --- inter (new vs old) ---
        s_new = torch.logsumexp(output[:, new_classes], dim=1)
        s_old = torch.logsumexp(output[:, old_classes], dim=1)
        # target = 1 means "new"
        inter_target = torch.ones_like(labels, dtype=torch.long, device=self.device)
        # CE on two logits [s_old, s_new]
        inter_logits = torch.stack([s_old, s_new], dim=1)
        loss_inter = F.cross_entropy(inter_logits, inter_target)

        # Final LODE loss
        loss = rho * loss_inter + loss_intra 
        return loss 
    
def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

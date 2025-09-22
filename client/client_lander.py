import sys
sys.path.append('../')
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
from models.generator import DataIter
import numpy as np

tau = 2

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
    
class Client_LANDER(object):
    def __init__(self, args,idx):
        self.args = args
        self.idx = idx
        self.device = args["device"][0]
        self.model = None
        self.trainloader = None
        self.optimizer = None
        self._known_classes = 0
        self._cur_task = 0
        self._total_classes = 0
        
        self.r = args['r']
        self.ltc = args['ltc']
        self.cur = args["cur"]
        self.pre = args["pre"]
        self.lr = args["lr"]
        self.label_emb = None
        
    def local_finetune(self, teacher, task_id, client_id, syn_data_loader):
        self.model.train()
        teacher.eval()
        total_loss = 0
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, 
                                       momentum=self.args["momentum"], weight_decay=self.args["weight_decay"])
        
        alpha = np.log2(self._total_classes / 2 + 1)
        beta = np.sqrt(self._known_classes / self._total_classes)
        cur = self.cur * (1 + 1 / alpha) / beta
        pre = self.pre * alpha * beta
        syn_data_iter = DataIter(syn_data_loader)
        
        for epoch in range(self.args["local_epochs"]):
            
            total_local = 0.0
            total_syn = 0.0
            total_ce_loss = 0
            total_f_loss = 0
            total_kd_loss = 0
            total_sf_loss = 0
            
            loss_ce_cur = 0
            c_loss_f = 0
            loss_kd = 0
            s_loss_f = 0

            check = 0
            for batch_idx, (_, images, labels) in enumerate(self.trainloader):
                check = 1
                # need to fix here 
                images, labels = images.to(self.device), labels.to(self.device)
                syn_input = syn_data_iter.next()
                # syn_input = images 
                syn_input = syn_input.to(self.device)
                start_point = 0 if self.args["classIL"] != "True" else self._known_classes
                fake_targets = labels - self._known_classes
                
                c_output_list = self.model(images) # output of real samples 
                c_output = c_output_list["logits"]
                c_feature = c_output_list["att"]
                c_target_f = self.label_emb[labels]
                c_loss_f = torch.relu(torch.nn.functional.mse_loss(c_feature, c_target_f.detach()) - self.r)

                # for new tasks
                # loss_ce_cur = F.cross_entropy(c_output[:, start_point:], fake_targets)

                if self.args["lode"] == "True":
                    rho = 0.005 / self._cur_task
                    loss_ce_cur = self.lode_loss(c_output, labels, rho)
                else:
                    loss_ce_cur = F.cross_entropy(c_output[:, start_point :], fake_targets)

                s_out_list = self.model(syn_input.detach())
                s_out = s_out_list["logits"]
                s_f = s_out_list["att"]

                with torch.no_grad():
                    t_out_list = teacher(syn_input.detach())
                    t_out = t_out_list["logits"]
                    t_f = t_out_list["att"]
                    total_syn += syn_input.shape[0]
                    total_local += images.shape[0]

                end_point =  s_out.shape[-1] if self.args["classIL"] != "True" else self._known_classes
                loss_kd = _KD_loss(s_out[:, :end_point], t_out.detach(), tau)
                s_loss_f = torch.nn.functional.mse_loss(s_f, t_f.detach())
                loss = cur * (loss_ce_cur + self.ltc * c_loss_f) + pre * (loss_kd + s_loss_f)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.detach()
                total_ce_loss += loss_ce_cur.detach()
                total_f_loss += c_loss_f.detach()
                total_kd_loss += loss_kd.detach()
                total_sf_loss += s_loss_f.detach()
            
            if check == 1:
                print("---task {} =>  CE: {}, F: {}. TKL: {}, TF: {}, T: {}".format(self._cur_task, total_ce_loss, total_f_loss, total_kd_loss, total_sf_loss, total_loss))
 
        return self.model.state_dict(), total_syn, total_local, total_loss

    def local_updating(self):
        self.model.train()
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, 
                                       momentum=self.args["momentum"],weight_decay=self.args["weight_decay"])
        total_loss = 0
        total_ce_loss = 0
        total_f_loss = 0

        for epoch in range(self.args["local_epochs"]):
            loss_ce = 0 
            loss_f = 0
            check = 0
            for batch_idx, (_, images, labels) in enumerate(self.trainloader):
                check = 1
                images, labels = images.to(self.device), labels.to(self.device)
                output_list = self.model(images)
                output = output_list["logits"].to(self.device)
                feature = output_list["att"].to(self.device)
                target_f = self.label_emb[labels]
                # test 
                loss_f = torch.relu(torch.nn.functional.mse_loss(feature, target_f.detach()) - self.r)
                loss_ce = F.cross_entropy(output, labels)
                loss = loss_ce + self.ltc * loss_f
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.detach()
                total_ce_loss += loss_ce.detach()
                total_f_loss += loss_f.detach()

        if self.trainloader is not None and check == 1:
            print("---task {} =>  CE: {}, F: {} T: {}".
              format(self._cur_task, total_ce_loss, total_f_loss, total_loss))
        
        return self.model.state_dict(), total_loss
        

    def lode_loss(self, output, labels, rho):
        
        total_classes = output.shape[1]
        old_classes = list(range(self._known_classes))
        new_classes = list(range(self._known_classes, total_classes))

        # --- intra (new-only) CE (masking; matches your reference) ---
        mask = torch.zeros_like(output)
        mask[:, new_classes] = 1
        logits_new = output - (1 - mask) * 1e9
        loss_intra = F.cross_entropy(logits_new, labels)

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

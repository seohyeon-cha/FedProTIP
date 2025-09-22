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
import numpy as np
from  models.alexnet import compute_conv_output_size
from collections import defaultdict

class Client_FedProTIP(object):
    def __init__(self, args,idx):
        self.args = args
        self.idx = idx
        self.device = args["device"][0]
        self.model = None
        self.trainloader = None
        self.optimizer = None
        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = 0
        self.space_mats = []
        self.references = []
        self.references_all = []
        self.orth_set = {}
        self.frozen_layer_names = []
        self.scheduler = None 

        self.base_set = []
        self.task_bases = []   # {layer_name: [Q^(1)_l, Q^(2)_l, ...]}
        self.sim_mats   = np.zeros((1, 1), dtype=np.float64)
        
    def _update(self):
        for batch_idx, (_, images, labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)["logits"].to(self.device)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            self.optimizer.step()
        return 
    
    def _finetune(self, proj_mats):
        for batch_idx, (_, images, labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            fake_targets = labels - self._known_classes
            output = self.model(images)["logits"].to(self.device)
            start_point = 0 if self.args["classIL"] != "True" else self._known_classes
            loss = F.cross_entropy(output[:, start_point:], fake_targets)
            loss.backward()
            
            if self.args["global_proj"] != "True":
                if 'vit' in self.args['net']:
                    kk = 0
                    for i in range(12):
                        for k, (m, params) in enumerate(self.model.vit.blocks[i].named_parameters()):
                            if ('attn' in m or 'mlp' in m) and 'weight' in m:
                                if params.grad is not None:
                                    if kk < len(proj_mats) and len(params.size())!=1:
                                        if params.requires_grad is True:
                                            sz =  params.grad.data.size(0)
                                            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1), proj_mats[kk]).view(params.size())
                                    
                                    elif (len(params.size()) == 1):
                                        if params.grad is None:
                                            continue
                                        params.grad.data.fill_(0)
                                kk += 1
                else:                    
                    kk = 0
                    for k, (m, params) in enumerate(self.model.convnet.named_parameters()):
                        if m not in self.frozen_layer_names:
                            if kk < len(proj_mats) and len(params.size())!=1:
                                if params.requires_grad is True:
                                    sz =  params.grad.data.size(0)
                                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1), proj_mats[kk]).view(params.size())
                                kk += 1
                            
                            elif (len(params.size()) == 1):
                                if params.grad is None:
                                    continue
                                params.grad.data.fill_(0)
            
            self.optimizer.step()
        return

    def local_training(self, ep_g, task_id, proj_mats):
        self.model.train()
        if self.scheduler is None:
            lr = self.args["lr"]
        else:
            lr = self.scheduler.get_last_lr()[0]
        
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, 
                                    momentum=self.args["momentum"],weight_decay=self.args["weight_decay"])

        for epoch in range(self.args["local_epochs"]):
            if task_id == 0:
                self._update()
            else:
                self._finetune(proj_mats)

        return self.model.state_dict()

    def update_GPM(self, activation_list, orth_set):
        threshold =  self.args["threshold"] + self._cur_task * 0.001
        print ('Threshold: ', threshold)
        print('Current Task: ', self._cur_task)
        
        first_client = 0
        for layer_name in activation_list.keys():
            if len(orth_set[layer_name]) == 0:
                first_client = 1

        if (self._cur_task == 0 and first_client == 1):  
        # After First Task 
            for layer_name in activation_list.keys(): 
                activation = activation_list[layer_name]
                if not layer_name == 'space':
                # if True:
                    select_number = min(activation.shape[1], 512) 
                    if select_number < activation.shape[1]:
                        pool = set(range(activation.shape[1]))
                        selected = np.random.choice(tuple(pool), size=select_number, replace=False)
                        activation = activation[:, selected]

                act_hat = activation
                if not layer_name == 'space':
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-5)
                    sval_total = (S**2).sum()
                    sval_ratio = (S**2)/sval_total
                    r = np.sum(np.cumsum(sval_ratio) < threshold)
                    orth_set[layer_name] = U[:,0:r]
                else:
                    U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-5)
                    sval_total = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    if self.args['net'] == 'alexnet':
                        r =  np.sum(np.cumsum(sval_ratio) < 0.998)
                    elif 'resnet' in self.args['net'] and self.args["pretrained"] == "False":
                        r = 20
                    else:
                        r = 30 
                    
                    orth_set[layer_name] = U[:, 0:r]
        else:
            for layer_name in activation_list.keys():
                activation = activation_list[layer_name]
                if not layer_name == 'space':
                # if True:
                    select_number = min(activation.shape[1], 512) # sampling from batches 
                    if select_number < activation.shape[1]:
                        pool = set(range(activation.shape[1]))
                        selected = np.random.choice(tuple(pool), size=select_number, replace=False)
                        activation = activation[:, selected]

                if not layer_name == 'space':
                    U1,S1,Vh1 = np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    orth_set[layer_name] = np.array(orth_set[layer_name])
                    act_hat = activation - np.dot(np.dot(orth_set[layer_name], orth_set[layer_name].transpose()), activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total
                    accumulated_sval = (sval_total-sval_hat)/sval_total
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating GPM for layer: ' + layer_name)
                        continue
                    
                    # update GPM
                    Ui=np.hstack((orth_set[layer_name], U[:,0:r]))
                    Ui = self.GramSchmidt(Ui)
                    if Ui.shape[1] > Ui.shape[0] :
                        orth_set[layer_name] = Ui[:, 0:Ui.shape[0]]
                    else:
                        orth_set[layer_name] = Ui
                else:
                    U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1 ** 2).sum()
                    # Projected Representation (Eq-8)
                    orth_set[layer_name] = np.array(orth_set[layer_name])
                    act_hat = activation - np.dot(np.dot(orth_set[layer_name], orth_set[layer_name].transpose()), activation)
                    U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                    
                    # criteria (Eq-9)
                    sval_hat = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    accumulated_sval = (sval_total - sval_hat) / sval_total
                    

                    if self._cur_task == 0:
                        r = int(20 / (self.args["n_clients"] - 1))
                    else:
                        r = int(20 / self.args["n_clients"])

                    # Model from scratch 
                    if 'resnet' in self.args['net'] and self.args["pretrained"] == "False":
                        if self._cur_task == 0:
                            r = int(20 / (self.args["n_clients"] - 1))
                        else:
                            r = int(20 / self.args["n_clients"])

                    # update GPM
                    Ui = np.hstack((orth_set[layer_name], U[:, 0:r]))
                    Ui = self.GramSchmidt(Ui)
                    if Ui.shape[1] > Ui.shape[0]:
                        orth_set[layer_name] = Ui[:, 0:Ui.shape[0]]
                    else:
                        orth_set[layer_name] = Ui
        print('-' * 40)
        print('Gradient Constraints Summary')
        print('-' * 40)
        i = 0
        for key in orth_set:
            if key not in self.frozen_layer_names:
                print('Layer {} : {}/{}'.format(key, orth_set[key].shape[1], orth_set[key].shape[0]))
            else:
                print(f'Layer {key} is frozen')
            i += 1

        print('-' * 40)
        return orth_set
    
    def GramSchmidt(self, matrix):
        Q, _ = np.linalg.qr(matrix)
        return Q

    def _center_gram(self, K):
        """Center a Gram matrix."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def linear_cka(self, X, Y):
        """
        Compute linear CKA between two activation matrices.
        X: n×d, Y: n×d
        Returns similarity in [0,1].
        """
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        K = X @ X.T
        L = Y @ Y.T

        Kc = self._center_gram(K)
        Lc = self._center_gram(L)

        hsic = np.sum(Kc * Lc)
        norm_x = np.sqrt(np.sum(Kc ** 2))
        norm_y = np.sqrt(np.sum(Lc ** 2))
        return float(hsic / (norm_x * norm_y + 1e-12))


    def collect_activations_alexnet(self):
        layer_names = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'fc1.weight', 'fc2.weight', 'space']

        self.model.eval()
        activation = {}
        for key in layer_names:
            activation[key] = []
            
        for batch_idx, (_,images, targets) in enumerate(self.trainloader):
            images = images.to(self.device)
            _ = self.model(images)
            act_list = [self.model.convnet.act['conv1.weight'], self.model.convnet.act['conv2.weight'], self.model.convnet.act['conv3.weight'],
                       self.model.convnet.act['fc1.weight'], self.model.convnet.act['fc2.weight'], self.model.convnet.act['space']]
            
            for j, key in enumerate(layer_names):
                if key == 'space':
                    act = act_list[j].detach().cpu().numpy()
                    activation[key].append(np.dot(act.transpose(), act))
                else:
                    activation[key].append(act_list[j].detach().cpu())
            
            if batch_idx >= self.args["n_batches"] -1:
                break

        for name in activation.keys():
            if name == 'space':
                # activation[name] = np.mean(activation[name], axis=0) # average ver batches 
                self.space_mats.append(activation[name])
                activation[name] = np.mean(activation[name], axis=0)
            else:
                activation[name] = torch.concat(activation[name], dim=0).numpy()
                
        # bsz = min(len(self.trainloader.dataset), 125) 
        bsz = min(len(self.trainloader.dataset), int(self.args["n_batches"])*self.args["batch_size"])
        for i in range(len(layer_names)):
            layer_name = layer_names[i]
            k=0
            if i < 3: 
                ksz = self.model.convnet.ksize[i]
                s = compute_conv_output_size(self.model.convnet.map[i],self.model.convnet.ksize[i])
                mat = np.zeros((self.model.convnet.ksize[i]*self.model.convnet.ksize[i]*self.model.convnet.in_channel[i], s*s*bsz))
                act = activation[layer_name]
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) #take each vector 
                            k +=1
                activation[layer_name] = mat

            else:
                act = activation[layer_name]
                if not layer_name == 'space':
                    activation[layer_name] = act[0:bsz].transpose()

        return activation


    def collect_activations_resnet(self):
        layer_names = ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', \
                           'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight','layer2.0.downsample.0.weight', \
                           'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', \
                           'layer3.0.downsample.0.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', \
                           'layer4.0.conv2.weight', 'layer4.0.downsample.0.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight', 'space']
        
        if 'cifar' in self.args["dataset"]: 
            map_list    = [32, 32,32,32,32, 32,16,32,16,16, 16,8,16,8,8, 8,4,8,4,4] 
            stride_list = [1, 1,1,1,1, 2,1,2,1,1, 2,1,2,1,1, 2,1,2,1,1]  
        elif self.args["img_size"] == 64:
            map_list    = [64, 64,64,64,64, 64,32,64,32,32, 32,16,32,16,16, 16,8,16,8,8]
            stride_list = [1, 1,1,1,1, 2,1,2,1,1, 2,1,2,1,1, 2,1,2,1,1]       
        elif 'cubs' in self.args["dataset"]:
            map_list = [128, 32,32,32,32, 32,16,32,16,16, 16,8,16,8,8, 8,4,8,4,4] 
            stride_list = [2, 1,1,1,1, 2,1,2,1,1, 2,1,2,1,1, 2,1,2,1,1]  
        elif self.args["img_size"] == 224:
            map_list    = [224, 56,56,56,56, 56,28,56,28,28, 28,14,28,14,14, 14,7,14,7,7]
            stride_list = [2, 1,1,1,1, 2,1,2,1,1, 2,1,2,1,1, 2,1,2,1,1]  
        
        if "reduced" in self.args["net"]:
            in_channel  = [3, 16,16,16,16, 16,32,16,32,32, 32,64,32,64,64, 64,128,64,128,128] 
        elif self.args["net"] == "resnet18":
            in_channel  = [3, 64,64,64,64, 64,128,64,128,128, 128,256,128,256,256, 256,512,256,512,512]
        else:
            in_channel  = [3, 32,32,32,32, 32,64,32,64,64, 64,128,64,128,128, 128,256,128,256,256]   
            

        self.model.eval()
        activation = {}
        for key in [key for key in layer_names if key not in self.frozen_layer_names]:
            activation[key] = []
        
        bsz = min(len(self.trainloader.dataset), int(self.args["n_batches"])*self.args["batch_size"])  
        for batch_idx, (_,images, _) in enumerate(self.trainloader):
            images = images.to(self.device)
            _ = self.model(images)
            act_list = [self.model.convnet.act['conv_in'], 
            self.model.convnet.layer1[0].act['conv_0'], self.model.convnet.layer1[0].act['conv_1'], self.model.convnet.layer1[1].act['conv_0'],
            self.model.convnet.layer1[1].act['conv_1'], self.model.convnet.layer2[0].act['conv_0'], self.model.convnet.layer2[0].act['conv_1'], 
            self.model.convnet.layer2[0].act['conv_0'], self.model.convnet.layer2[1].act['conv_0'], self.model.convnet.layer2[1].act['conv_1'], 
            self.model.convnet.layer3[0].act['conv_0'], self.model.convnet.layer3[0].act['conv_1'], self.model.convnet.layer3[0].act['conv_0'], 
            self.model.convnet.layer3[1].act['conv_0'], self.model.convnet.layer3[1].act['conv_1'], self.model.convnet.layer4[0].act['conv_0'],
            self.model.convnet.layer4[0].act['conv_1'], self.model.convnet.layer4[0].act['conv_0'], self.model.convnet.layer4[1].act['conv_0'], 
            self.model.convnet.layer4[1].act['conv_1'], self.model.convnet.act['space']]
            
            for j, key in enumerate(layer_names):
                if key not in self.frozen_layer_names:
                    if key == 'space':
                        act = act_list[j].detach().cpu()
                        # activation[key].append(torch.matmul(act.T, act))
                        activation[key].append(act_list[j].detach().cpu())
                    else:
                        activation[key].append(act_list[j].detach().cpu())

            if batch_idx >= self.args["n_batches"] -1:
                break

        for name in activation.keys():
            if name == 'space':
                self.space_mats.append(activation[name])
                activation[name] = torch.concat(activation[name], dim=0).numpy()
                # activation[name] = torch.stack(activation[name]).mean(dim=0) # average ver batches 
            else:
                activation[name] = torch.cat(activation[name], dim=0)
                if "downsample" not in name:
                    activation[name] = F.pad(activation[name], (1, 1, 1, 1), "constant", 0)
                
        
        bsz = min(len(self.trainloader.dataset), int(self.args["n_batches"])*self.args["batch_size"])  
        for i in range(len(layer_names)):
            layer_name = layer_names[i]
            if layer_name not in self.frozen_layer_names:
                if not layer_name == 'space':
                    k=0
                    if 'conv1.weight' == layer_name and "cifar" not in self.args["dataset"] and self.args["img_size"] >= 128: # different conv1 for cifar100 and others
                        ksz = 7
                        pad = 3 
                    else:
                        ksz = 3
                        pad = 1
                    st = stride_list[i]
                    act = activation[layer_name]
                    if "downsample" in layer_name:
                        ksz=1
                        pad=0
                    
                    s=self.model.convnet.compute_conv_output_size(map_list[i],ksz,stride=stride_list[i],padding=pad)
                    mat = torch.zeros((ksz*ksz*in_channel[i],s*s*bsz))
                    for kk in range(bsz):
                        for ii in range(s):
                            for jj in range(s):
                                act_ext = act[kk, :, st*ii:ksz+st*ii,st*jj:ksz+st*jj]
                                if (act_ext.shape[-1] == ksz) and (act_ext.shape[-2] == ksz):
                                    mat[:,k] = act_ext.reshape(-1) # take each vector 
                                    k +=1
                    activation[layer_name] = mat.numpy()
                else:
                    act = activation[layer_name]
                    activation[layer_name] = act[0:bsz].transpose()

        return activation

    
    def collect_activations_vit(self):
        layer_names = [
            name
            for i in range(12)
            for name in (
                f'block{i}.attn.qkv.weight',
                f'block{i}.attn.proj.weight',
                f'block{i}.mlp.fc2.weight',
                f'block{i}.mlp1.weight',
            )
        ] + ['space']
        self.model.eval()
        activation = {}
        for key in layer_names:
            activation[key] = []
        
        bsz = min(len(self.trainloader.dataset), int(self.args["n_batches"])*self.args["batch_size"])
        for batch_idx, (_,images, targets) in enumerate(self.trainloader):
            images = images.to(self.device)
            _ = self.model(images)
            act_list = []
            for i in range(12):
                blk = self.model.vit.blocks[i]
                act_list.extend([
                    blk.attn.act['attn_0'],
                    blk.attn.act['attn_1'],
                    blk.mlp.act['mlp_0'],
                    blk.mlp.act['mlp_1']
                ])
            act_list.append(self.model.vit.act['space'])
            
            for j, key in enumerate(layer_names):
                if key == 'space':
                    act = act_list[j].detach().cpu()
                    activation[key].append(act_list[j].detach().cpu())
                else:
                    act = act_list[j].detach().cpu().numpy()
                    if len(act.shape) == 2:
                        act = np.expand_dims(act, 1)
                    act = np.transpose(act, (2, 0, 1)).reshape(act.shape[-1], -1)
                    activation[key].append(act)

            if batch_idx >= self.args["n_batches"] -1:
                break

        for name in activation.keys():
            if name == 'space':
                self.space_mats.append(activation[name])
                arrs = [a.transpose() if isinstance(a, np.ndarray) else a.detach().cpu().numpy().transpose()
                            for a in activation["space"]]
                activation[name] = np.concatenate(arrs, axis=1) 
            else:
                activation[name] = np.concatenate(activation[name], axis=1) 
        return activation 
    
    
    def collect_activations(self, frozen_layer_names):
        self.frozen_layer_names = frozen_layer_names
        if 'resnet18' in self.args['net']:
            return self.collect_activations_resnet()
        if self.args['net'] == 'alexnet':
            return self.collect_activations_alexnet()
        if 'vit' in self.args['net']:
            return self.collect_activations_vit()


    def compute_references(self, space_idx, orth_set):
        self.references_all = []
        for task, act_batches in enumerate(self.space_mats):
            references = []
            for act in act_batches:
                if len(space_idx) > 2:
                    reference = []
                    for t in range(2, len(space_idx)):
                        start = space_idx[t - 1]
                        end = space_idx[t]
                        selected_bases = orth_set['space'][:, start:end]
                        inner_product = np.dot(act, np.dot(selected_bases, selected_bases.transpose()))
                        reference.append(np.linalg.norm(inner_product).item())
                    references.append(reference)
            self.references_all.append(references)


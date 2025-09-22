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

class Client_FOT(object):
    def __init__(self, args, idx):
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
        

    def _update(self):
        for batch_idx, (_, images, labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)["logits"].to(self.device)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            self.optimizer.step()

        return 
    
    def _finetune(self,proj_mats):
        for batch_idx, (_, images, labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            fake_targets = labels - self._known_classes
            start_point = 0 if self.args["classIL"] != "True" else self._known_classes
            output = self.model(images)["logits"].to(self.device)
            loss = F.cross_entropy(output[:, start_point :], fake_targets)
            loss.backward()
            self.optimizer.step()

        return


    def local_training(self, ep_g, task_id, proj_mats):
        self.model.train()
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args["lr"], 
                                    momentum=self.args["momentum"], weight_decay=self.args["weight_decay"])
        
        # for i, (name, param) in enumerate(self.model.named_parameters()):
        #     if param.requires_grad == False:
        #         print(f'Client model froze layer {i}: {name}')

        for epoch in range(self.args["local_epochs"]):
            if task_id == 0:
                self._update()
            else:
                 self._finetune(proj_mats)

        return self.model.state_dict()


    def collect_activations_resnet(self, orth_set):
        layer_names = ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', \
                           'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight','layer2.0.downsample.0.weight', \
                           'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', \
                           'layer3.0.downsample.0.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', \
                           'layer4.0.conv2.weight', 'layer4.0.downsample.0.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight', 'space']
        
        if 'cifar' in self.args["dataset"]: 
            map_list    = [32, 32,32,32,32, 32,16,32,16,16, 16,8,16,8,8, 8,4,8,4,4] 
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
                        activation[key].append(torch.matmul(act.T, act))
                    else:
                        activation[key].append(act_list[j].detach().cpu())

            if batch_idx >= self.args["n_batches"] -1:
                break

        for name in activation.keys():
            if name == 'space':
                self.space_mats.append(activation[name])
                activation[name] = torch.stack(activation[name]).mean(dim=0) # average ver batches 
            else:
                activation[name] = torch.cat(activation[name], dim=0)
                if "downsample" not in name:
                    activation[name] = F.pad(activation[name], (1, 1, 1, 1), "constant", 0)

        ratio_dict = {}
        bsz_dict = {}       
        
        for i in range(len(layer_names)):
            layer_name = layer_names[i]
            if layer_name not in self.frozen_layer_names:
                if not layer_name == 'space':
                    k=0
                    if 'conv1.weight' == layer_name and "cifar" not in self.args["dataset"]: # different conv1 for cifar100 and others
                        ksz = 7
                        pad = 3 
                    else:
                        ksz = 3
                        pad = 1
                    st = stride_list[i]
                    act = activation[layer_name]
                    if "downsample" in layer_name:
                        ksz = 1
                        pad = 0
                    
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
                    activation[layer_name] = activation[layer_name].numpy()
                
                ratio = 1
                if len(orth_set[layer_name]) != 0:
                    Uf = torch.Tensor(np.dot(orth_set[layer_name], orth_set[layer_name].transpose()))
                    sz = mat.size(0)
                    projected = torch.mm(Uf, mat.view(sz,-1)).view(mat.size())
                    remaining = mat - projected
                    rem_norm = torch.norm(remaining)
                    orj_norm = torch.norm(mat)
                    ratio = (rem_norm / orj_norm).cpu()
                ratio_dict[layer_name] = ratio
                bsz_dict[layer_name] = bsz 
        return activation, ratio_dict, bsz_dict


    def collect_activations_alexnet(self, orth_set):
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

        ratio_dict = {}
        bsz_dict = {}        
        # bsz = min(len(self.trainloader.dataset), 125) # on purpose? it was line below 
        # bsz = min(len(self.trainloader.dataset), self.args["n_batches"]*self.args["batch_size"])
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
                if isinstance(activation[layer_name], torch.Tensor):
                    act = activation[layer_name].detach().cpu().numpy()
                else:
                    act = activation[layer_name]
                if not layer_name == 'space':
                    activation[layer_name] = act[0:bsz].transpose()

            ratio = 1
            if len(orth_set[layer_name]) != 0:
                Uf = torch.Tensor(np.dot(orth_set[layer_name], orth_set[layer_name].transpose()))
                sz = activation[layer_name].shape[0]
                mat = torch.from_numpy(activation[layer_name]).float()
                projected = torch.mm(Uf, mat.view(sz,-1)).view(mat.size())
                remaining = mat - projected
                rem_norm = torch.norm(remaining)
                orj_norm = torch.norm(mat)
                ratio = (rem_norm / orj_norm).cpu()
                
            ratio_dict[layer_name] = ratio
            bsz_dict[layer_name] = bsz 

        return activation, ratio_dict, bsz_dict

    def collect_activations_vit(self, orth_set):
        layer_names = [
            name
            for i in range(6, 12) # frozen 0-5
            for name in (
                f'block{i}.attn.qkv.weight',
                f'block{i}.attn.proj.weight',
                f'block{i}.mlp.fc2.weight',
                f'block{i}.mlp1.weight',
            )
        ] 

        self.model.eval()
        activation = {}
        for key in layer_names:
            activation[key] = []
        
        bsz = min(len(self.trainloader.dataset), int(self.args["n_batches"])*self.args["batch_size"])
        for batch_idx, (_,images, targets) in enumerate(self.trainloader):
            images = images.to(self.device)
            _ = self.model(images)
            act_list = []
            for i in range(6, 12):
                blk = self.model.vit.blocks[i]
                act_list.extend([
                    blk.attn.act['attn_0'],
                    blk.attn.act['attn_1'],
                    blk.mlp.act['mlp_0'],
                    blk.mlp.act['mlp_1']
                ])
            
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

            if batch_idx >= self.args["n_batches"] - 1:
                break
        
        ratio_dict = {}
        bsz_dict = {}
        for name in activation.keys():
            if name == 'space':
                continue
            else:
                activation[name] = np.concatenate(activation[name], axis=1) 

            ratio = 1
            if len(orth_set[name]) != 0:
                Uf = torch.Tensor(np.dot(orth_set[name], orth_set[name].transpose()))
                sz = activation[name].shape[0]
                mat = torch.from_numpy(activation[name]).float()
                projected = torch.mm(Uf, mat.view(sz,-1)).view(mat.size())
                remaining = mat - projected
                rem_norm = torch.norm(remaining)
                orj_norm = torch.norm(mat)
                ratio = (rem_norm / orj_norm).cpu()
            
            ratio_dict[name] = ratio
            bsz_dict[name] = bsz 

        return activation, ratio_dict, bsz_dict 
    
    def collect_activations(self, frozen_layer_names, orth_set):
        self.frozen_layer_names = frozen_layer_names
        if 'resnet18' in self.args['net']:
            return self.collect_activations_resnet(orth_set)
        if self.args['net'] == 'alexnet':
            return self.collect_activations_alexnet(orth_set)
        if 'vit' in self.args['net']:
            return self.collect_activations_vit(orth_set)

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
                        inner_product = np.dot(act, selected_bases)
                        reference.append(np.linalg.norm(np.mean(inner_product, axis=1)).item())
                    references.append(reference)
            self.references_all.append(references)
        # print('references:')
        # print(self.references_all)
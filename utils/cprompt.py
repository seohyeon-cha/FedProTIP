import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from models.vit import VisionTransformer
import numpy as np
import math
import copy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def tensor_prompt(a, b, c=None, d=None, ortho=False, same=False):
    if same == False:
        if c is None and d is None:
            p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
        elif d is None:
            p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(a,b,c,d), requires_grad=True)
        if ortho:
            nn.init.orthogonal_(p)
        else:
            nn.init.uniform_(p)
    else:
        if c is None and d is None:
            p = torch.nn.Parameter(torch.FloatTensor(1,b).uniform_(0, 1).repeat(a, 1), requires_grad=True)
        elif d is None:
            p = torch.nn.Parameter(torch.FloatTensor(1,b,c).uniform_(0, 1).repeat(a, 1, 1), requires_grad=True)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(1,b,c,d).uniform_(0, 1).repeat(a, 1, 1, 1), requires_grad=True)
    return p    


class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, device='cuda:0', clients_local=10, num_clients = 10, args=None):
        print(" in CODA prompt")
        super().__init__()
        
        self.args = args
        self.task_id = 0
        self.max_classes = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.counter = 0
        self.next_task_locs= None
        self.device = device
        self.task_count_f = 0
        self.clients_local = clients_local
        self.num_clients = num_clients
        self.client_index = -1
        self.trained_task_id = None
        self.not_trained_task_id = None
        self.client_learned_global_task_id = None
        self.ep_g = None

        # e prompt init
        for e in self.e_layers:
            e_l = self.e_p_length

            #baseline
            p1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, e_l, emb_d, same=True) #TODO: 50 * 5 * 8 * 768
            k1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d, same=True)
            a1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d, same=True)

            setattr(self, f'e_p_share_{e}',p1)
            setattr(self, f'e_k_share_{e}',k1)
            setattr(self, f'e_a_share_{e}',a1)

        self.weight = torch.eye(int(self.args.numclass/self.args.class_per_task), device=self.device)
        self.weight_c = torch.eye(int(self.args.numclass/self.args.class_per_task), device=self.device)
        self.fc_weight = torch.eye(self.args.numclass, device=self.device)
    
    
    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_task_number = prompt_param[0] #TODO: 50
        self.e_pool_size_0 = prompt_param[1] #TODO: 5
        self.e_pool_size_1 = prompt_param[2] #TODO: 5
        self.e_p_length = prompt_param[3] #TODO: 8

        # prompt locations
        self.e_layers = [0, 1, 2, 3, 4]

        # location of ortho penalty
        self.ortho_mu = prompt_param[4] 
        print("ortho_mu ", self.ortho_mu)
        
    def process_frequency(self, next_task_locs = None):
        
        self.task_count_f += 1
        for e in self.e_layers:
            K1 = getattr(self,f'e_k_share_{e}')
            A1 = getattr(self,f'e_a_share_{e}')
            P1 = getattr(self,f'e_p_share_{e}')
            k1 = self.gram_schmidt(K1)
            a1 = self.gram_schmidt(A1)
            p1 = self.gram_schmidt(P1)
            setattr(self, f'e_p_share_{e}',p1)
            setattr(self, f'e_k_share_{e}',k1)
            setattr(self, f'e_a_share_{e}',a1)
        

    def gram_schmidt(self, vv):
        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = (len(vv.shape) >= 3)
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0] * self.e_pool_size_1,-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        if "full" not in self.args.method:
            s = int(self.task_id * self.num_clients * self.e_pool_size_1)
            f = int((self.task_id + 1) * self.num_clients * self.e_pool_size_1)
        else:
            if self.task_id == 0:
                s = int(self.task_id * self.num_clients * self.e_pool_size_1)
                f = int((self.task_id + 1) * self.num_clients * self.e_pool_size_1)
            else:
                s = int((self.task_id + 49) * self.e_pool_size_1)
                f = int((self.task_id + 50) * self.e_pool_size_1)
        
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 
        
    def next_task_locs_f(self, next_task_locs = None):
        if next_task_locs:
            self.next_task_locs = next_task_locs

    def get_aqk(self, x_querry, l, x_block, trained_task_id=None, aq_k=None):
        e_valid = False

        if l in self.e_layers:
            #print(trained_task_id)
            e_valid = True
            B, C = x_querry.shape
            weight = self.weight.detach().clone()
            relative_trained_task_id = {0: [[0], [1, 3]], 1: [[1], [2, 4]], 2: [[2], [0, 5]]}
            
            if True:
                K_all = None
                A_all = None
                p_all = None
                
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                
                for i in trained_task_id:
                    trained_task_id_removed = copy.deepcopy(trained_task_id)
                    trained_task_id_removed.remove(i)
                    K_share = torch.cat((K1[i].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                    A_share = torch.cat((A1[i].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                    p_share = torch.cat((p1[i].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                    weight_share = torch.cat((weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed]), dim=0).unsqueeze(0)
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                    if K_all is None:
                        K_all = K_share
                        A_all = A_share
                        p_all = p_share 
                    else:
                        K_all = torch.cat((K_all, K_share), dim=0)
                        A_all = torch.cat((A_all, A_share), dim=0)
                        p_all = torch.cat((p_all, p_share), dim=0)
                p_all = p_all / len(trained_task_id)
                
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all) 
            n_K = nn.functional.normalize(K_all, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)

            if aq_k is None:
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            else:
                aq_k = aq_k[l]

            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            loss = 0
        else:
            loss = 0

        if e_valid:
            p_return = [Ek, Ev]
            mean_aq_k = torch.mean(aq_k, dim=0)
        else:
            p_return = None
            P_ = None
            indices_taskchoosing = None
            mean_aq_k = None
            
        return p_return, x_block, mean_aq_k

    def forward_sharedcodap(self, x_querry, l, x_block, train=False, task_id=None, aq_k=None, global_task_id=None):
        e_valid = False
        indices_taskchoosing = None
        same = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            if "full" not in self.args.method:
                client_global_task_id = self.task_id * self.num_clients + self.client_index
            else:
                if self.task_id == 0:
                    client_global_task_id = self.client_index
                else:
                    client_global_task_id = self.task_id + 49

            if client_global_task_id != global_task_id:
                K1 = getattr(self,f'e_k_share_{l}')[global_task_id].clone().detach()
                A1 = getattr(self,f'e_a_share_{l}')[global_task_id].clone().detach()
                p1 = getattr(self,f'e_p_share_{l}')[global_task_id].clone().detach()
            else:
                same = True
                K1 = getattr(self,f'e_k_share_{l}')[global_task_id]
                A1 = getattr(self,f'e_a_share_{l}')[global_task_id]
                p1 = getattr(self,f'e_p_share_{l}')[global_task_id]
            K_all = K1
            A_all = A1
            p_all = p1
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all)
            n_K = nn.functional.normalize(K_all, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all)
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
        
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None
        return p_return, x_block, same
    
    def forward(self, x_querry, l, x_block, train=False, task_id=None, aq_k=None):
        # e prompts
        e_valid = False
        indices_taskchoosing = None
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            weight = self.weight.detach().clone()
            weight = torch.zeros((int(self.args.numclass/self.args.class_per_task), int(self.args.numclass/self.args.class_per_task)), device=self.device)
            
            for i in range(int(self.args.numclass/self.args.class_per_task)):
                weight[i][i] = 1.0
            relative_trained_task_id = {0: [[0], [1, 3]], 1: [[1], [2, 4]], 2: [[2], [0, 5]]}

            if self.client_index == -1:
                K_all = None
                A_all = None
                p_all = None

                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')

                for i in self.trained_task_id:
                    trained_task_id_removed = copy.deepcopy(self.trained_task_id)
                    trained_task_id_removed.remove(i)
                    K_share = torch.cat((K1[i].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                    A_share = torch.cat((A1[i].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                    p_share = torch.cat((p1[i].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                    weight_share = torch.cat((weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed]), dim=0).unsqueeze(0)
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                    if K_all is None:
                        K_all = K_share
                        A_all = A_share
                        p_all = p_share 
                    else:
                        K_all = torch.cat((K_all, K_share), dim=0)
                        A_all = torch.cat((A_all, A_share), dim=0)
                        p_all = torch.cat((p_all, p_share), dim=0)
                p_all = p_all / len(self.trained_task_id)
            else:
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')

                if "full" not in self.args.method:
                    global_task_id = self.task_id * self.num_clients + self.client_index
                else:
                    if self.task_id == 0:
                        global_task_id = self.client_index
                    else:
                        global_task_id = self.task_id + 49
                trained_task_id_removed = copy.deepcopy(self.trained_task_id)
                #trained_task_id_removed_forclient = copy.deepcopy(self.client_learned_global_task_id)
                trained_task_id_removed.remove(global_task_id)
                #trained_task_id_removed_forclient.remove(global_task_id)
                
                K_all = None
                A_all = None
                p_all = None
                
                if train:
                    for i in self.trained_task_id:
                        if i != global_task_id:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            trained_task_id_removed_again.remove(i)
                            K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[i].detach().clone().unsqueeze(0),K1[trained_task_id_removed_again].detach().clone()), dim=0)
                            A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[i].detach().clone().unsqueeze(0),A1[trained_task_id_removed_again].detach().clone()), dim=0)
                            p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[i].detach().clone().unsqueeze(0),p1[trained_task_id_removed_again].detach().clone()), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                        else:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed_again].detach().clone()), dim=0)
                            A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed_again].detach().clone()), dim=0)
                            p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed_again].detach().clone()), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                    p_all = p_all / len(self.trained_task_id)
                    
                else:
                    for i in self.trained_task_id:
                        if i != global_task_id:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            trained_task_id_removed_again.remove(i)
                            K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[i].unsqueeze(0),K1[trained_task_id_removed_again]), dim=0)
                            A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[i].unsqueeze(0),A1[trained_task_id_removed_again]), dim=0)
                            p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[i].unsqueeze(0),p1[trained_task_id_removed_again]), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                        else:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed_again]), dim=0)
                            A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed_again]), dim=0)
                            p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed_again]), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                    p_all = p_all / len(self.trained_task_id)

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all) 

            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K_all, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            
            if aq_k is None:
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            else:
                aq_k = aq_k[l]
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all)
            #P_ = torch.einsum('bk,kld->bld', aq_k_all, p_all)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            loss = 0

            if train and self.ortho_mu > 0:
                if self.ortho_mu == 1:
                    loss = self.ortho_penalty(K)
                elif self.ortho_mu == 2:
                    loss = self.ortho_penalty(A)
                elif self.ortho_mu == 3:
                    loss = self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 4:
                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(A)
                elif self.ortho_mu == 5:
                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 6:
                    loss += self.ortho_penalty(A)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 7:
                    # print("using all ortho penalty")

                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(A)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
            #print(p_return)
        else:
            p_return = None
            P_ = None
            indices_taskchoosing = None

        # return
        if train:
            if aq_k is not None:
                return p_return, loss, x_block, P_, indices_taskchoosing, torch.mean(aq_k, dim=0)
            else:
                return p_return, loss, x_block, P_, indices_taskchoosing, None
        else:
            return p_return, 0, x_block, indices_taskchoosing


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import math
import copy
from torch.utils.data import DataLoader

from utils.inc_net import Increment_ViT
from utils.cprompt import CodaPrompt


class ViT_cprompt(nn.Module):
    def __init__(self, num_classes=10, pretrained=False, mode=1, prompt_flag=False, 
                 prompt_param=None, task_size=10, device='cuda:0', local_clients=10, num_clients=10, 
                 class_distribution=None, tasks_global=3, class_distribution_real=None, 
                 class_distribution_proportion=None, class_distribution_client_di=None, 
                 params=None, args=None):
        super(ViT_cprompt, self).__init__()

        # get last layer
        self.params = params 
        self.args = args
        self.fc = nn.Linear(512, num_classes, bias=True)
        self.numclass = num_classes
        self.total_class_list = list(range(self.numclass))
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.task_size = task_size
        self.client_index = -1
        self.class_distribution = class_distribution
        self.class_distribution_real = class_distribution_real
        self.class_distribution_proportion = class_distribution_proportion
        self.class_distribution_client_di = class_distribution_client_di
        self.client_class_min_output = []
        self.client_class_max_output = []
        self.global_class_max_output_previous = []
        self.client_class_min_output_not_contain_previous = []
        self.global_class_min_output_contain_previous = []
        self.global_class_min_output = []
        self.global_class_max_output = []
        self.ep_g = 0
        self.tasks_global = tasks_global
        self.learned_classes = []
        self.unlearned_classes = []
        self.device = device
        self.num_clients = num_clients
        self.current_class = []
        #self.initial_promptchoosing = {}

        # get feature encoder
        self.feat = Increment_ViT(args, pretrained=pretrained)
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none').cuda(self.device)

        # create prompting module
        if self.prompt_flag == 'codap' or self.prompt_flag == 'cprompt':
            self.prompt = CodaPrompt(768, task_size, prompt_param, device=device, clients_local=local_clients, num_clients=num_clients, args=self.args)
    
    
    def calculate_prompt_choosing(self, train_dataset, c, t, trained_task_id, current_trained_task_id, finished_task):
        with torch.no_grad():
            indices =[]
            for i in current_trained_task_id:
                indices.append(trained_task_id.index(i))
            choosing_class = {}
            classes = self.class_distribution[c][t]
            classes_real = self.class_distribution_real[c][t]
            classes_proportion = self.class_distribution_proportion[c][t]
            if self.class_distribution_client_di is not None:
                class_distribution_client_di = self.class_distribution_client_di[c][t]
            else:
                class_distribution_client_di = None
            mean_aqk_task = None

            for i in range(len(classes)):
                train_dataset.getTrainData([classes[i]], [], [], c, classes_real=[classes_real[i]], classes_proportion=classes_proportion, class_distribution_client_di=class_distribution_client_di)
                train_loader = DataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    batch_size=self.args.batch_size,
                                    num_workers=8,
                                    pin_memory=True)
                mean_aqk_class = None
                for step, (indexs, images, target) in enumerate(train_loader):
                    if isinstance(self.device, int):
                        images, target = images.cuda(self.device), target.cuda(self.device)
                    else:
                        images, target = images.cuda(), target.cuda()
                    
                    with torch.no_grad():
                        q, _, _, q_map = self.feat(images)
                        q = q[:,0,:]
                    mean_aqk_list = self.feat.get_aqk(images, prompt=self.prompt, client_index=c, q=q, task_id=t, trained_task_id = trained_task_id, finished_task=finished_task).unsqueeze(0)
                    #print(mean_aqk_list.size())
                    mean_aqk_list = mean_aqk_list.reshape(mean_aqk_list.shape[0], mean_aqk_list.shape[1], len(trained_task_id), -1)
                    mean_aqk_list = mean_aqk_list[:, :, indices, :]
                    mean_aqk_list = mean_aqk_list.reshape(mean_aqk_list.shape[0], mean_aqk_list.shape[1], -1)
                    #print(mean_aqk_list.size())
                    if mean_aqk_class is None:
                        mean_aqk_class = mean_aqk_list
                    else:
                        mean_aqk_class = torch.cat((mean_aqk_class, mean_aqk_list), dim=0)
                mean_aqk_class = torch.mean(mean_aqk_class, dim=0)
                choosing_class[classes[i]] = mean_aqk_class
                if mean_aqk_task is None:
                    mean_aqk_task = mean_aqk_class.unsqueeze(0)
                else:
                    mean_aqk_task = torch.cat((mean_aqk_task, mean_aqk_class.unsqueeze(0)), dim=0)
        return torch.mean(mean_aqk_task, dim=0), choosing_class   
        

    
    def updateweight_with_promptchoosing(self, clients_index, clients_index_push, old_client_0, train_dataset, new_task, task_id, models, global_trained_task_id, choosing, choosing_class, finished_task, finished_task_forchoosing, finished_class, global_task_id_real, class_real, args, ep_g):

        trained_task_id_previous = copy.deepcopy(global_trained_task_id)
        trained_task_id_current = copy.deepcopy(global_trained_task_id)
        
        if new_task:
            if task_id > 0:
                for c in clients_index:
                    global_task_id = task_id * self.prompt.num_clients + c
                    trained_task_id_current = sorted(list(trained_task_id_current + [global_task_id]))
            else:
                for c in clients_index:
                    global_task_id = 0 * self.prompt.num_clients + c
                    trained_task_id_current = sorted(list(trained_task_id_current + [global_task_id]))
        
        new_task_id = []
        for c in clients_index:
            if new_task:
                if task_id > 0:
                    previous_client_id = c
                    previous_task_id = models[c].real_task_id
                    previous_global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                    finished_task[previous_global_task_id] = trained_task_id_previous
                    
                    current_client_id = c
                    current_task_id = task_id
                    global_task_id = task_id * self.prompt.num_clients + c
                    finished_task[global_task_id] = trained_task_id_current
                    new_task_id.append(global_task_id)
                    
            else: 
                current_client_id = c
                current_task_id = models[c].real_task_id
                global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                finished_task[global_task_id] = global_trained_task_id
                new_task_id.append(global_task_id)

        for c in clients_index:
            if new_task:
                if task_id > 0:
                    previous_client_id = c
                    previous_task_id = models[c].real_task_id
                    previous_global_task_id = models[c].real_task_id * self.prompt.num_clients + c

                    if c in clients_index_push:
                        previous_choosing_, previous_choosing_class_ = self.calculate_prompt_choosing(train_dataset, previous_client_id, previous_task_id, global_trained_task_id, trained_task_id_previous, finished_task=finished_task)
                    else:
                        previous_choosing_, previous_choosing_class_ = models[c].model.calculate_prompt_choosing(train_dataset, previous_client_id, previous_task_id, global_trained_task_id, trained_task_id_previous, finished_task=finished_task)
                    choosing[previous_global_task_id] = previous_choosing_.detach().cpu()
                    
                    for cl in self.class_distribution[c][models[c].real_task_id]:
                        choosing_class[cl] = previous_choosing_class_[cl].detach().cpu()
                        finished_class[cl] = trained_task_id_previous

                current_client_id = c
                current_task_id = task_id
                global_task_id = task_id * self.prompt.num_clients + c

                if c in clients_index_push:
                    choosing_, choosing_class_ = self.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, trained_task_id_current, trained_task_id_current, finished_task=finished_task)
                else:
                    choosing_, choosing_class_ = models[c].model.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, trained_task_id_current, trained_task_id_current, finished_task=finished_task)
                choosing[global_task_id] = choosing_.detach().cpu()

                for cl in self.class_distribution[c][task_id]:
                    choosing_class[cl] = choosing_class_[cl].detach().cpu()
                    finished_class[cl] = trained_task_id_current

        if ep_g % args.tasks_global == 0:
            weight = None
            for t_1 in choosing.keys():
                weight_line = None
                for t_2 in choosing.keys():
                    if t_1 in range(10,15) and t_2 in range(10,15):
                        prompt_choosing_1 = choosing[t_1]
                        prompt_choosing_2 = choosing[t_2]
                        finished_task_1 = finished_task[t_1]
                        finished_task_2 = finished_task[t_2]
                        print(finished_task_1)
                        print(finished_task_2)
                            
                        prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], len(finished_task_1), -1)
                        prompt_choosing_1 = prompt_choosing_1[:, len(finished_task_1)-5:, :]
                        prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], -1)
                    
                        prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], len(finished_task_2), -1)
                        prompt_choosing_2 = prompt_choosing_2[:, len(finished_task_2)-5:, :]
                        prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], -1)
                    else:
                        prompt_choosing_1 = choosing[t_1]
                        prompt_choosing_2 = choosing[t_2]

                        finished_task_1 = finished_task[t_1]
                        finished_task_2 = finished_task[t_2]
                        if len(finished_task_1) > len(finished_task_2):
                            
                            indices = []
                            for i in finished_task_2:
                                indices.append(finished_task_1.index(i))
                            prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], len(finished_task_1), -1)
                            prompt_choosing_1 = prompt_choosing_1[:, indices, :]
                            prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], -1)
                        else:
                            indices = []
                            for i in finished_task_1:
                                indices.append(finished_task_2.index(i))
                            prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], len(finished_task_2), -1)
                            prompt_choosing_2 = prompt_choosing_2[:, indices, :]
                            prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], -1)
                    
                    prompt_choosing_1 = nn.functional.normalize(prompt_choosing_1, dim=1)
                    prompt_choosing_2 = nn.functional.normalize(prompt_choosing_2, dim=1)
                    similarity = torch.einsum('bd,bd->b', prompt_choosing_1, prompt_choosing_2)
                    weight_point = torch.mean(similarity, dim=0).unsqueeze(0)
                    
                    weight_point = weight_point**self.params['task_index']
                    
                    if weight_line is None:
                        weight_line = weight_point
                    else:
                        weight_line = torch.cat((weight_line, weight_point), dim=0)
                
                topk_for_task = len(trained_task_id_current)
                if topk_for_task > weight_line.shape[0]:
                    topk_for_task = weight_line.shape[0]
                _, idx = weight_line.topk(topk_for_task)

                line_choose = torch.ones(weight_line.shape)
                line_choose[idx] = 0
                weight_line = weight_line.masked_fill(line_choose.bool(), 0)
                #print(weight_line)
                weight_line = weight_line / weight_line.sum()
                if weight is None:
                    weight = weight_line.unsqueeze(0)
                else:
                    weight = torch.cat((weight, weight_line.unsqueeze(0)), dim=0)
            
            if "full" not in self.args.method:
                fc_weight = None
                for c_1 in choosing_class.keys():
                    fc_weight_line = None
                    for c_2 in choosing_class.keys():
                        prompt_choosing_1 = choosing_class[c_1]
                        prompt_choosing_2 = choosing_class[c_2]
                        finished_task_1 = finished_class[c_1]
                        finished_task_2 = finished_class[c_2]

                        if len(finished_task_1) > len(finished_task_2):
                            indices = []
                            for i in finished_task_2:
                                indices.append(finished_task_1.index(i))
                            prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], len(finished_task_1), -1)
                            prompt_choosing_1 = prompt_choosing_1[:, indices, :]
                            prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], -1)
                        else:
                            indices = []
                            for i in finished_task_1:
                                indices.append(finished_task_2.index(i))
                            prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], len(finished_task_2), -1)
                            prompt_choosing_2 = prompt_choosing_2[:, indices, :]
                            prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], -1)
                        
                        prompt_choosing_1 = nn.functional.normalize(prompt_choosing_1, dim=1)
                        prompt_choosing_2 = nn.functional.normalize(prompt_choosing_2, dim=1)
                        similarity = torch.einsum('bd,bd->b', prompt_choosing_1, prompt_choosing_2)
                        
                        if int(c_1 // self.args.class_per_task) == int(c_2 // self.args.class_per_task) and c_1 != c_2: 
                            fc_weight_point = torch.zeros(1)
                        else:
                            fc_weight_point = torch.mean(similarity, dim=0).unsqueeze(0)
                            fc_weight_point = fc_weight_point**self.params['class_index']

                        if fc_weight_line is None:
                            fc_weight_line = fc_weight_point
                        else:
                            fc_weight_line = torch.cat((fc_weight_line, fc_weight_point), dim=0)
                    
                    _, idx = fc_weight_line.topk(self.params['topk_for_class'])
                    line_choose = torch.ones(fc_weight_line.shape)
                    line_choose[idx] = 0
                    fc_weight_line = fc_weight_line.masked_fill(line_choose.bool(), 0)
                    
                    fc_weight_line = fc_weight_line / fc_weight_line.sum()
                    if fc_weight is None:
                        fc_weight = fc_weight_line.unsqueeze(0)
                    else:
                        fc_weight = torch.cat((fc_weight, fc_weight_line.unsqueeze(0)), dim=0)

            for i in range(len(list(choosing.keys()))):
                self.prompt.weight[list(choosing.keys()), list(choosing.keys())[i]] = torch.tensor(weight[:, i], device=self.device)
            self.prompt.weight = torch.tensor(self.prompt.weight, device=self.device)
            self.prompt.weight_c[new_task_id] = self.prompt.weight.clone()[new_task_id]
            if "full" not in self.args.method:
                for i in range(len(list(choosing_class.keys()))):
                    self.prompt.fc_weight[list(choosing_class.keys()), list(choosing_class.keys())[i]] = torch.tensor(fc_weight[:, i], device=self.device)
                self.prompt.fc_weight = torch.tensor(self.prompt.fc_weight, device=self.device)
            
            for c in clients_index:
                if c in old_client_0:
                    current_client_id = c
                    current_task_id = models[c].real_task_id
                    if "full" not in self.args.method:
                        global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                    else:
                        if models[c].real_task_id == 0:
                            global_task_id = c
                        else:
                            global_task_id = models[c].real_task_id + 49
                    _, idx = self.prompt.weight[global_task_id].topk(self.params['topk_for_task_selection'])
                    finished_task_forchoosing[global_task_id] = idx
                else:
                    current_client_id = c
                    current_task_id = task_id
                    if "full" not in self.args.method:
                        global_task_id = task_id * self.prompt.num_clients + c
                    else:
                        if task_id == 0:
                            global_task_id = c
                        else:
                            global_task_id = task_id + 49
                    _, idx = self.prompt.weight[global_task_id].topk(self.params['topk_for_task_selection'])
                    finished_task_forchoosing[global_task_id] = idx

        print(class_real)
        return choosing, choosing_class, finished_task, finished_task_forchoosing, finished_class, global_task_id_real, class_real

    def Incremental_learning(self, task_id):
        
        self.task_id = task_id
        self.prompt.task_id = self.task_id
        if "noortho" in self.args.method:
            pass
        else:
            self.prompt.process_frequency()

    def set_global_class_min_output(self, global_class_output, global_class_output_now):
        self.global_class_min_output = []
        self.global_class_min_output_contain_previous = []
        self.global_class_max_output_previous = self.global_class_max_output
        self.global_class_max_output = global_class_output
         
        for i in range(self.numclass):
            if i in global_class_output:
                continue
            else:
                self.global_class_min_output.append(i) 
        for i in range(self.numclass):
            if i in global_class_output_now:
                continue
            else:
                self.global_class_min_output_contain_previous.append(i)
        self.fc.global_class_min_output = self.global_class_min_output

    def set_client_class_min_output(self):
        client_class_output = self.current_class
        self.client_class_min_output = []
        self.client_class_min_output_not_contain_previous = []
        self.unlearned_classes = []
        self.client_class_max_output = client_class_output
        for i in range(self.numclass):
            if i in client_class_output:
               continue
            else:
                self.client_class_min_output.append(i)
        for i in range(self.numclass):
            if (i in client_class_output) or (i in self.global_class_max_output_previous):
               continue
            else:
                self.client_class_min_output_not_contain_previous.append(i)
        for i in range(self.numclass):
            if i in self.learned_classes:
               continue
            else:
                self.unlearned_classes.append(i)
           

    def forward(self, x, pen=False, train=False, aq_k=None, device=0, ova='none', client_learned_task_id=None, labels=None):
        
        #torch.autograd.set_detect_anomaly(True)
        if self.prompt is not None:
            
            with torch.no_grad():
                q, _, _, q_map = self.feat(x)
                q = q[:,0,:]
            
            if "classincremental" in self.args.method:
                if train:  
                    out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map, out_divide = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k, ep_g=self.ep_g, client_index=self.prompt.client_index)
                else:
                    out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map, out_divide = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k)
            else:
                if train:  
                    out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k, ep_g=self.ep_g, client_index=self.prompt.client_index)
                else:
                    out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k)
            #print(indices_taskchoosing)
            #out = out[:,0,:]
            if "classincremental" not in self.args.method:
                if "v2" in self.args.method:
                    out = out[:,3 * self.prompt.e_p_length,:]
                else:
                    out = out[:,0,:]
            else:
                if "v2" in self.args.method:
                    out = out[:,3 * self.prompt.e_p_length,:]
                    out_divide = out_divide[:,3 * self.prompt.e_p_length_2,:]
                else:
                    out = out[:,0,:]
                    out_divide = out_divide[:,0,:]
            
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]

        out = out.view(out.size(0), -1)
        pre_logits = out # for fedmoon
        
        weight = self.prompt.fc_weight.detach().clone()

        if not pen:
            if self.client_index == -1:
                out = self.fc(out, self.global_class_min_output, self.global_class_max_output)
            else:
                out = self.fc(out, self.client_class_min_output, self.client_class_max_output)
            
            if "classincremental" in self.args.method:
                out_divide = nn.functional.normalize(out_divide, dim=1)
                task_embedding = nn.functional.normalize(self.prompt.task_embedding, dim=1)
                out_divide = torch.mm(out_divide, task_embedding.transpose(0,1))

            if "classincremental" in self.args.method and train:
                client_task_min_output = sorted(list(set(list(range(self.prompt.e_task_number))) - set(self.prompt.client_learned_global_task_id)))
                global_task_id = self.prompt.task_id * self.prompt.num_clients + self.prompt.client_index
                global_task_id = self.prompt.global_task_id_real[global_task_id]
                trained_task_id_removed = sorted(set(self.prompt.client_learned_global_task_id)-set([global_task_id]))
                out_divide[:,client_task_min_output] = -float('inf')
                #out_divide_2 = F.softmax(out_divide, dim=1)
                out_divide_2 = out_divide
                if len(trained_task_id_removed) > 0:
                    #logits_loss_for_divide = self.criterion_fn(out_divide, torch.tensor(global_task_id, device=self.device).repeat(out_divide.shape[0])).mean()
                    logits_loss_for_divide = 0
                    repre_loss_for_divide = (1.0 - out_divide_2[:,global_task_id].mean() \
                         + 1.0 * ((out_divide_2[:,trained_task_id_removed] * out_divide_2[:,trained_task_id_removed]).mean()) + \
                            prompt_loss + \
                                0) * 1.0
                else:

                    logits_loss_for_divide = 0
                    repre_loss_for_divide = (1.0 - out_divide_2[:,global_task_id].mean() \
                         + \
                            prompt_loss + \
                                0) * 1.0

            elif "classincremental" in self.args.method and not train:
                if self.client_index == -1:
                    global_task_min_output = sorted(list(set(list(range(self.prompt.e_task_number))) - set(self.prompt.trained_task_id)))
                    out_divide[:,global_task_min_output] = -float('inf')
                else:
                    client_task_min_output = sorted(list(set(list(range(self.prompt.e_task_number))) - set(self.prompt.client_learned_global_task_id)))
                    out_divide[:,client_task_min_output] = -float('inf')
                    print(out_divide[:,self.prompt.client_learned_global_task_id])
                
            control_loss = 0
            if self.client_index == -1 and not train:
                if "classincremental" in self.args.method:
                    detect_task_id = torch.max(out_divide, dim=1)[1].squeeze()
                    if len(detect_task_id.shape) == 0:
                        detect_task_id = detect_task_id.unsqueeze(0)
                    if self.prompt.task_id == 1:
                        pass
                    for i in range(out.shape[0]):
                        detect_class_list = self.class_distribution[int(detect_task_id[i] % self.args.num_clients)][int(detect_task_id[i] // self.args.num_clients)]
                        sample_class_min_output = sorted(set(self.total_class_list)-set(detect_class_list))
                        out[i,sample_class_min_output] = -float('inf')

                else:
                    out[:,self.global_class_min_output] = -float('inf')

            elif not train:
                if "classincremental" in self.args.method:
                    detect_task_id = torch.max(out_divide, dim=1)[1].squeeze()
                    if len(detect_task_id.shape) == 0:
                        detect_task_id = detect_task_id.unsqueeze(0)
                    if self.prompt.task_id == 1:
                        pass
                    for i in range(out.shape[0]):
                        detect_class_list = self.class_distribution[int(detect_task_id[i] % self.args.num_clients)][int(detect_task_id[i] // self.args.num_clients)]
                        sample_class_min_output = sorted(set(self.total_class_list)-set(detect_class_list))
                        out[i,sample_class_min_output] = -float('inf')
                else:
                    out[:,self.client_class_min_output] = -float('inf')
            else:
                out[:,self.client_class_min_output] = -float('inf')
        if "classincremental" in self.args.method:
            if self.prompt is not None and train:
                return out, prompt_loss, pre_logits, prompt_client, control_loss, mean_aqk_list, q_map, out_map, logits_loss_for_divide, repre_loss_for_divide
            else:
                return out
        else:
            if self.prompt is not None and train:
                return out, prompt_loss, pre_logits, prompt_client, control_loss, mean_aqk_list, q_map, out_map
            else:
                return out
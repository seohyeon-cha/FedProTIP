import sys
sys.path.append('../')
from FedProTIP.client.client_fedprotip import Client_FedProTIP as Client
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from progress.bar import Bar
import numpy as np
from utils.inc_net import IncrementalNet, Increment_ViT
from utils.data_manager import DataManager, partition_data, DatasetSplit, average_weights, setup_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import copy
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

class Server_FedProTIP(object):
    def __init__(self,args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self.args = args
        self.seed = args["seed"]
        self.device = args["device"][0]
        self.wandb = args["wandb"]
        self.clients = []
        self.data_manager = DataManager(args["dataset"], args["class_shuffle"], args["seed"], args["increment"], args["increment"], args)
        pretrained = True if self.args["pretrained"]=="True" else False 
        if 'vit' in self.args["net"]:
            self.global_model = Increment_ViT(args, pretrained=pretrained).to(self.device)
        else:
            self.global_model = IncrementalNet(args, pretrained=pretrained).to(self.device)
        self.freeze_layers_flag = False

        for idx in range(args["n_clients"]):
            self.clients.append(Client(args, idx))
        
        self.class_order = torch.tensor(self.data_manager.get_class_order()).to(self.device)
        self.topk = 5
        self.each_task = args["increment"]
        self.save_dir = args["save_dir"]
        
        if 'resnet18' in args['net']:
            self.layer_names = ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', \
                           'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight','layer2.0.downsample.0.weight', \
                           'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', \
                           'layer3.0.downsample.0.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', \
                           'layer4.0.conv2.weight', 'layer4.0.downsample.0.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight', 'space']
        elif args['net'] == 'alexnet':
            self.layer_names = ['conv1.weight', 'conv2.weight','conv3.weight', 'fc1.weight', 'fc2.weight', 'space']
        elif 'vit' in args['net']:
            self.layer_names = [
                name for i in range(12) for name in (
                    f'block{i}.attn.qkv.weight',
                    f'block{i}.attn.proj.weight',
                    f'block{i}.mlp.fc2.weight',
                    f'block{i}.mlp1.weight',
                )
            ] + ['space']

        self.orth_set = {}
        for key in self.layer_names:
            self.orth_set[key] = []

        self.space_idx = [0]
        self.test_dataset = []
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
        cnn_curve_gt = {"top1": [], "top5": []}
        cnn_curve_with_pred = {"top1": [], "top5": []}
        cnn_curve_without_pred = {"top1": [], "top5": []}
        self.task_pred_acc = []
        acc_t_t_gt, acc_t_t_with_pred, acc_t_t_without_pred = [], [], []
        acc_t_T_gt, acc_t_T_with_pred, acc_t_T_without_pred = [], [], []
        self.acc_matrix_gt_true = []
        self.acc_matrix_gt_false_wpred = []
        self.acc_matrix_gt_false_wopred = []

        acc_matrix = np.zeros((self.args["n_tasks"], self.args["n_tasks"]+1))
        ft_list = []

        for task in range(self.data_manager.nb_tasks):
            self._cur_task += 1
            self._total_classes = self._known_classes + self.data_manager.get_task_size(self._cur_task)
            
            if self.args["classIL"] != "True" and task > 0:
                pass
            else:
                self.global_model.update_fc(self._cur_task, 1) # increment fc neurons  
            
            print("Learning on {}-{}".format(self._known_classes, self._total_classes))
            train_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train",mode="train")
            self.test_dataset.append(self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test", mode="test"))
            setup_seed(self.seed)
            
            # FL training
            user_groups,_ = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["n_clients"])
            prog_bar = tqdm(range(self.args["epochs"]))

            self.scheduler = None 
            proj_mats = []
            if task > 0:
                self.layer_names = [i for i in self.layer_names if i not in self.frozen_layer_names]
                for key in self.layer_names: 
                    if key == 'space':
                        continue
                    Uf = torch.Tensor(np.dot(self.orth_set[key], self.orth_set[key].transpose())).to(self.device)
                    proj_mats.append(Uf)

            for _, epoch in enumerate(prog_bar):
                local_weights = []
                m = max(int(self.args["frac"] * self.args["n_clients"]), 1)
                selected_clients = np.random.choice(range(self.args["n_clients"]), m, replace=False)
                for idx in selected_clients:
                    print("Fine-tuning on client: ", idx)
                    local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                                                    batch_size=self.args["batch_size"], shuffle=True, num_workers=1)
                    self.clients[idx].trainloader = local_train_loader
                    self.clients[idx].model = copy.deepcopy(self.global_model)
                    self.clients[idx].scheduler = self.scheduler 
                    self.clients[idx]._known_classes = self._known_classes
                    self.clients[idx]._total_classes = self._total_classes
                    self.clients[idx]._cur_task = self._cur_task    
                    w = self.clients[idx].local_training(epoch, self._cur_task, proj_mats)
                    local_weights.append(copy.deepcopy(w))

                # update global weights
                global_weights = average_weights(local_weights)
                self.global_model.load_state_dict(global_weights)

            # Freeze layers after learning the first task
            if task == 0 and not self.freeze_layers_flag and self.args["pretrained"] == "True":
                self.freeze_layers()
                self.freeze_layers_flag = True  # Ensure layers are frozen only once

            for idx in range(self.args["n_clients"]):
                if self.clients[idx].trainloader != None:
                    self.clients[idx].model = copy.deepcopy(self.global_model)
                    activation_list = self.clients[idx].collect_activations(self.frozen_layer_names) # trainloader not updated yet
                    self.orth_set = self.clients[idx].update_GPM(activation_list, self.orth_set)

            self.space_idx.append(self.orth_set['space'].shape[1])
            print(self.space_idx)

            if self.args['gt'] == 'False':
                for idx in range(self.args["n_clients"]):
                    if self.clients[idx].trainloader != None:
                        self.clients[idx].compute_references(self.space_idx, self.orth_set)
 
            # Evaluate scenarios
            self.args["gt"] = "True"
            cnn_accy_gt, _ = self.eval_task()
            acc_t_t_gt.append(cnn_accy_gt["grouped"]["new"])
            self.acc_matrix_gt_true.append(cnn_accy_gt["grouped"]["record"])

            self.args["gt"] = "False"
            self.args["use_pred_task"] = "True"
            cnn_accy_with_pred, _ = self.eval_task()
            acc_t_t_with_pred.append(cnn_accy_with_pred["grouped"]["new"])
            self.acc_matrix_gt_false_wpred.append(cnn_accy_with_pred["grouped"]["record"])

            self.args["use_pred_task"] = "False"
            cnn_accy_without_pred, _ = self.eval_task()
            acc_t_t_without_pred.append(cnn_accy_without_pred["grouped"]["new"])
            self.acc_matrix_gt_false_wopred.append(cnn_accy_without_pred["grouped"]["record"])
            
            if self.wandb:
                wandb.log({
                    "Task": task + 1,
                    "Accuracy (GT1)": cnn_accy_gt["top1"],
                    "Accuracy (GT0 w/ Prediction)": cnn_accy_with_pred["top1"],
                    "Accuracy (Gt0 w/o Prediction)": cnn_accy_without_pred["top1"]
                })

            self._known_classes = self._total_classes
            cnn_curve_gt["top1"].append(cnn_accy_gt["top1"])
            cnn_curve_with_pred["top1"].append(cnn_accy_with_pred["top1"])
            cnn_curve_without_pred["top1"].append(cnn_accy_without_pred["top1"])
            
            print("(GT 1) Grouped: {}".format(cnn_accy_gt["grouped"]))
            print("(GT 0 w/ pred) Grouped: {}".format(cnn_accy_with_pred["grouped"]))
            print("(GT 0 w/o pred) Grouped: {}".format(cnn_accy_without_pred["grouped"]))

            print("(GT 1) CNN top1 curve: {}".format(cnn_curve_gt["top1"]))
            print("(GT 0 w/ pred) CNN top1 curve: {}".format(cnn_curve_with_pred["top1"]))
            print("(GT 0 w/o pred) CNN top1 curve: {}".format(cnn_curve_without_pred["top1"]))
            
            record = np.array(cnn_accy_with_pred["grouped"]["record"])
            acc_matrix[task] = np.pad(record, (0, self.args["n_tasks"]+1 - len(record)), mode='constant')
            
            print('-'*50)
            
            # Simulation Results 
            print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
            print(acc_matrix)
            ft = - np.mean((acc_matrix[task, : self.args["n_tasks"]]-np.diag(acc_matrix[:, :self.args["n_tasks"]]))[:task+1]) 
            
            ft_list.append(ft)
            print ('FT: {:5.2f}%'.format(ft))
            print('-'*50)
            
            plt.figure()
            row_avg = np.sum(acc_matrix[task, :self.args["n_tasks"]])/(task+1)
            # Plots
            acc_matrix[task, -1] = row_avg
            array = acc_matrix
        
            n_tasks = self.args["n_tasks"]

            task_labels = [f"T{i+1}" for i in range(n_tasks)]
            df_cm = pd.DataFrame(array, index=task_labels, columns=task_labels + ["Avg Acc"])

            sn.heatmap(df_cm, annot=True, annot_kws={"size": n_tasks})
            # plt.savefig('acc_matrix_test_44.png')
            
        acc_t_T_gt = cnn_accy_gt["grouped"]["record"]
        acc_t_T_with_pred = cnn_accy_with_pred["grouped"]["record"]
        acc_t_T_without_pred = cnn_accy_without_pred["grouped"]["record"]

        print("******** FT (GT1) ********")
        self.compute_forgetting_from_matrix(self.acc_matrix_gt_true)
        print("******** FT (GT0 with Prediction) ********")
        self.compute_forgetting_from_matrix(self.acc_matrix_gt_false_wpred)
        print("******** FT (GT0 without Prediction) ********")
        self.compute_forgetting_from_matrix(self.acc_matrix_gt_false_wopred)

        print(acc_matrix)
        print("Forgetting Records:")
        print(ft_list)

    def _compute_accuracy(self):
        self.global_model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(self.test_loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.global_model(inputs)["logits"]
            predicts = torch.argmax(outputs, dim=1)
            correct += len(torch.where(predicts.cpu() == targets)[0])
            total += len(targets)
        return np.around(correct * 100 / total, decimals=2)

    def eval_task(self):
        y_pred_all = [] 
        y_true_all = []
        self.task_acc = 0
        self.task_count = 0
        
        evaluate = self._evaluate_domain if self.args["classIL"] != "True" else self._evaluate
        for testset in self.test_dataset:
            self.test_loader = DataLoader(testset, batch_size=self.args["test_batch_size"], shuffle=True)
            y_pred, y_true = self._eval_cnn(self.test_loader)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)

        y_pred_all, y_true_all = np.concatenate(y_pred_all), np.concatenate(y_true_all)
        cnn_accy = evaluate(y_pred_all, y_true_all)
        
        if self.args['gt'] == 'False' and self.args["use_pred_task"] == 'True':
            self.task_pred_acc.append(round(self.task_acc/self.task_count, 3))
            print("Prediction Accuracy: ", self.task_pred_acc)

        return cnn_accy, None

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
    
    def _eval_cnn(self, loader):
        self.global_model.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self.device)
                outputs = self.global_model(inputs)["logits"]
                
                if self.args["gt"] == "True":
                    for i in range(len(outputs)):
                        task_id = int(targets[i]//self.each_task)
                        outputs[i][0:task_id*self.each_task] = - float('inf')
                        outputs[i][(task_id+1)*self.each_task:] = - float('inf')  
                    targets = targets 
                
                elif self.args["gt"] == "False" and self.args["use_pred_task"] == "True":
                    if 'vit' in self.args['net']:
                        act = self.global_model.vit.act['space'].detach().cpu().numpy()
                    else:
                        act = self.global_model.convnet.act['space'].detach().cpu().numpy()
                    # act = np.dot(act.transpose(), act)

                    pred_task = 0
                    scores = []
                    for t in range(2, len(self.space_idx)):
                        start = self.space_idx[t - 1]
                        end = self.space_idx[t]
                        selected_bases = self.orth_set['space'][:, start:end]
                        inner_product = np.dot(act, np.dot(selected_bases, selected_bases.transpose()))
                        scores.append(np.linalg.norm(inner_product).item())
                        # inner_product = np.dot(act, selected_bases)
                        # scores.append(np.linalg.norm(np.mean(inner_product, axis=1)).item())
                    scores = np.array(scores)
                    scores_norm = scores / (np.linalg.norm(scores))
                    print(scores)
                    print(f"real tasks: {int(targets[0]/self.args['increment'])}")

                    similarity_all = []
                    votes = []
                    for client_idx, client in enumerate(self.clients):
                        if client.trainloader != None:
                            similarity_client = []
                            references_all = client.references_all
                            for task, references in enumerate(references_all):
                                similarity = []
                                for r in references:
                                    r_norm = r / (np.linalg.norm(r))
                                    s = np.dot(scores_norm, r_norm)
                                    similarity.append(s.tolist())
                                similarity_client.append(np.mean(similarity, axis=0))
                            if self._cur_task > 0:
                                votes.append(np.argmax(similarity_client))  # Task with the highest similarity
                            similarity_all.append(similarity_client)
                    similarity_all = np.mean(similarity_all, axis=0)
                        # print(f"similarity: {similarity}")
                    # similarity_all = np.mean(similarity_all, axis=0)
                    similarity_all = np.exp(similarity_all) / np.sum(np.exp(similarity_all))


                    if self._cur_task > 0:
                        print(f"similarity: {[round(sim, 4) for sim in similarity_all]}")
                        print(f"vote: {votes}")
                    
                    if self._cur_task == 1:  
                        mean_references_task_0 = []
                        mean_references_task_1 = []
                        for client in self.clients:
                            if client.trainloader != None:
                                mean_references_task_0.append(np.mean(client.references_all[0], axis=0))                          
                                mean_references_task_1.append(np.mean(client.references_all[1], axis=0))       

                        # Compute prediction based on aggregated references
                        if scores[0] < np.mean(mean_references_task_0, axis=0):
                            pred_task = 0
                        elif scores[0] > np.mean(mean_references_task_1, axis=0):
                            pred_task = 1
                        else:
                            pred_task = (
                                0 if np.abs(np.mean(mean_references_task_0, axis=0) - scores[0]) < np.abs(np.mean(mean_references_task_1, axis=0) - scores[0]) else 1
                            )
                        print(f'prediction: {pred_task}')
                        print("-"*80)

                    if self._cur_task > 1:
                        pred_task_1 = np.argmax(similarity_all) # check argmax
                        pred_task_2 = max(set(votes), key=votes.count)  # Majority vote
                        pred_task = pred_task_2
                        print(f'prediction by mean: {pred_task_1} | prediction by vote: {pred_task_2}')
                        print("-"*80)

                    if pred_task == int(targets[0]/self.args['increment']):
                        self.task_acc += 1

                    for i in range(len(outputs)):
                        task_id = pred_task
                        outputs[i][0:task_id*self.each_task] = -float('inf')
                        outputs[i][(task_id+1)*self.each_task:] = -float('inf') 
                    self.task_count += 1    
                
                predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
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

    def _eval_nme(self, loader, class_means):
        self.global_model.eval()
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



def assert_all_eval(model):
    bad = []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.Dropout)):
            if m.training:
                bad.append(name)
    if bad:
        print("[EVAL CHECK] Modules still in training mode:", bad)
    else:
        print("[EVAL CHECK] All BN/Dropout are in eval()")
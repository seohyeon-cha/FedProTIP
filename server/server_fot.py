import sys
sys.path.append('../')
from client.client_fot import Client_FOT as Client
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from progress.bar import Bar
import numpy as np
from utils.toolkit import count_parameters
from utils.inc_net import IncrementalNet, Increment_ViT
from utils.data_manager import DataManager, partition_data, DatasetSplit, average_weights, setup_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import copy
import wandb
import gc


class Server_FOT(object):
    def __init__(self,args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self.args = args
        self.seed = args["seed"]
        self.device = args["device"][0]
        self.wandb = args["wandb"]
        self.clients = []
        self.data_manager = DataManager(args["dataset"],args["class_shuffle"], args["seed"],args["increment"],args["increment"],args)
        pretrained = True if self.args["pretrained"]=="True" else False 
        if 'vit' in self.args["net"]:
            self.global_model = Increment_ViT(args, pretrained=pretrained).to(self.device)
        else:
            self.global_model = IncrementalNet(args, pretrained=pretrained).to(self.device)

        # for name, item in self.global_model.state_dict().items():
        #     print(name)
        # create clients
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
                name for i in range(6, 12) for name in (
                    f'block{i}.attn.qkv.weight',
                    f'block{i}.attn.proj.weight',
                    f'block{i}.mlp.fc2.weight',
                    f'block{i}.mlp1.weight',
                )
            ] 

        self.orth_set = {}
        for key in self.layer_names:
            self.orth_set[key] = []

        self.space_idx = [0]
        self.test_dataset = []
        self.frozen_layer_names = []
        self.freeze_layers_flag = False

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
        self.task_pred_acc = []
        self.acc_matrix_gt_true = []
        self.acc_matrix_gt_false = []

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

            for idx in range(self.args["n_clients"]):
                self.clients[idx].model = copy.deepcopy(self.global_model)

            proj_mats = []
            if task > 0:
                self.layer_names = [i for i in self.layer_names if i not in self.frozen_layer_names]
                for key in self.layer_names: 
                    if key == 'space':
                        continue
                    if isinstance(self.orth_set[key], np.ndarray) == False:
                        self.orth_set[key] = self.orth_set[key].detach().cpu().numpy()
                    Uf=torch.Tensor(np.dot(self.orth_set[key], self.orth_set[key].transpose())).to(self.device)
                    proj_mats.append(Uf)
            
            for _, epoch in enumerate(prog_bar):
                local_weights = []
                local_gradients = {}
                m = max(int(self.args["frac"] * self.args["n_clients"]), 1)
                selected_clients = np.random.choice(range(self.args["n_clients"]), m, replace=False)
                for idx in selected_clients:
                    print("Fine-tuning on client: ", idx)
                    local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                                                    batch_size=self.args["batch_size"], shuffle=True, num_workers=1)
                    self.clients[idx].trainloader = local_train_loader
                    self.clients[idx].model = copy.deepcopy(self.global_model)
                    self.clients[idx]._known_classes = self._known_classes
                    self.clients[idx]._total_classes = self._total_classes
                    self.clients[idx]._cur_task = self._cur_task    
                    w = self.clients[idx].local_training(epoch, self._cur_task, proj_mats)
                    local_weights.append(copy.deepcopy(w))

                    del local_train_loader
                    del self.clients[idx].model
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                self.global_model_old = copy.deepcopy(self.global_model)
                if 'vit' in self.args['net']:
                    global_weights_old = {}
                    for i in range(12): # 12 blocks for ViTB16
                        global_weights_old[i] = self.global_model_old.vit.blocks[i].state_dict()
                        # update global weights
                    global_weights = average_weights(local_weights)
                    self.global_model.load_state_dict(global_weights)
                    global_weights = {}
                    for i in range(6, 12):
                        global_weights[i] = self.global_model.vit.blocks[i].state_dict()
                    
                    if self._cur_task > 0:
                        kk = 0
                        for i in range(6, 12):
                            for k, (key, params) in enumerate(self.global_model.vit.blocks[i].named_parameters()):
                                if ('attn' in key or 'mlp' in key) and 'weight' in key:
                                    if params.grad is not None:
                                        if kk < len(proj_mats) and len(global_weights_old[i][key].shape) > 1:
                                            grad = global_weights_old[i][key] - global_weights[i][key]                            
                                            sz =  global_weights[key].size(0)
                                            grad = grad - torch.mm(grad.view(sz,-1), proj_mats[kk]).view(grad.size())
                                            
                                            global_weights[i][key] = global_weights_old[i][key] - grad
                                        elif len(params.size()) == 1:
                                            if params.grad is None:
                                                continue
                                            params.grad.data.fill_(0)
                                    kk += 1
                            self.global_model.vit.blocks[i].load_state_dict(global_weights[i])
                else:
                    global_weights_old = self.global_model_old.convnet.state_dict()
                    # update global weights
                    global_weights = average_weights(local_weights)
                    self.global_model.load_state_dict(global_weights)
                    global_weights = self.global_model.convnet.state_dict()

                    if self._cur_task > 0:
                        kk = 0
                        for k, (key, params) in enumerate(self.global_model.convnet.named_parameters()):
                            if key not in self.frozen_layer_names:
                                if kk < len(proj_mats) and len(global_weights_old[key].shape) > 1:
                                    grad = global_weights_old[key] - global_weights[key]                            
                                    sz =  global_weights[key].size(0)
                                    grad = grad - torch.mm(grad.view(sz,-1), proj_mats[kk]).view(grad.size())
                                    kk += 1
                                    global_weights[key] = global_weights_old[key] - grad
                                elif len(params.size()) == 1:
                                    if params.grad is None:
                                        continue
                        self.global_model.convnet.load_state_dict(global_weights)

                if (epoch+1) % 10 == 0:
                    acc_at_round, _ = self.eval_task()
                    if self.wandb:
                        wandb.log({
                            "Global round": (epoch+1) + task * self.args["epochs"],
                            "Mean task accuracy": acc_at_round["top1"] 
                        })

            # Freeze layers after learning the first task
            if task == 0 and not self.freeze_layers_flag and self.args["pretrained"] == "True":
                self.freeze_layers()
                self.freeze_layers_flag = True  # Ensure layers are frozen only once

            # Randomize activation of each client 
            act_lists = []
            ratio_lists = {}
            num_samples_lists = {}
            for idx in range(self.args["n_clients"]):
                if self.clients[idx].trainloader != None:
                    self.clients[idx].model = copy.deepcopy(self.global_model)
                    activation_list, ratio_dict, bsz_dict = self.clients[idx].collect_activations(self.frozen_layer_names, self.orth_set) # trainloader not updated yet
    
                    # Multiply activations by Gaussian random vectors
                    gaussian_random_vectors = {}
                    noisy_activations = {}
                    kk = 0
                    for name, activation in activation_list.items():
                        if name != 'space':
                            activation = torch.from_numpy(activation).float()
                            # project activation on orthogonal subspace
                            if task > 0 :
                                activation = activation - torch.mm(proj_mats[kk].cpu(), activation).view(activation.size())
                                kk += 1 
                            # Define the sampling dimension as 5 times the activation dimension
                            activation_dim = activation.shape[0]  # d_ℓ
                            sampling_dim = 5 * activation_dim      # s_ℓ = 5 * d_ℓ

                            # Generate Gaussian random vector with appropriate dimensions
                            random_vector = torch.normal(0, 1, size=(activation.shape[-1], sampling_dim))

                            # Project the activation using the Gaussian random vector
                            if name not in gaussian_random_vectors:
                                gaussian_random_vectors[name] = random_vector
                            noisy_activations[name] = torch.mm(activation, gaussian_random_vectors[name])

                        if idx == 0:
                            ratio_lists[name] = [ratio_dict[name]]
                            num_samples_lists[name] = [bsz_dict[name]]
                        else:
                            ratio_lists[name].append(ratio_dict[name])
                            num_samples_lists[name].append(bsz_dict[name])
                    
                    act_lists.append(noisy_activations)
                

            # Average randomized activations 
            avg_activation_list = {}
            for act_list in act_lists:
                for key, value in act_list.items():
                    avg_activation_list[key] = avg_activation_list.get(key, 0) + value
                    
            avg_activation_list = {key: value / len(act_lists) for key, value in avg_activation_list.items()}

            self.threshold_list = {}
            for key in avg_activation_list.keys():
                weights = np.array(num_samples_lists[key]) / np.sum(num_samples_lists[key])
                weighted_avg = np.sum(weights * np.array(ratio_lists[key]))
                org_eps = self.args["threshold"] + self._cur_task * 0.001
                new_eps = (weighted_avg - (1 - org_eps)) / weighted_avg
                self.threshold_list[key] = new_eps

            # Expand orthogonal set 
            self.orth_set = self.update_GPM(avg_activation_list, self.orth_set)

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
            
        acc_t_T_gt_true = cnn_accy_gt_true["grouped"]["record"]
        acc_t_T_gt_false = cnn_accy_gt_false["grouped"]["record"]

        # Compute forgetting
        print("Forgetting (gt=True):")
        self.compute_forgetting_from_matrix(self.acc_matrix_gt_true)
        print("Forgetting (gt=False):")
        self.compute_forgetting_from_matrix(self.acc_matrix_gt_false)

    
    def _compute_accuracy(self):
        self.global_model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(self.test_loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.global_model(inputs)["logits"]
            predicts = torch.argmax(outputs, dim=1)
            correct += torch.where(predicts.cpu() == targets).sum()
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
                    outputs[i][0:task_id*self.each_task] = - float('inf')
                    outputs[i][(task_id+1)*self.each_task:] = - float('inf')
                    
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

    def update_GPM(self, activation_list, orth_set):
        print('Current Task: ', self._cur_task)
        # After First Task 
        for layer_name in activation_list.keys():
            threshold = self.threshold_list[layer_name]
            print ('Threshold: ', threshold)
            activation = activation_list[layer_name]
            act_hat = activation
            if not layer_name == 'space':
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)
            else:
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                if self.args['net'] == 'alexnet':
                    r =  np.sum(np.cumsum(sval_ratio) < 0.998)
                else:
                    r = np.sum(np.cumsum(sval_ratio) < 0.995)
            
            if len(orth_set[layer_name]) == 0:
                Ui = U[:, 0:r]
            else:
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
            if key not in self.frozen_layer_names and key != 'space':
                print('Layer {} : {}/{}'.format(key, orth_set[key].shape[1], orth_set[key].shape[0]))
            else:
                print(f'Layer {key} is frozen')
            i += 1

        print('-' * 40)
        return orth_set

    def GramSchmidt(self, matrix):
        """Perform Gram-Schmidt orthogonalization."""
        Q, _ = np.linalg.qr(matrix)
        return Q

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

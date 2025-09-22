import sys
sys.path.append('../')
from client.client_lga import Client_base as Client
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as opAccuracyim
from progress.bar import Bar
import numpy as np
from PIL import Image 
from torchvision import transforms
from utils.toolkit import count_parameters
from utils.inc_net import IncrementalNet
from utils.proxy_net_glfc import LeNet, weights_init
from utils.data_manager import DataManager, partition_data, DatasetSplit, average_weights, setup_seed, DummyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from utils.proxy_server_lga import proxyServer
from utils.data_manager import pil_loader
import wandb

class Server_LGA(object):
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
        self.global_model = IncrementalNet(args, pretrained=pretrained).to(self.device)
        hidden = 37632 if "imagenet" in self.args["dataset"] else 768
        num_classes = args["increment"] if self.args["classIL"] != "True" else args["n_classes"]
        self.encode_model = LeNet(hidden=hidden, num_classes=num_classes)
        self.model_old = None
        self.proxy_lr = 2.0 

        # for LGA
        self.radius = 0

        # create clients
        for idx in range(args["n_clients"]):
            self.clients.append(Client(args, idx))
        self.class_order = torch.tensor(self.data_manager.get_class_order()).to(self.device)
        self.topk = 5
        self.each_task = args["increment"]
        self.save_dir = args["save_dir"]
        self.test_dataset = []
        self.freeze_layers_flag = False
        self.frozen_layer_names = []

    def freeze_layers(self):
        print("Freezing layers 1 and 2 of ResNet18.")
        for k, (name, param) in enumerate(self.global_model.convnet.named_parameters()):
            if k < 30: # block 1 (~14) / block 2 (~29) / block 3 (~44) / block 4 (~59)
                print(f'Freeze layer {k}: {name}')
                param.requires_grad = False
                self.frozen_layer_names.append(name)

    def train(self):
        prev_dataset, prev_user_groups = None, None 
        cnn_curve_gt_true = {"top1": [], "top5": []}
        cnn_curve_gt_false = {"top1": [], "top5": []}
        acc_t_t_gt_true, acc_t_t_gt_false = [], []
        acc_t_T_gt_true, acc_t_T_gt_false = [], []
        self.acc_matrix_gt_true = []
        self.acc_matrix_gt_false = []

        # initialize enocde model
        self.encode_model.apply(weights_init)

        for task in range(self.data_manager.nb_tasks):
            self._cur_task += 1
            self._total_classes = self._known_classes + self.data_manager.get_task_size(self._cur_task)
            
            print("Learning on {}-{}".format(self._known_classes, self._total_classes))
            train_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train",mode="train")
            self.test_dataset.append(self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test", mode="test"))
            setup_seed(self.seed)
            
            if task == 0:
                proxy_model = copy.deepcopy(self.global_model)
                proxy_server = proxyServer(self.args, self.proxy_lr, proxy_model, train_dataset.trsf)
            if task == 1:
                dataset_radius = copy.deepcopy(prev_dataset)
                self.transform = train_dataset.trsf
                radius_model = copy.deepcopy(self.global_model)
                self.compute_radius(radius_model, dataset_radius)
                proxy_server.best_perf = 0
            
            proxy_server.radius = self.radius
            proxy_server.encode_model = copy.deepcopy(self.encode_model) 
            proxy_server.model.update_fc(self._cur_task, 0)
            self.global_model.update_fc(self._cur_task, 0)
            proxy_server._known_classes = self._known_classes 
            
            setup_seed(self.seed)

            # FL training
            # we assume there are only old_client_1 (which trains on both memory and new data)
            user_groups,_ = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["n_clients"])
            prog_bar = tqdm(range(self.args["epochs"]))
            for _, epoch in enumerate(prog_bar):
                local_weights = []
                m = max(int(self.args["frac"] * self.args["n_clients"]), 1)
                selected_clients = np.random.choice(range(self.args["n_clients"]), m, replace=False)
                self.model_old = proxy_server.model_back() 
                pool_grad = []

                for idx in selected_clients:
                    print("Fine-tuning on client: ", idx)
                    local_trainset = DatasetSplit(train_dataset, user_groups[idx])
                    local_train_loader = DataLoader(local_trainset, 
                                                    batch_size=self.args["batch_size"], shuffle=True, num_workers=1)

                    if prev_user_groups != None:
                        prev_trainset = DatasetSplit(prev_dataset, prev_user_groups[idx])
                        self.clients[idx].prev_trainset = DummyDataset(prev_trainset.dataset.images[prev_user_groups[idx]], prev_trainset.dataset.labels[prev_user_groups[idx]], prev_trainset.dataset.trsf, use_path=True)
                        self.clients[idx].prev_transform = prev_trainset.dataset.trsf  
                    self.clients[idx].trainset = DummyDataset(local_trainset.dataset.images[user_groups[idx]], local_trainset.dataset.labels[user_groups[idx]], local_trainset.dataset.trsf, use_path=True)
                    self.clients[idx].curr_transform = local_trainset.dataset.trsf
                    self.clients[idx].trainloader = local_train_loader
                    self.clients[idx]._known_classes = self._known_classes
                    self.clients[idx]._total_classes = self._total_classes
                    self.clients[idx].task_id_old = task
                    self.clients[idx].model = copy.deepcopy(self.global_model)
                    self.clients[idx].new_task = False 
                    if epoch == 0:
                        self.clients[idx].encode_model = copy.deepcopy(self.encode_model)
                        self.clients[idx].new_task = True
                    self.clients[idx].proxy_lr = self.proxy_lr

                    # local update & exemplar add
                    w, proto_grad = self.clients[idx].local_training(self._cur_task, self.model_old)
                    local_weights.append(copy.deepcopy(w))
                    if proto_grad != None:
                        for grad_i in proto_grad:
                            pool_grad.append(grad_i) # update prototypes
                    del local_train_loader, w
                    torch.cuda.empty_cache()

                # update global weights
                global_weights = average_weights(local_weights)
                self.global_model.load_state_dict(global_weights)

                # proxy serer update 
                proxy_server.model.load_state_dict(global_weights) # copy updated weights
                print('pool grad length: ', len(pool_grad))
                proxy_server.ep_g = epoch
                proxy_server.dataloader(pool_grad)

                if (epoch+1) % 10 == 0:
                    acc_at_round, _ = self.eval_task()
                    if self.wandb:
                        wandb.log({
                            "Global round": (epoch+1) + task * self.args["epochs"],
                            "Mean task accuracy": acc_at_round["top1"] 
                        })

                # for exemplar sampling
                prev_dataset = train_dataset
                prev_user_groups = user_groups

            # Freeze layers after learning the first task
            if task == 0 and not self.freeze_layers_flag and self.args["pretrained"] == "True":
                self.freeze_layers()
                self.freeze_layers_flag = True  # Ensure layers are frozen only once

            """
            Evaluation
            """
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
            print(predicts)
            print(targets)
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
                    outputs[i][0:task_id*self.each_task] = -float('inf')
                    outputs[i][(task_id+1)*self.each_task:] = -float('inf')
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
    
    def Image_transform(self, images, transform):
        tt = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((300, 300), interpolation=3),
                                transforms.RandomCrop((self.args["img_size"], self.args["img_size"]))])
        if not isinstance(images[0], np.ndarray):
            data = transform(pil_loader(images[0])).unsqueeze(0)
            data_notrsf = tt(pil_loader(images[0])).unsqueeze(0)
            for index in range(1, len(images)):
                data = torch.cat((data, transform(pil_loader(images[index])).unsqueeze(0)), dim=0)
                data_notrsf = torch.cat((data_notrsf, tt(pil_loader(images[index])).unsqueeze(0)), dim=0)
        else:
            data = transform(Image.fromarray(images[0])).unsqueeze(0)
            data_notrsf = tt(Image.fromarray(images[0])).unsqueeze(0)
            for index in range(1, len(images)):
                data = torch.cat((data, transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
                data_notrsf = torch.cat((data_notrsf, tt(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data, data_notrsf
    
    def compute_radius(self, model, dataset):
        num_img = 10
        class_means=[]
        radius = []
        task_size = self.args["increment"]

        with torch.no_grad():
            classes=list(range(task_size))
            for i in classes:
                idx = [j for j in range(len(dataset)) if dataset.labels[j] == i]
                if len(idx) > 0:
                    images = dataset.images[idx]
                    x, x_notrsf = self.Image_transform(images, self.transform) # check
                    x = x.to(self.device) 
                    model.eval()
                    for i in range(num_img):
                        j = 50 * i
                        imgs = x[j:j + 50]
                        feature = model.convnet(imgs)['features']
                        if i == 0:
                            features = feature
                        else:
                            features = torch.cat((features, feature), 0)
                        del feature
                        features = features.detach().cpu().numpy()
                        features = torch.from_numpy(features).to(self.device)
                        torch.cuda.empty_cache()

                    features = features.detach().cpu().numpy()
                    feature_dim = features.shape[1]
                    cov = np.cov(features.T)
                    radius.append(np.trace(cov) / feature_dim)
            self.radius = np.sqrt(np.mean(radius))
        print('radius_', self.radius)

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

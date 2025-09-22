import sys
sys.path.append('../')
from client.client_lander import Client_LANDER as Client
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as opAccuracyim
from progress.bar import Bar
import numpy as np
from utils.toolkit import count_parameters,weight_init
from utils.inc_net import IncrementalNet, Increment_ViT
from utils.data_manager import DataManager, partition_data, DatasetSplit, average_weights, setup_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import pickle
import math
from models.generator import NAYER,NLGenerator_IN,NLGenerator, get_norm_and_transform,UnlabeledImageDataset, transforms, DataIter
import os
import wandb 
import shutil

bn_mmt = 0.9
        
class Server_LANDER(object):
    def __init__(self, args):
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
        # create clients
        for idx in range(args["n_clients"]):
            self.clients.append(Client(args, idx))
        self.class_order = torch.tensor(self.data_manager.get_class_order())
        self.topk = 5
        self.each_task = args["increment"]
        self.save_dir = args["save_dir"] + "-" + str(self.seed)

        # label embedding 
        emb_file = 'DomainIL' if 'domainnet' in args['dataset'] else args['dataset']
        le_name = "./label_embedding/" + emb_file + "_le.pickle"
        with open(le_name, "rb") as label_file:
            label_emb = pickle.load(label_file)
            if args['dataset'] == 'imagenet-r' or 'domainnet' in args["dataset"]:
            #  Reorder label embeddings to match self.class_order
                label_emb_list = []
                for class_idx in self.class_order:
                    class_name = list(label_emb.keys())[class_idx]
                    label_emb_list.append(label_emb[class_name])
                self.label_emb = torch.tensor(label_emb_list, device=self.device, dtype=torch.float).detach().squeeze(1)
            else:
                label_emb = label_emb[self.class_order]
                self.label_emb = label_emb.to(self.device).float().detach()

            
        self.r = args['r']
        self.ltc = args['ltc']
        self.transform, self.normalizer = get_norm_and_transform(self.args, self.args["dataset"])
        self.save_dir = args["save_dir"]
        self.n_tasks = args["n_tasks"]
        self.nums = args["nums_syn"]

        self.test_dataset = []
        self.freeze_layers_flag = False 
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
        cnn_curve_gt_true = {"top1": [], "top5": []}
        cnn_curve_gt_false = {"top1": [], "top5": []}
        acc_t_t_gt_true, acc_t_t_gt_false = [], []
        acc_t_T_gt_true, acc_t_T_gt_false = [], []
        self.acc_matrix_gt_true = []
        self.acc_matrix_gt_false = []

        for task in range(self.data_manager.nb_tasks):
            self._cur_task += 1
            self._total_classes = self._known_classes + self.data_manager.get_task_size(self._cur_task)

            self.global_model.update_fc(self._cur_task, 1) # increment fc neurons  

            print("Learning on {}-{}".format(self._known_classes, self._total_classes))
            train_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train",mode="train")
            self.test_dataset.append(self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test", mode="test"))
            setup_seed(self.seed)


            test_dataset = self.data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
            self.test_loader_lander = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=1)

            if self._cur_task == 0 and (not os.path.exists(self.save_dir)):
                os.makedirs(self.save_dir)
            if self._cur_task != 0:
                self.syn_data_loader = self.get_syn_data_loader()
                
            # FL training
            self.best_model = None
            self.lowest_loss = np.inf
            user_groups,_ = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["n_clients"])

            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.global_model.parameters()), lr=self.args['lr'], 
                                        momentum=self.args['momentum'], weight_decay=self.args['weight_decay'])
            
            prog_bar = tqdm(range(self.args["epochs"]))
            for _, epoch in enumerate(prog_bar):
                local_weights = []
                loss_weight = []
                m = max(int(self.args["frac"] * self.args["n_clients"]), 1)
                selected_clients = np.random.choice(range(self.args["n_clients"]), m, replace=False)
                for idx in selected_clients:
                    print("Fine-tuning on client: ", idx)
                    local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                                                    batch_size=self.args["batch_size"], shuffle=True, num_workers=1)
                    self.clients[idx].trainloader = local_train_loader
                    # self.clients[idx].lr = scheduler.get_last_lr()[0]
                    self.clients[idx].label_emb = self.label_emb
                    self.clients[idx]._known_classes = self._known_classes
                    self.clients[idx]._total_classes = self._total_classes
                    self.clients[idx]._cur_task = self._cur_task
                    self.clients[idx].model = copy.deepcopy(self.global_model)
                    if self._cur_task == 0:
                        w, total_loss = self.clients[idx].local_updating()
                    else:
                        w, total_syn, total_local, total_loss = self.clients[idx].local_finetune(self._old_network, task, idx, self.syn_data_loader)
                        
                    local_weights.append(copy.deepcopy(w))
                    loss_weight.append(total_loss)
                    del local_train_loader, w
                    torch.cuda.empty_cache()
                
                # update global weights
                # scheduler.step()
                sum_loss = sum(loss_weight) 
                if sum_loss < self.lowest_loss:
                    self.lowest_loss = sum_loss
                    self.best_model = copy.deepcopy(self.global_model.state_dict())
                    
                global_weights = average_weights(local_weights)
                self.global_model.load_state_dict(global_weights)

                if (epoch+1) % 10 == 0:
                    acc_at_round, _ = self.eval_task()
                    if self.wandb:
                        wandb.log({
                            "Global round": (epoch+1) + task * self.args["epochs"],
                            "Mean task accuracy": acc_at_round["top1"] 
                        })
                    print(f'Accuracy: {acc_at_round["top1"] }')

            # Freeze layers after learning the first task
            if task == 0 and not self.freeze_layers_flag and self.args["pretrained"] == "True":
                self.freeze_layers()
                self.freeze_layers_flag = True  # Ensure layers are frozen only once
            
            if self._cur_task + 1 != self.n_tasks:
                self.data_generation()
                if self._cur_task >= 1:
                    self.remove_syn_imgs()
            

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

            self._known_classes = self._total_classes
            self._old_network = self.global_model.copy().freeze()
            
        acc_t_T_gt_true = cnn_accy_gt_true["grouped"]["record"]
        acc_t_T_gt_false = cnn_accy_gt_false["grouped"]["record"]


        # Compute forgetting
        print("Forgetting (gt=True):")
        self.compute_forgetting_from_matrix(self.acc_matrix_gt_true)
        print("Forgetting (gt=False):")
        self.compute_forgetting_from_matrix(self.acc_matrix_gt_false)

    
    def _compute_accuracy(self, model, test_loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.argmax(outputs, dim=1)
            correct += (predicts.cpu() == targets).sum()
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


    def get_syn_data_loader(self):
        if self.args["dataset"] == "cifar100":
            dataset_size = 50000
        elif self.args["dataset"] == "tiny_imagenet":
            dataset_size = 100000
        elif self.args["dataset"] == "imagenet100":
            dataset_size = 130000
        elif self.args["dataset"] == "imagenet":
            dataset_size = 1000000
        elif self.args["dataset"] == "cubs":
            dataset_size = 5994
        elif self.args["dataset"] == "imagenet-r":
            dataset_size = 24000
        elif 'domainnet' in self.args["dataset"]:
            dataset_size = 10000 * 6

        iters = math.ceil(dataset_size / (self.args["n_clients"] * self.args["n_tasks"] * self.args["batch_size"]))
        syn_bs = self.args["syn_bs"]*self.args["batch_size"]
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task - 1))
        print("iters{}, syn_bs:{}, data_dir: {}".format(iters, syn_bs, data_dir))
        
        
        syn_dataset = UnlabeledImageDataset(data_dir, transform=self.transform, nums=self.nums)
        syn_data_loader = torch.utils.data.DataLoader(syn_dataset, batch_size=syn_bs, shuffle=True, persistent_workers=True, num_workers=4)
        return syn_data_loader


    def data_generation(self):
        if self.args["dataset"] == "cifar100":
            img_size = 32
            img_shape = (3, img_size, img_size)
            if img_size == 32:
                generator = NLGenerator(ngf=64, img_size=img_size, nc=3, nl=10,
                                        label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                        sbz=self.args['synthesis_batch_size'])
            else:
                generator = NLGenerator_IN(ngf=64, img_size=img_size, nc=3, nl=10,
                            label_emb=self.label_emb, le_emb_size=self.args['nz'],
                            sbz=self.args['synthesis_batch_size']) 
        elif self.args["dataset"] == "tiny_imagenet":
            img_size = 64
            img_shape = (3, 64, 64)
            generator = NLGenerator(ngf=64, img_size=img_size, nc=3, nl=10,
                                    label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                    sbz=self.args['synthesis_batch_size'])
        elif self.args["dataset"] == "imagenet-r":
            img_size = 224
            img_shape = (3, 224, 224)
            generator = NLGenerator_IN(ngf=64, img_size=img_size, nc=3, nl=10,
                                      label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                      sbz=self.args['synthesis_batch_size'])
        elif self.args["dataset"] == "DomainNet" or "domainnet" in self.args["dataset"]:
            img_size = 224
            img_shape = (3, 224, 224)
            generator = NLGenerator_IN(ngf=64, img_size=img_size, nc=3, nl=10,
                                      label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                      sbz=self.args['synthesis_batch_size'])

        student = copy.deepcopy(self.global_model)
        student.apply(weight_init)
        tmp_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        synthesizer = NAYER(copy.deepcopy(self.global_model), student, generator, num_classes=self._total_classes,
                            img_size=img_shape, save_dir=tmp_dir, transform=self.transform, normalizer=self.normalizer,
                            synthesis_batch_size=self.args['synthesis_batch_size'], syn_sample_bs=self.args["syn_sample_bs"], 
                            iterations=self.args['g_steps'], warmup=self.args['warmup'], lr_g=self.args['lr_g'], adv=self.args['adv'], bn=self.args['bn'],
                            oh=self.args['oh'], ltc=self.ltc, r=self.r, device=self.device, bn_mmt=bn_mmt,
                            args=self.args, label_emb=self.label_emb)

        for it in range(self.args['syn_round'] + self.args['warmup']):
            synthesizer.synthesize(self._cur_task)  # generate synthetic data
            if it > self.args['warmup']:
                ms = synthesizer.get_student()
                test_accs = self._compute_accuracy(ms, self.test_loader_lander)
                mt = synthesizer.get_teacher()
                test_acct = self._compute_accuracy(mt, self.test_loader_lander)
                print("Student Test Acc: %s - %s" % (test_accs, test_acct))

        print("For task {}, data generation completed! ".format(self._cur_task))

    def remove_syn_imgs(self):
        folder_path = os.path.join(self.save_dir, "task_{}".format(self._cur_task-1))
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (folder_path, e))


    def compute_forgetting_from_matrix(self, acc_matrix):
        """
        acc_matrix: list of lists, acc_matrix[t][k] = accuracy on task k after training task t.
        (0-indexed tasks: t = 0..T-1, k = 0..t)
        """

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

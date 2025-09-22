import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)



def accuracy(y_pred, y_true, nb_old, increment=10, domain=False):
    assert len(y_pred) == len(y_true), "Data length error."
    
    if domain == True: 
        domain_id = (y_true // increment).astype(np.int64)
        class_id = (y_true % increment).astype(np.int64)
        all_acc = {}
        all_acc["total"] = np.around(
            (y_pred == class_id).sum() * 100 / len(class_id), decimals=2
        )

        # Grouped accuracy
        acc_save_list = []
        max_task = int(domain_id.max()) if len(domain_id) else -1
        for task_id in range(max_task + 1):
            idxes = np.where(
                np.logical_and(y_true >= task_id * increment, y_true < (task_id +1) * increment)
            )[0]
            label = "{}-{}".format(
                str(task_id * increment).rjust(2, "0"), str((task_id +1) * increment - 1).rjust(2, "0")
            )
            acc_id =  np.around(
                (y_pred[idxes] == class_id[idxes]).sum() * 100 / len(idxes), decimals=2
            )
            all_acc[label] = acc_id
            acc_save_list.append(acc_id)
        last_label = label 
        all_acc["record"] = acc_save_list
        
        # Old accuracy
        idxes = np.where(y_true < nb_old)[0]
        all_acc["old"] = (
            0
            if len(idxes) == 0
            else np.around(
                (y_pred[idxes] == class_id[idxes]).sum() * 100 / len(idxes), decimals=2
            )
        )

        # New accuracy
        idxes = np.where(y_true >= nb_old)[0]
        all_acc["new"] = np.around(
            (y_pred[idxes] == class_id[idxes]).sum() * 100 / len(idxes), decimals=2
        )
        if np.isnan(all_acc["new"]):
            all_acc["new"] = all_acc[last_label]

    else:
        all_acc = {}
        all_acc["total"] = np.around(
            (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
        )

        # Grouped accuracy
        acc_save_list = []
        for class_id in range(0, np.max(y_true), increment):
            idxes = np.where(
                np.logical_and(y_true >= class_id, y_true < class_id + increment)
            )[0]
            label = "{}-{}".format(
                str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
            )
            acc_id =  np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )
            all_acc[label] = acc_id
            acc_save_list.append(acc_id)
        last_label = label 
        all_acc["record"] = acc_save_list
        
        # Old accuracy
        idxes = np.where(y_true < nb_old)[0]
        all_acc["old"] = (
            0
            if len(idxes) == 0
            else np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )
        )

        # New accuracy
        idxes = np.where(y_true >= nb_old)[0]
        all_acc["new"] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
        if np.isnan(all_acc["new"]):
            all_acc["new"] = all_acc[last_label]

    return all_acc



def split_images_labels(imgs): # all img names
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
                
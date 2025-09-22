import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy, wandb
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from kornia import augmentation
import time, os, math
import torch.nn.init as ini
from PIL import Image
import pickle
import shutil

T = 20.0
student_train_step = 50

class NLGenerator(nn.Module):
    def __init__(self, ngf=64, img_size=32, nc=3, nl=100, label_emb=None, le_emb_size=256, le_size=512, sbz=200):
        super(NLGenerator, self).__init__()
        self.params = (ngf, img_size, nc, nl, label_emb, le_emb_size, le_size, sbz)
        self.le_emb_size = le_emb_size
        self.label_emb = torch.nn.Parameter(label_emb, requires_grad=False)
        self.init_size = img_size // 4
        self.le_size = le_size
        self.nl = nl
        self.sbz = sbz
        self.nle = int(np.ceil(self.sbz / self.nl))
        self.n1 = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, targets=None):
        le = self.label_emb[targets]
        le = self.n1(le)
        v = None
        for i in range(self.nle):
            if (i + 1) * self.nl > le.shape[0]:
                sle = le[i * self.nl:]
            else:
                sle = le[i * self.nl:(i + 1) * self.nl]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))

        out = self.l1(v)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def reinit(self):
        return NLGenerator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
                           self.params[5], self.params[6], self.params[7]).cuda()


class NLGenerator_IN(nn.Module):
    def __init__(self, ngf=64, img_size=224, nc=3, nl=100, label_emb=None, le_emb_size=256, le_size=512, sbz=200):
        super(NLGenerator_IN, self).__init__()
        self.params = (ngf, img_size, nc, nl, label_emb, le_emb_size, le_size, sbz)
        self.le_emb_size = le_emb_size
        self.label_emb = torch.nn.Parameter(label_emb, requires_grad=False)
        self.init_size = img_size // 16
        self.le_size = le_size
        self.nl = nl
        self.nle = int(np.ceil(sbz/nl))
        self.sbz = sbz

        self.n1 = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            #nn.Conv2d(nz, ngf, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.LeakyReLU(0.2, inplace=True),
            # 7x7

            #nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 28x28

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 56x56

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 112 x 112

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 224 x 224

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def forward(self, targets=None):
        le = self.label_emb[targets]
        # le = self.sig1(le)
        le = self.n1(le)
        v = None
        for i in range(self.nle):
            if (i+1)*self.nl > le.shape[0]:
                sle = le[i*self.nl:]
            else:
                sle = le[i*self.nl:(i+1)*self.nl]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))

        out = self.l1(v)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def reinit(self):
        return NLGenerator_IN(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
                               self.params[5], self.params[6], self.params[7]).cuda()




def get_norm_and_transform(args, dataset):

    if dataset == "cifar100":
        data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
        if 'vit' in args['net']:
            train_transform = transforms.Compose([
                    transforms.Resize(224), 
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**dict(data_normalize)),
                ]) 
        
    elif dataset == "tiny_imagenet":
        data_normalize = dict(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
    elif dataset == "imagenet":
        data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(**dict(data_normalize)),
        ])
    elif dataset == "imagenet-r":
        data_normalize = dict(mean=[0, 0, 0], std=[1, 1, 1])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**dict(data_normalize)),
        ])
    elif dataset == "domainnet":
        data_normalize = dict(mean=[0, 0, 0], std=[1, 1, 1])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(**dict(data_normalize)),
         ]) 

    normalizer = Normalizer(**dict(data_normalize))
    return train_transform, normalizer


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


# normalizer = Normalizer(**dict(data_normalize))


def _collect_all_images(nums, root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            if nums != None:
                files.sort()
                files = files[:nums]
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, self.root)  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        # print("1: %s" % str(np.array(img).shape))
        if self.transform:
            img = self.transform(img)
        # print("2: %s" % str(img.shape))
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
        self.root, len(self), self.transform)


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67)  # , alpha=40


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)  # , alpha=0.67


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '-%d.png' % (idx))


class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        if module.track_running_stats == True:
            self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                        self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data)

    def remove(self):
        self.hook.remove()


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False)
        self._idx += 1

    def get_dataset(self, nums=None, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform, nums=nums)

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)


def custom_cross_entropy(preds, target):
    return torch.mean(torch.sum(-target * preds.log_softmax(dim=-1), dim=-1))


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


class NAYER():
    def __init__(self, teacher, student, generator, num_classes, img_size, iterations=100, lr_g=0.1, label_emb=None,
                 synthesis_batch_size=128, syn_sample_bs=128, adv=0.0, bn=1, oh=1, r=1e-1, ltc=0.2, save_dir='run/fast', transform=None,
                 normalizer=None, device="gpu:0", warmup=10, bn_mmt=0, args=None):
        super(NAYER, self).__init__()
        self.teacher = teacher
        self.student = student
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.args = args
        self.label_emb = label_emb
        self.ltc = ltc
        self.r = r
        self.device = device

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.syn_sample_bs = syn_sample_bs
        self.normalizer = normalizer

        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.generator = generator.to(device).train()
        self.ep = 0
        self.ep_start = self.args['warmup'] + 1 # need more time to the student train well.
        self.prev_z = None
        
        if "imagenet" in self.args["dataset"]:
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.args["img_size"], self.args["img_size"]], padding=4),
                augmentation.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
            ])
        else:
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.args["img_size"], self.args["img_size"]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ])

        if self.args["dataset"] == "cifar100" and img_size == 32 and 'vit' in self.args['net']:  
            self.aug = transforms.Compose([ 
                    augmentation.Resize(224), 
                    augmentation.RandomResizedCrop(size=[224, 224]),    
                    augmentation.RandomHorizontalFlip(),
                    normalizer,
                ])
            
        self.mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=True, device=self.device)
        self.std = torch.tensor([0.2, 0.2, 0.2], requires_grad=True, device=self.device)

        self.bn_mmt = bn_mmt
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d) and m.track_running_stats != False:
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

    def synthesize(self, _cur_task=0):
        self._cur_task = _cur_task
        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        best_oh = 1e6

        # if self.ep % 200 == 0:
        #     self.generator = self.generator.reinit()

        best_inputs = None
        self.generator.re_init_le()

        targets, ys = self.generate_ys(cr=0.0)
        ys = ys.to(self.device)
        targets = targets.to(self.device)

        optimizer = torch.optim.Adam([
            {'params': self.generator.parameters()},
            {'params': [self.mean], 'lr': 0.01},
            {'params': [self.std], 'lr': 0.01}
        ], lr=self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations):
            inputs = self.generator(targets=targets)
            inputs_aug = self.aug(inputs)
            inputs_aug = (inputs_aug - self.mean[None, :, None, None]) / (self.std[None, :, None, None])
            output_list = self.teacher(inputs_aug)
            t_out = output_list["logits"]
            feature = output_list["att"]

            loss_bn = sum([h.r_feature for h in self.hooks]) 
            loss_oh = custom_cross_entropy(t_out, ys.detach())

            if self.adv > 0 and (self.ep - 1 > self.ep_start):
                s_out = self.student(inputs_aug)["logits"]
                mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(
                    1) * mask).mean()  # decision adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            target_f = self.label_emb[targets]
            loss_f = torch.relu(torch.nn.functional.mse_loss(feature, target_f.detach()) - self.r)

            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.ltc * loss_f

            if loss_oh.item() < best_oh:
                best_oh = loss_oh
            # print("%s - bn %s - bn %s - oh %s - adv %s -fr %s - %s - %s" % (
                # it,
                # float((loss_bn * self.bn).data.cpu().detach().numpy()),
                # float(loss_bn.data.cpu().detach().numpy()),
                # float((loss_oh).data.cpu().detach().numpy()),
                # float((loss_adv).data.cpu().detach().numpy()),
                # float(loss_f.data.cpu().detach().numpy()),
                # str(self.mean.detach().cpu()[0]),
                # str(self.std.detach().cpu()[0])))

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        if self.args['warmup'] <= self.ep:
            self.data_pool.add(best_inputs)
            self.student_train(self.student, self.teacher)

    def student_train(self, student, teacher):
        # lr = 0.1 (original code)
        if 'domainnet' in self.args["dataset"]:
            std_lr = 1e-3
        else:
            std_lr = 5e-3

        optimizer = torch.optim.SGD(student.parameters(), lr=std_lr, momentum=0.9, weight_decay=0.0002)
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data()
        data_iter = DataIter(loader)
        criterion = KLDiv(T=T)

        prog_bar = tqdm(range(student_train_step))
        for _, com in enumerate(prog_bar):
            images = data_iter.next()
            images = images.to(self.device)
            with torch.no_grad():
                t_out_list = teacher(images)
                t_out = t_out_list["logits"]
                t_f = t_out_list["att"]
            s_out_list = student(images.detach())
            s_out = s_out_list["logits"]
            s_f = s_out_list["att"]
            loss_s = criterion(s_out, t_out.detach())
            s_loss_f = torch.nn.functional.mse_loss(s_f, t_f.detach())
            loss = loss_s + self.ltc * s_loss_f

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def get_all_syn_data(self):
        syn_dataset = UnlabeledImageDataset(self.save_dir, transform=self.transform, nums=self.args['nums_syn'])
        loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=self.syn_sample_bs, shuffle=True, persistent_workers=True,
            num_workers=4)
        return loader

    def generate_ys(self, cr=0.0):

        if self.args["classIL"] != "True":
            num_classes = self.args["increment"]
        else: 
            num_classes = self.num_classes

        s = self.synthesis_batch_size // num_classes
        v = self.synthesis_batch_size % num_classes
        target = torch.randint(num_classes, (v,))

        for i in range(s):
            tmp_label = torch.tensor(range(0, num_classes))
            target = torch.cat((tmp_label, target))

        ys = torch.zeros(self.synthesis_batch_size, num_classes)
        ys.fill_(cr / (num_classes - 1))
        ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))

        return target, ys
    
    def get_student(self):
        return self.student

    def get_teacher(self):
        return self.teacher
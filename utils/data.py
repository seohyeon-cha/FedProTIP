import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.datautils.class_names import domainnet_classnames
from utils.tiny_imagenet import TinyImageNet
from utils.dataset_utils import *
import os
import os.path
import requests
from tqdm import tqdm
from zipfile import ZipFile
from shutil import move, rmtree
import csv


data_dir = "../data"

class iData(object):
    def __init__(self, args):
        self.train_trsf = []
        self.test_trsf = []
        self.common_trsf = []
        self.class_order = None

    def get_train_trsf(self):
        return self.train_trsf

    def get_test_trsf(self):
        return self.test_trsf

    def get_common_trsf(self):
        return self.common_trsf

    def get_class_order(self):
        return self.class_order


class iCIFAR10(iData):
    def __init__(self, args):
        super(iCIFAR10, self).__init__(args)
        self.use_path = False
        self.train_trsf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
        ]
        self.test_trsf = []
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            )
        ]

        self.class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):

    def __init__(self, args):
        super(iCIFAR100, self).__init__(args)
        self.use_path = False
        self.train_trsf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor()
        ]
        self.test_trsf = [transforms.ToTensor()]
        self.common_trsf = [
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
        if "vit" in args["net"] and args["img_size"] == 224:
            self.train_trsf = [
                transforms.Resize(224), 
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor()
            ]
            self.test_trsf = [transforms.Resize(224), transforms.ToTensor()]
            self.common_trsf = [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
     
        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    def __init__(self, args):
        super(iImageNet1000, self).__init__(args)
        self.use_path = True
        self.train_trsf = [
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(1000).tolist()

    def download_data(self):
        data_root = os.path.join(data_dir, 'imagenet')
        train_root = os.path.join(data_root, 'train')
        val_root = os.path.join(data_root, 'val')

        train_dset = datasets.ImageFolder(train_root)
        test_dset = datasets.ImageFolder(val_root)

        # train_dset = datasets.ImageNet(data_root, split='train', transform=self.train_trsf)
        # test_dset = datasets.ImageNet(data_root, split='val', transform=self.test_trsf)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):  # 1300*100 = 13w, 5 tasks, each task 20*1300=2.6w
    def __init__(self, args):
        super(iImageNet100, self).__init__(args)
        self.use_path = True
        self.train_trsf = [
            # transforms.RandomResizedCrop(224),
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.CenterCrop(128),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dir = "{}/imagenet100/train/".format(data_dir)
        test_dir = "{}/imagenet100/val/".format(data_dir)

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


# Samimport os
import logging
import numpy as np
from PIL import Image
from torchvision import transforms

# https://github.com/aimagelab/mammoth/tree/master/datasets
class TinyImageNet200(iData):
    def __init__(self, args):
        super(TinyImageNet200, self).__init__(args)
        self.use_path = True  # IMPORTANT: your DataManager expects file paths
        self.train_trsf = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.CenterCrop(64),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975],
                                 [0.2302, 0.2265, 0.2262]),
        ]
        self.class_order = np.arange(200).tolist()

        # same OneDrive link Mammoth uses
        self._mammoth_onedrive_url = (
            "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/"
            "EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD"
        )
        self._zip_name = "tiny-imagenet-processed.zip"
        self._processed_dirname = "processed"

    def download_data(self):
        root = data_dir  # keep using your global/data_dir
        proc_dir = os.path.join(root, self._processed_dirname)
        zip_path = os.path.join(root, self._zip_name)

        os.makedirs(root, exist_ok=True)

        # 1) ensure processed files exist (download + unzip if needed)
        if not os.path.isdir(proc_dir) or len(os.listdir(proc_dir)) == 0:
            logging.info("Downloading Tiny-ImageNet (Mammoth preprocessed)…")
            self._download_processed_zip(self._mammoth_onedrive_url, zip_path)
            logging.info("Extracting…")
            self._extract_zip(zip_path, root)

        # 2) load shards (float32 in [0,1] likely)
        logging.info("Loading shards…")
        x_train, y_train = self._load_split(proc_dir, "train")
        x_val,   y_val   = self._load_split(proc_dir, "val")

        # 3) write images to a deterministic cache as PNGs and return path lists
        train_cache_dir = os.path.join(root, "cache_tinyimg", "train")
        val_cache_dir   = os.path.join(root, "cache_tinyimg", "val")
        os.makedirs(train_cache_dir, exist_ok=True)
        os.makedirs(val_cache_dir, exist_ok=True)

        self.train_data = np.array(self._ensure_image_cache(x_train, train_cache_dir, prefix="tr"))
        self.test_data  = np.array(self._ensure_image_cache(x_val,   val_cache_dir,   prefix="va"))

        # keep targets as int64 numpy arrays
        self.train_targets = y_train.astype(np.int64, copy=False)
        self.test_targets  = y_val.astype(np.int64, copy=False)

        logging.info(f"Train paths: {len(self.train_data)} | Test paths: {len(self.test_data)}")
        logging.info("TinyImageNet200 ready with file-path returns (use_path=True).")

    # ----------------- helpers -----------------
    def _download_processed_zip(self, url: str, zip_path: str):
        try:
            from onedrivedownloader import download
            download(url, filename=zip_path, unzip=False, unzip_path=None, clean=False)
        except Exception as e:
            logging.warning(f"onedrivedownloader failed ({e}). Falling back to requests.")
            import requests
            r = requests.get(url, allow_redirects=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(r.content)

    @staticmethod
    def _extract_zip(zip_path: str, root: str):
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

    @staticmethod
    def _load_split(proc_dir: str, split: str):
        xs, ys = [], []
        for num in range(1, 21):
            x_path = os.path.join(proc_dir, f"x_{split}_{num:02d}.npy")
            y_path = os.path.join(proc_dir, f"y_{split}_{num:02d}.npy")
            if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
                raise FileNotFoundError(f"Missing shard {num:02d} for split='{split}' in {proc_dir}")
            xs.append(np.load(x_path, mmap_mode="r"))
            ys.append(np.load(y_path, mmap_mode="r"))
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        return X, Y

    @staticmethod
    def _ensure_image_cache(X: np.ndarray, out_dir: str, prefix: str):
        """
        Ensure each image in X is saved as a PNG and return a list of absolute file paths.
        Assumes X is either float in [0,1] or uint8 in [0,255]. Shape: [N, 64, 64, 3].
        """
        N = X.shape[0]
        paths = [os.path.join(out_dir, f"{prefix}_{i:06d}.png") for i in range(N)]

        # Fast path: if files already exist (same count), assume cached
        need_write = False
        for p in paths:
            if not os.path.exists(p):
                need_write = True
                break

        if not need_write:
            return paths

        # Write (idempotent; overwrites only missing files)
        for i, p in enumerate(paths):
            if os.path.exists(p):
                continue
            img = X[i]
            # normalize dtype
            if img.dtype != np.uint8:
                img = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
            # ensure shape correct
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Unexpected image shape at {i}: {img.shape} (expected [64,64,3])")
            Image.fromarray(img, mode="RGB").save(p, format="PNG", compress_level=1)

        return paths

# CUBS dataset
# original code
# https://github.com/CVIR/ConvPrompt/blob/main/continual_datasets/continual_datasets.py#L420

class CUB200(iData):
    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):    
        super(CUB200, self).__init__(args)    
        self.use_path = True
        self.train_trsf = [
                    transforms.Resize((300, 300), interpolation=3),
                    # transforms.RandomCrop((224, 224)),
                    transforms.RandomCrop((args["img_size"], args["img_size"])), 
                    transforms.RandomHorizontalFlip(),
                ]
        self.test_trsf = [transforms.Resize(256, interpolation=3),
                # transforms.CenterCrop(224),
                transforms.CenterCrop(args["img_size"])
                ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.class_order = np.arange(200).tolist()
        self.root = root
        self.download = download

    def download_data(self):
        self.root = os.path.expanduser(self.root)
        self.url = 'https://data.deepai.org/CUB200(2011).zip'
        self.filename = 'CUB200(2011).zip'

        fpath = os.path.join(self.root, self.filename)
        if not os.path.isfile(fpath):
            if not self.download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, self.root, filename=self.filename)

        if not os.path.exists(os.path.join(self.root, 'CUB_200_2011')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(self.root)
            zip_ref.close()

            import tarfile
            tar_ref = tarfile.open(os.path.join(self.root, 'CUB_200_2011.tgz'), 'r')
            tar_ref.extractall(self.root)
            tar_ref.close()

            self.split()
        
        fpath_train = os.path.join(self.root, 'CUB_200_2011', 'train')
        fpath_test = os.path.join(self.root, 'CUB_200_2011', 'test')

        train_dst = datasets.ImageFolder(fpath_train, transform=self.train_trsf)
        val_dst = datasets.ImageFolder(fpath_test, transform=self.test_trsf)
        self.train_data, self.train_targets = split_images_labels(train_dst.imgs)
        self.test_data, self.test_targets = split_images_labels(val_dst.imgs)

                      
    def split(self):
        train_folder = self.root + 'CUB_200_2011/train'
        test_folder = self.root + 'CUB_200_2011/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        images = self.root + 'CUB_200_2011/images.txt'
        train_test_split = self.root + 'CUB_200_2011/train_test_split.txt'

        with open(images, 'r') as image:
            with open(train_test_split, 'r') as f:
                for line in f:
                    image_path = image.readline().split(' ')[-1]
                    image_path = image_path.replace('\n', '')
                    class_name = image_path.split('/')[0].split(' ')[-1]
                    src = self.root + 'CUB_200_2011/images/' + image_path

                    if line.split(' ')[-1].replace('\n', '') == '1':
                        if not os.path.exists(train_folder + '/' + class_name):
                            os.mkdir(train_folder + '/' + class_name)
                        dst = train_folder + '/' + image_path
                    else:
                        if not os.path.exists(test_folder + '/' + class_name):
                            os.mkdir(test_folder + '/' + class_name)
                        dst = test_folder + '/' + image_path
                    
                    move(src, dst)

# Split ImageNet-R 
# https://github.com/CVIR/ConvPrompt/blob/main/continual_datasets/continual_datasets.py#L420
# Followed train/val split of L2P https://github.com/google-research/l2p

class Imagenet_R(iData):
    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=True):  
        super(Imagenet_R, self).__init__(args)          
        self.root = os.path.expanduser(root)
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]

        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ]
        self.common_trsf = [
            transforms.Normalize(
                mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
            ),
        ]
        self.class_order = np.arange(200).tolist()
        self.root = root
        self.download = download

    def download_data(self):
        self.url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
        self.filename = 'imagenet-r.tar'
        self.train_subset_url = "https://gist.githubusercontent.com/gqk/e127fe18bf179bdcbdf5e29a8c1ae523/raw/train_list.txt"
        self.train_filename = 'train_list.txt'
        self.test_subset_url = "https://gist.githubusercontent.com/gqk/e127fe18bf179bdcbdf5e29a8c1ae523/raw/val_list.txt"
        self.test_filename = 'val_list.txt'

        self.fpath = os.path.join(self.root, 'imagenet-r')
        if not os.path.isfile(self.fpath):
            if not self.download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, self.root, filename=self.filename)

        if not os.path.exists(os.path.join(self.root, 'imagenet-r')):
            import tarfile
            tar_ref = tarfile.open(os.path.join(self.root, self.filename), 'r')
            tar_ref.extractall(self.root)
            tar_ref.close()
        
        self.train_txt = os.path.join(self.fpath, self.train_filename)
        self.test_txt = os.path.join(self.fpath, self.test_filename)
        if not os.path.exists(self.train_txt):
            print("Downloading trainset indexes...", end=" ")
            download_url(self.train_subset_url, self.fpath, filename=self.train_filename)
            print("Done!")
        if not os.path.exists(self.test_txt):
            print("Downloading testset indexes...", end=" ")
            download_url(self.test_subset_url, self.fpath, filename=self.test_filename)
            print("Done!")

        # Read train/test file paths from the downloaded text files
        with open(self.train_txt, 'r') as f:
            self.train_file_list = [line.strip() for line in f.readlines()]
        with open(self.test_txt, 'r') as f:
            self.test_file_list = [line.strip() for line in f.readlines()]

        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.dataset = datasets.ImageFolder(self.fpath, transform=self.train_trsf)
            
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            print(f"Trainset size: {train_size} | Testset size: {val_size}")
            train, val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices
    
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]
            self.split()
        
        fpath_train = self.fpath + '/train'
        fpath_test = self.fpath + '/test'

        train_dst = datasets.ImageFolder(fpath_train, transform=self.train_trsf)
        val_dst = datasets.ImageFolder(fpath_test, transform=self.test_trsf)
        self.train_data, self.train_targets = split_images_labels(train_dst.imgs)
        self.test_data, self.test_targets = split_images_labels(val_dst.imgs)
        print(f"Train set size: {len(self.train_data)} | Test set size: {len(self.test_data)}")
    
    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))
        
        for path in self.train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)
        
        for c in self.dataset.classes:
            path = os.path.join(self.fpath, c)
            rmtree(path)



class iDomainNet(iData):

    MANY_SHOT_THRES = 100
    FEW_SHOT_THRES = 20

    def __init__(self, args, base_dir="../data"):
        super(iDomainNet, self).__init__(args)
        self.args = args
        self.use_path = True
        self.base_dir = base_dir
        self.domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        self.domain_links = {
            "clipart": "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
            "infograph": "https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
            "painting": "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
            "quickdraw": "https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
            "real": "https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
            "sketch": "https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
        }
        self.txt_files = [
            "clipart_train.txt", "clipart_test.txt",
            "infograph_train.txt", "infograph_test.txt",
            "painting_train.txt", "painting_test.txt",
            "quickdraw_train.txt", "quickdraw_test.txt",
            "real_train.txt", "real_test.txt",
            "sketch_train.txt", "sketch_test.txt",
        ]
        self.base_dir = os.path.join(self.base_dir, "DomainNet")
        
        class_order = np.arange(6 * 345).tolist()
        self.class_order = class_order
        self.cls_num = 345
        self.domain_order = self.args["domain_order"]
        self.domain_names = self.get_domain_names()
        print(self.domain_names)

        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ]

    def get_domain_names(self):
        if self.domain_order == 1:
            return ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        elif self.domain_order == 2:
            return ["infograph", "painting", "sketch", "clipart", "quickdraw", "real"]
        elif self.domain_order == 3:
            return ["painting", "quickdraw", "real", "sketch", "clipart", "infograph"]
        elif self.domain_order == 4:
            return ["real", "sketch", "painting", "infograph", "quickdraw", "clipart"]
        elif self.domain_order == 5:
            return ["sketch", "clipart", "quickdraw", "real", "infograph", "painting"]
        elif self.domain_order == 6:
            return ["clipart", "real", "painting", "sketch", "infograph", "quickdraw"]

    def download_file(self, url, destination):
        """
        Download a file from a URL to a local destination.
        """
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(destination, "wb") as file, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit="B",
            unit_scale=True,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                pbar.update(len(data))

    def download_data(self):
        os.makedirs(self.base_dir, exist_ok=True)
        for domain, url in self.domain_links.items():
            zip_path = os.path.join(self.base_dir, f"{domain}.zip")
            extract_path = os.path.join(self.base_dir, domain)
            if not os.path.exists(extract_path): 
                print(f"Downloading {domain} data...")
                self.download_file(url, zip_path)

                print(f"Extracting {domain} data...")
                with ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(self.base_dir)
                os.remove(zip_path)

        print("All DomainNet data and split files downloaded and extracted.")

        reversed_data = {name: index for index, name in domainnet_classnames.items()}
        train_x, train_y = [], []
        test_x, test_y = [], []
        with open(os.path.join("utils/datautils", "domainnet.csv"), "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            print(header)
            for row in csv_reader:
                domain_type = row[0]
                data_path = row[2]
                cls_name = os.path.basename(os.path.dirname(data_path))
                data_type = row[3]
                cls_id = reversed_data[cls_name]
                domain_id = self.domain_names.index(domain_type)
                absulute_path = os.path.join(self.base_dir, data_path)
                if data_type == "train":
                    train_x.append(absulute_path)
                    train_y.append(cls_id + domain_id * 345)
                else:
                    test_x.append(absulute_path)
                    test_y.append(cls_id + domain_id * 345)
        
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)
        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)
        print(f"Before sampling: Train data {len(self.train_data)} | Test data {len(self.test_data)}")

        # --------- Fixed-size per-domain sampling ---------
        samples_per_domain = 10000  # change as needed
        rng = np.random.default_rng(self.args["seed"])  # reproducible

        # Train set sampling
        train_data_list, train_target_list = [], []
        for domain_id in range(len(self.domain_names)):
            mask = (self.train_targets // 345) == domain_id
            domain_data = self.train_data[mask]
            domain_targets = self.train_targets[mask]
            if len(domain_data) < samples_per_domain:
                raise ValueError(f"Domain {domain_id} has only {len(domain_data)} train samples.")
            idx = rng.choice(len(domain_data), size=samples_per_domain, replace=False)
            train_data_list.append(domain_data[idx])
            train_target_list.append(domain_targets[idx])
        self.train_data = np.concatenate(train_data_list)
        self.train_targets = np.concatenate(train_target_list)

        print(f"After sampling: Train data {len(self.train_data)} | Test data {len(self.test_data)}")

class DomainNet(iData):
    def __init__(self, args, base_dir="../data", supercategory_file="supercategories.txt"):
        super(DomainNet, self).__init__(args)
        self.use_path = True
        self.domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        self.domain_links = {
            "clipart": "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
            "infograph": "https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
            "painting": "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
            "quickdraw": "https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
            "real": "https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
            "sketch": "https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
        }
        self.txt_files = [
            "clipart_train.txt", "clipart_test.txt",
            "infograph_train.txt", "infograph_test.txt",
            "painting_train.txt", "painting_test.txt",
            "quickdraw_train.txt", "quickdraw_test.txt",
            "real_train.txt", "real_test.txt",
            "sketch_train.txt", "sketch_test.txt",
        ]
        self.txt_base_url = "https://raw.githubusercontent.com/NeurAI-Lab/DN4IL-dataset/main/"
        self.base_dir = os.path.join(base_dir, "dn4il")

        self.supercategory_file = os.path.join(base_dir, supercategory_file)
        self.supercategories, self.class_to_index = self._load_supercategories()
        self.train_trsf = [
            transforms.RandomResizedCrop(224, scale=(0.75, 1)),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize((224, 224)),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.class_order = np.arange(100).tolist()  # Define class order as 0 to 99

    def _load_supercategories(self):
        """
        Load supercategories and classes from the specified text file.
        Map classes to indices 0-99.
        """
        supercategories = {}
        class_to_index = {}
        current_index = 0

        if not os.path.exists(self.supercategory_file):
            raise FileNotFoundError(f"Supercategory file {self.supercategory_file} not found.")

        with open(self.supercategory_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    supercategory, classes = line.split(":")
                    supercategory = supercategory.strip()
                    class_list = [cls.strip() for cls in classes.split(",")]
                    supercategories[supercategory] = class_list
                    for cls in class_list:
                        class_to_index[cls] = current_index
                        current_index += 1

        return supercategories, class_to_index

    def _filter_samples_by_supercategory(self, samples):
        """
        Filter samples to include only those in the supercategories and map their labels to 0-99.
        """
        filtered_samples = []
        for path, label in samples:
            class_name = path.split('/')[-2]  # Extract the class name from the path
            if class_name in self.class_to_index:
                filtered_samples.append((path, self.class_to_index[class_name]))
        return filtered_samples

    def download_file(self, url, destination):
        """
        Download a file from a URL to a local destination.
        """
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(destination, "wb") as file, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit="B",
            unit_scale=True,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                pbar.update(len(data))

    def download_txt_files(self):
        """
        Download the required .txt files for DomainNet if they are not already present.
        """
        os.makedirs(self.base_dir, exist_ok=True)
        for txt_file in self.txt_files:
            file_path = os.path.join(self.base_dir, txt_file)
            if not os.path.exists(file_path):
                url = self.txt_base_url + txt_file
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded {txt_file}")
                else:
                    raise Exception(f"Failed to download {txt_file} from {url}")

    def download_data(self):
        """
        Prepare train and test datasets based on supercategories.
        """
        # Download domain-specific zip files
        os.makedirs(self.base_dir, exist_ok=True)
        for domain, url in self.domain_links.items():
            zip_path = os.path.join(self.base_dir, f"{domain}.zip")
            extract_path = os.path.join(self.base_dir, domain)

            if not os.path.exists(extract_path):
                print(f"Downloading {domain} data...")
                self.download_file(url, zip_path)

                print(f"Extracting {domain} data...")
                with ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(self.base_dir)
                os.remove(zip_path)

        print("All DomainNet data and split files downloaded and extracted.")

        # Ensure .txt files are downloaded
        self.download_txt_files()

        train_samples, test_samples = [], []

        for domain in self.domains:
            train_txt = os.path.join(self.base_dir, f"{domain}_train.txt")
            test_txt = os.path.join(self.base_dir, f"{domain}_test.txt")

            if not os.path.exists(train_txt) or not os.path.exists(test_txt):
                raise FileNotFoundError(f"Split files not found for {domain}")

            # Read train and test splits
            with open(train_txt, "r") as f:
                train_samples.extend([line.strip().split() for line in f.readlines()])
            with open(test_txt, "r") as f:
                test_samples.extend([line.strip().split() for line in f.readlines()])

        # Filter samples by supercategory and map class labels to 0-99
        train_samples = self._filter_samples_by_supercategory(train_samples)
        test_samples = self._filter_samples_by_supercategory(test_samples)

        # Split into images and labels
        train_data, train_targets = zip(*[
            (os.path.join(self.base_dir, path), label)
            for path, label in train_samples if os.path.exists(os.path.join(self.base_dir, path))
        ])
        test_data, test_targets = zip(*[
            (os.path.join(self.base_dir, path), label)
            for path, label in test_samples if os.path.exists(os.path.join(self.base_dir, path))
        ])

        self.train_data, self.train_targets = np.array(train_data), np.array(train_targets)
        self.test_data, self.test_targets = np.array(test_data), np.array(test_targets)
        print(f"Train data: {len(self.train_data)} | Test data: {len(self.test_data)}")

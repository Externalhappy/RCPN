from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
import json

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class FileListDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
    def __getitem__(self, index):
        image = pil_loader(self.images[index])
        target = self.labels[index]
        if self.transform is not None:
            sample = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index
    def __len__(self):
        return len(self.images)

class ImageFolderWithIndex(datasets.ImageFolder):
    """Custom dataset that includes image file index. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithIndex, self).__getitem__(index)
        # make a new tuple that includes original and the path
        tuple_with_index = (original_tuple + (index,))
        return tuple_with_index

class Flowers(Dataset):
    def __init__(self, root, transform=None, istrain=True):
        image_dir = os.path.join(root, "jpg")
        split_path = os.path.join(root, "split_zhou_OxfordFlowers.json")
        self.data = self.read_split(split_path, image_dir, istrain)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, idx
    def read_split(self, filepath, path_prefix, istrain):
        def _convert(items):
            out = []
            for impath, label, _ in items:
                impath = os.path.join(path_prefix, impath)
                item = (impath, int(label))
                out.append(item)
            return out
        def read_json(fpath):
            """Read json file from a path."""
            with open(fpath, "r") as f:
                obj = json.load(f)
            return obj
        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        if istrain:
            return _convert(split["train"])
        return _convert(split["test"])

class CUBDataset(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):

        img_root = os.path.join(root, 'images')
        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))
        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = list()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if fn not in filenames_to_use and int(idx) in indices_to_use:
                    filenames_to_use.append(fn)

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]
        _, targets_to_use = list(zip(*imgs_to_use))
        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use
        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))
            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(CUBDataset, self).__getitem__(index)
        # if self.bboxes is not None:
        #     # squeeze coordinates of the bounding box to range [0, 1]
        #     width, height = sample.width, sample.height
        #     x, y, w, h = self.bboxes[index]
        #     scale_resize = 500 / width
        #     scale_resize_crop = scale_resize * (375 / 500)
        #     x_rel = scale_resize_crop * x / 375
        #     y_rel = scale_resize_crop * y / 375
        #     w_rel = scale_resize_crop * w / 375
        #     h_rel = scale_resize_crop * h / 375
        #     target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])
        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)
        return sample, target, index
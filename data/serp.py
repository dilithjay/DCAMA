r""" Serpentine zone segmentation dataset """
import os
from glob import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetSerp(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = "val" if split in ["val", "test"] else "trn"
        self.fold = fold
        self.nfolds = 4
        self.nclass = 1
        self.benchmark = "serp"
        self.shot = shot
        self.split_coco = split if split == "val" else "train"
        self.base_path = os.path.join(datapath, "Serp", str(fold))
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.count = 0

        self.class_ids = [0]
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        (
            query_img,
            query_mask,
            support_imgs,
            support_masks,
            query_name,
            support_names,
            class_sample,
            org_qry_imsize,
        ) = self.load_frame()

        query_img = self.transform(query_img)
        # print(query_img.max(), query_img.min(), query_img.mean(), query_img.std())
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(
                query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode="nearest"
            ).squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(
                smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode="nearest"
            ).squeeze()
        support_masks = torch.stack(support_masks)

        batch = {
            "query_img": query_img.float(),
            "query_mask": query_mask.float(),
            "query_name": query_name,
            "org_query_imsize": org_qry_imsize,
            "support_imgs": support_imgs.float(),
            "support_masks": support_masks.float(),
            "support_names": support_names,
            "class_id": torch.tensor(class_sample),
        }

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == "trn" else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        train_imgs = glob(os.path.join(self.base_path, self.split_coco, "*.npy"))
        img_metadata_classwise = {
            0: list(map(lambda x: os.path.relpath(x, os.path.dirname(os.path.dirname(x))), train_imgs))
        }
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, "annotations", name)
        mask = torch.tensor(np.load(mask_path))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_region = os.path.basename(query_name).split("_")[0]
        query_img = np.load(os.path.join(self.base_path, query_name)).astype(float).transpose(1, 2, 0)
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.shape

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = set()
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if os.path.basename(support_name).startswith(query_region) and query_name != support_name:
                support_names.add(support_name)
            if len(support_names) == self.shot:
                break

        support_names = list(support_names)
        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_img = np.load(os.path.join(self.base_path, support_name)).astype(float).transpose(1, 2, 0)
            support_imgs.append(support_img)
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return (
            query_img,
            query_mask,
            support_imgs,
            support_masks,
            query_name,
            support_names,
            class_sample,
            org_qry_imsize,
        )

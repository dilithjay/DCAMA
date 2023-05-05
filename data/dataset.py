r""" Dataloader builder for few-shot semantic segmentation dataset  """
from glob import glob
import os
import numpy as np
from torch.utils.data import DataLoader, Sampler, Dataset
from torchvision import transforms

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS
from data.serp import DatasetSerpMini


class FSSDataset:
    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):
        cls.datasets = {
            "pascal": DatasetPASCAL,
            "coco": DatasetCOCO,
            "fss": DatasetFSS,
            "serp": DatasetSerpMini,
        }

        cls.img_mean = [
            452.36395548,
            669.48234239,
            409.39103663,
            987.94130831,
            2457.23722236,
            2872.30241926,
            3011.18175418,
            3097.38396507,
            1786.85631331,
            929.30668321,
        ]

        cls.img_std = [
            177.24756019,
            144.58550688,
            95.04011083,
            224.49394865,
            485.14565224,
            580.77737498,
            649.43248944,
            622.79759571,
            419.01506965,
            298.34517013,
        ]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=cls.img_mean, std=cls.img_std)]
        )

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        nworker = 0

        dataset = cls.datasets[benchmark](
            cls.datapath,
            fold=fold,
            transform=cls.transform,
            split=split,
            shot=shot,
            use_original_imgsize=cls.use_original_imgsize,
        )
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, num_workers=nworker, pin_memory=True)

        return dataloader

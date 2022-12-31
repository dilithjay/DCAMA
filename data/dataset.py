r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS
from data.serp import DatasetSerp


class FSSDataset:
    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            "pascal": DatasetPASCAL,
            "coco": DatasetCOCO,
            "fss": DatasetFSS,
            "serp": DatasetSerp,
        }

        cls.img_mean = [455, 675, 400, 1000, 2480, 2905, 3040, 3130, 1810, 950]

        cls.img_std = [185, 148, 99, 225, 465, 557, 625, 594, 412, 306]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=cls.img_mean, std=cls.img_std)]
        )

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        nworker = nworker if split == "trn" else 0

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

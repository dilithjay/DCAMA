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
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'serp': DatasetSerp,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.ToTensor()])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold,
                                          transform=cls.transform,
                                          split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, num_workers=nworker,
                                pin_memory=True)

        return dataloader

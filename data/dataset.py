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

        cls.img_mean = [                                                                                
            0.12375696117681859,                                                         
            0.1092774636368323,                                                          
            0.1010855203267882,                                                          
            0.1142398616114001,                                                          
            0.1592656692023089,                                                          
            0.18147236008771792,                                                         
            0.1745740312291377,                                                          
            0.19501607349635292,                                                         
            0.15428468872076637,                                                         
            0.10905050699570007,                                                         
        ]
        cls.img_std = [                                                                                
            0.03958795985905458,                                                         
            0.047778262752410296,                                                        
            0.06636616706371974,                                                         
            0.06358874912497474,                                                         
            0.07744387147984592,                                                         
            0.09101635085921553,                                                         
            0.09218466562387101,                                                         
            0.10164581233948201,                                                         
            0.09991773043519253,                                                         
            0.08780632509122865,                                                         
        ]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        
        cls.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.img_mean, std=cls.img_std)
        ])

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

import numpy as np
from glob import glob
from tqdm import tqdm


paths = glob("/home/dilith/Projects/DCAMA/datasets/Serp/*/*/*.npy")
for path in tqdm(paths):
    img = np.load(path)
    idx = img[1] + img[5] / (img[7] + 0.0001) + img[5] / (img[8] + 0.0001) + img[7] - img[2]

    max_z = 3
    idx = (idx - idx.mean()) / idx.std()
    idx = (np.clip(idx, -max_z, max_z) + max_z) / (2 * max_z)

    img = np.concatenate([img, [idx]])
    np.save(path, img)

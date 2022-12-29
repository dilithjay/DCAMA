from glob import glob
import os
import shutil
import random 


def partition (list_in, n=5, seed=42):
    random.seed(seed)
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


base_path = r"D:\FYP\Code\Explore\data\train"
dest_path = r"D:\FYP\Code\DCAMA\datasets\Serp"
images = glob(os.path.join(base_path, "*.npy"))
masks = glob(os.path.join(base_path, "*.png"))

img_folds = partition(images)
mask_folds = partition(masks)

for i in range(5):
    path = os.path.join(dest_path, str(i))
    os.makedirs(os.path.join(path, 'train'))
    os.makedirs(os.path.join(path, 'val'))
    os.makedirs(os.path.join(path, 'annotations', 'train'))
    os.makedirs(os.path.join(path, 'annotations', 'val'))
    
    img_folds_cp = img_folds.copy()
    mask_folds_cp = mask_folds.copy()
    img_fold = img_folds_cp.pop(i)
    mask_fold = mask_folds_cp.pop(i)
    
    for img in img_fold:
        shutil.copyfile(img, os.path.join(path, 'val', os.path.basename(img)))
    for mask in mask_fold:
        shutil.copyfile(mask, os.path.join(path, 'annotations', 'val', os.path.basename(mask)))
    for img_fold in img_folds_cp:
        for img in img_fold:
            shutil.copyfile(img, os.path.join(path, 'train', os.path.basename(img)))
    for mask_fold in mask_folds_cp:
        for mask in mask_fold:
            shutil.copyfile(mask, os.path.join(path, 'annotations', 'train', os.path.basename(mask)))

import numpy as np
from glob import glob
from tqdm import tqdm
import sys


expression_ = ['c8', '+', 'c2', '*', 'c8', '-', 'c7', '-', 'c8', '+', 'c4', '=']


def eval_expression(exp: list, image: np.ndarray = None):
    expression = ""

    for token in exp:
        if token[0] == "c":
            channel = eval(token[1:])
            expression += f"(image[{channel}] + 0.0001)"  # To prevent divide by zero
        elif token == "sq":
            expression += "**2"
        elif token == "sqrt":
            expression += "**0.5"
        elif token == "=":
            break
        else:
            expression += token

    return eval(expression)


paths = glob("datasets/Serp/*/*/*.npy")
for path in tqdm(paths):
    img = np.load(path)
    idx = eval_expression(expression_, img)

    max_z = 3
    idx = (idx - idx.mean()) / idx.std()
    idx = (np.clip(idx, -max_z, max_z) + max_z) / (2 * max_z)

    img = np.concatenate([img, [idx]])
    np.save(path, img)

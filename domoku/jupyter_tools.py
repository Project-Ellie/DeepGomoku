import numpy as np


def print_bin(binary_sample, combine=False):
    binary_sample = np.squeeze(binary_sample)
    print(f'shape: {binary_sample.shape}')
    if combine:
        print(np.rollaxis(binary_sample, 2, 0)[0] + 2 * np.rollaxis(binary_sample, 2, 0)[1])
    else:
        print(np.rollaxis(binary_sample, 2, 0)[0])
        print()
        print(np.rollaxis(binary_sample, 2, 0)[1])


def vis(tensor, scale=1):
    npa = np.squeeze(tensor[0, :, :, 0].numpy())
    print((npa*scale).astype(int))

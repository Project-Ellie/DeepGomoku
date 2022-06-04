import numpy as np


def print_bin(binary_sample, combine=False):
    binary_sample = np.squeeze(binary_sample)
    print(f'shape: {binary_sample.shape}')
    if combine:
        image = sum([(i+1) * np.rollaxis(binary_sample, 2, 0)[i] for i in range(binary_sample.shape[-1])])
        print(image)
    else:
        for i in range(binary_sample.shape[-1]):
            print(np.rollaxis(binary_sample, 2, 0)[i])
            print()


def vis(tensor, scale=1):
    npa = np.squeeze(tensor[0, :, :, 0].numpy())
    print((npa*scale).astype(int))

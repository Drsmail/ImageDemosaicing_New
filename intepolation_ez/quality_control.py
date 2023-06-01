from skimage.metrics import structural_similarity as SSIM

import numpy as np
from scipy import signal

def PSNR(original, compressed, **kwargs):

    mse = np.mean((original - compressed) ** 2, **kwargs)
    max_pixel = 1

    # mse[mse == 0] = np.inf # MSE is zero means no noise is present in the signal .
    # psnr = 20 * np.log10(max_pixel / np.sqrt(mse[mse != np.inf]))

    mse = np.mean(mse) # MSE is zero means no noise is present in the signal .
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr

def rgb2gray(rgb):
    rgb_swapped = np.swapaxes(rgb, 0, -1)
    return np.dot(rgb_swapped[:3,... ], [0.2989, 0.5870, 0.1140])

def my_SSIM(original, compressed):

    original = rgb2gray(original)
    compressed = rgb2gray(compressed)

    a = SSIM(original,compressed)

    K1 = 0.01
    K2 = 0.03
    max_pixel = 1
    C1 = (K1 * max_pixel) ** 2
    C2 = (K2 * max_pixel) ** 2

    N = original.shape[1] * original.shape[2]

    mu1 = signal.convolve2d(original, np.ones((11, 11)) / 121, mode='same', boundary='symm') / N
    mu2 = signal.convolve2d(compressed, np.ones((11, 11)) / 121, mode='same', boundary='symm') / N

    sigma1 = signal.convolve2d(original ** 2, np.ones((11, 11)) / 121, mode='same', boundary='symm') / N - mu1 ** 2
    sigma2 = signal.convolve2d(compressed ** 2, np.ones((11, 11)) / 121, mode='same', boundary='symm') / N - mu2 ** 2
    sigma12 = signal.convolve2d(original * compressed, np.ones((11, 11)) / 121, mode='same', boundary='symm') / N - mu1 * mu2

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

    ssim = np.mean(numerator / denominator)

    return ssim



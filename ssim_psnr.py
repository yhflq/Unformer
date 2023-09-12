import argparse
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np
import cv2

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    kernelsize = 11  
    sigma = 1.5
    window = cv2.getGaussianKernel(kernelsize, sigma)
    window = np.outer(window, window.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[kernelsize - 1:-kernelsize + 1, kernelsize - 1:-kernelsize + 1]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[kernelsize - 1:-kernelsize + 1, kernelsize - 1:-kernelsize + 1]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[kernelsize - 1:-kernelsize + 1, kernelsize - 1:-kernelsize + 1] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[kernelsize - 1:-kernelsize + 1, kernelsize - 1:-kernelsize + 1] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[kernelsize - 1:-kernelsize + 1, kernelsize - 1:-kernelsize + 1] - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


if __name__ == '__main__':
    print('===> Loading datasets')
    parser = argparse.ArgumentParser(description='Performance')
    parser.add_argument('--input_dir', default='./results/UIEBD/')
    parser.add_argument('--reference_dir', default='./datasets/UIE/UIEBD/test/label/')

    opt = parser.parse_args()
    print(opt)


    im_path = opt.input_dir
    re_path = opt.reference_dir
    avg_psnr = 0
    avg_ssim = 0
    n = 0

    for filename in os.listdir(im_path):
        #print(filename)
        #print(im_path + '/' + filename)
        n = n + 1
        im1 = cv2.imread(im_path + '/' + filename)
        im2 = cv2.imread(re_path + '/' + filename)

        (h, w, c) = im2.shape
        im1 = cv2.resize(im1, (w, h)) 

        score_psnr = psnr(im1, im2)
        score_ssim = calculate_ssim(im1, im2)

        avg_psnr += score_psnr
        avg_ssim += score_ssim
        print(filename)
        print("===> PSNR: {:.4f} dB ".format(score_psnr))
        print("===> SSIM: {:.4f} ".format(score_ssim))


    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))






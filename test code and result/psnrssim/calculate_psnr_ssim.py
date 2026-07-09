import os

import cv2

from psnr_ssim import calculate_psnr, calculate_ssim


gt_path = "../data/LOLv2/Synthetic/Test/Normal"
results_path = "../results/v2sys"

imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))
assert len(imgsName) == len(gtsName)

cumulative_psnr, cumulative_ssim = 0, 0

for i in range(len(imgsName)):
    print("Processing image: %s" % (imgsName[i]))
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)

    cur_psnr = calculate_psnr(res, gt, crop_border=0)
    cur_ssim = calculate_ssim(res, gt, crop_border=0)

    print("PSNR: %.4f, SSIM: %.4f" % (cur_psnr, cur_ssim))
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim

avg_psnr = cumulative_psnr / len(imgsName)
avg_ssim = cumulative_ssim / len(imgsName)

print("\nFinal Results:")
print("Average PSNR: %.4f" % avg_psnr)
print("Average SSIM: %.4f" % avg_ssim)
print("Evaluated on:", results_path)

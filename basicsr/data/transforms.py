import cv2
import random
import numpy as np

def mod_crop(img, scale):
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[: h - h_remainder, : w - w_remainder, ...]
    else:
        raise ValueError(f"Wrong img ndim: {img.ndim}.")
    return img
def paired_random_crop(img_gts, img_lqs, lq_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(lq_patch_size * scale)
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        print(gt_path)
        raise ValueError(
            f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ",
            f"multiplication of LQ ({h_lq}, {w_lq}).",
        )
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(
            f"LQ ({h_lq}, {w_lq}) is smaller than patch size "
            f"({lq_patch_size}, {lq_patch_size}). "
            f"Please remove {gt_path}."
        )
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)
    img_lqs = [
        v[top : top + lq_patch_size, left : left + lq_patch_size, ...] for v in img_lqs
    ]
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt : top_gt + gt_patch_size, left_gt : left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs
def paired_random_crop_DP(img_lqLs, img_lqRs, img_gts, gt_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqLs, list):
        img_lqLs = [img_lqLs]
    if not isinstance(img_lqRs, list):
        img_lqRs = [img_lqRs]
    h_lq, w_lq, _ = img_lqLs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ",
            f"multiplication of LQ ({h_lq}, {w_lq}).",
        )
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(
            f"LQ ({h_lq}, {w_lq}) is smaller than patch size "
            f"({lq_patch_size}, {lq_patch_size}). "
            f"Please remove {gt_path}."
        )
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)
    img_lqLs = [
        v[top : top + lq_patch_size, left : left + lq_patch_size, ...] for v in img_lqLs
    ]
    img_lqRs = [
        v[top : top + lq_patch_size, left : left + lq_patch_size, ...] for v in img_lqRs
    ]
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt : top_gt + gt_patch_size, left_gt : left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqLs) == 1:
        img_lqLs = img_lqLs[0]
    if len(img_lqRs) == 1:
        img_lqRs = img_lqRs[0]
    return img_lqLs, img_lqRs, img_gts
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5
    def _augment(img):
        if hflip:
            cv2.flip(img, 1, img)
        if vflip:
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img
    def _augment_flow(flow):
        if hflip:
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow
    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs
def img_rotate(img, angle, center=None, scale=1.0):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img
def data_augmentation(image, mode):
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image)
    elif mode == 3:
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        out = np.rot90(image, k=2)
    elif mode == 5:
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(image, k=3)
    elif mode == 7:
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception("Invalid choice of image transformation")
    return out
def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out

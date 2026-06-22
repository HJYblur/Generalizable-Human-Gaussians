###########################################
# imports
###########################################

import argparse
import glob
import os
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import skimage.metrics
import torch
from lpips import LPIPS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.default_config import HumanConfig

# USAGE: python metrics/compute_metrics.py
###########################################

tmp_ours = './outputs/metrics/eval_img_dilate_11_inpaint_3_black_input_2048/pred'
tmp_gt = './outputs/metrics/eval_img_dilate_11_inpaint_3_black_input_2048/gt'
tmperr = './outputs/metrics/eval_img_dilate_11_inpaint_3_black_input_2048/error'


def mae(imageA, imageB):
    err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err


def mse(imageA, imageB):
    errImage = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2, 2)
    errImage = np.sqrt(errImage)

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])

    return err, errImage


def _resize_to_square(image, size):
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _apply_aabb_crop(pred, gt, crop_h, crop_w):
    """Center a fixed crop on the GT foreground AABB."""
    h, w = pred.shape[0], pred.shape[1]
    crop_h = min(int(crop_h), h)
    crop_w = min(int(crop_w), w)

    ii, jj = np.where(~(gt == 0).all(-1))
    if ii.size == 0 or jj.size == 0:
        return None

    hmin, hmax = np.min(ii), np.max(ii)
    uu = (crop_h - (hmax + 1 - hmin)) // 2
    vv = crop_h - (hmax - hmin) - uu
    if hmin - uu < 0:
        hmin, hmax = 0, crop_h
    elif hmax + vv > h:
        hmin, hmax = h - crop_h, h
    else:
        hmin, hmax = hmin - uu, hmax + vv

    wmin, wmax = np.min(jj), np.max(jj)
    uu = (crop_w - (wmax + 1 - wmin)) // 2
    vv = crop_w - (wmax - wmin) - uu
    if wmin - uu < 0:
        wmin, wmax = 0, crop_w
    elif wmax + vv > w:
        wmin, wmax = w - crop_w, w
    else:
        wmin, wmax = wmin - uu, wmax + vv

    pred_crop = pred[hmin:hmax, wmin:wmax]
    gt_crop = gt[hmin:hmax, wmin:wmax]
    if pred_crop.shape[0] != crop_h or pred_crop.shape[1] != crop_w:
        raise ValueError(
            f"Unexpected crop size {pred_crop.shape[:2]}; expected ({crop_h}, {crop_w})"
        )
    return pred_crop, gt_crop


def _prepare_eval_pair(pred, gt, use_crop, image_size, crop_h, crop_w):
    if use_crop:
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        return _apply_aabb_crop(pred, gt, crop_h=crop_h, crop_w=crop_w)

    pred = _resize_to_square(pred, image_size)
    gt = _resize_to_square(gt, image_size)
    return pred, gt


def func(g_path, t_path, mask, use_crop=False, image_size=1024, crop_h=1000, crop_w=500):
    psnrs, ssims, mses, maes = [], [], [], []

    humans = set()
    for file_name in os.listdir(g_path):
        file_name_split = file_name[:-4].split('_')
        human_name = file_name_split[0]
        humans.add(human_name)

    humans = list(humans)
    humans.sort()

    for human_idx in range(len(humans)):
        human = humans[human_idx]

        for angle in [3, 8, 13]:
            sample_name = '{}_{}_novel00'.format(human, str(angle).zfill(3))
            print(sample_name)

            ours_path = os.path.join(g_path, '{}_{}_novel00.jpg'.format(human, str(angle).zfill(3)))
            g = imageio.imread(ours_path).astype('float32') / 255.

            gt_path = os.path.join(t_path, '{}_{}/0.jpg'.format(human, str(angle).zfill(3)))
            t = imageio.imread(gt_path).astype('float32') / 255.

            prepared = _prepare_eval_pair(
                g,
                t,
                use_crop=use_crop,
                image_size=image_size,
                crop_h=crop_h,
                crop_w=crop_w,
            )
            if prepared is None:
                print(f"skip {sample_name}: empty foreground for crop")
                continue
            g, t = prepared

            mseValue, errImg = mse(g, t)

            errImg = (errImg * 255.0).astype(np.uint8)
            errImg = cv2.applyColorMap(errImg, cv2.COLORMAP_JET)

            subject_angle_name = '{}_{}_novel00.png'.format(human, str(angle).zfill(3))
            cv2.imwrite(os.path.join(tmperr, subject_angle_name), errImg)

            mseValue_ours_gt, errImg_ours_gt = mse(g, t)
            maeValue = mae(g, t)
            psnr = 10 * np.log10((1 ** 2) / mseValue_ours_gt)

            imageio.imsave("{}/{}_source.png".format(tmp_ours, sample_name), (g * 255).astype('uint8'))
            imageio.imsave("{}/{}_target.png".format(tmp_gt, sample_name), (t * 255).astype('uint8'))

            psnrs += [psnr]
            ssims += [skimage.metrics.structural_similarity(g, t, channel_axis=2, data_range=1)]
            mses += [mseValue]

    return np.asarray(psnrs), np.asarray(ssims), np.asarray(mses), np.asarray(maes)


def evaluateErr(ours, target, mask, use_crop=False, image_size=1024, crop_h=1000, crop_w=500):
    mode = "AABB crop {}x{}".format(crop_h, crop_w) if use_crop else "full {}x{}".format(image_size, image_size)
    print(f"Evaluation mode: {mode}", flush=True)

    psnrs, ssims, mses, maes = func(
        g_path=ours,
        t_path=target,
        mask=mask,
        use_crop=use_crop,
        image_size=image_size,
        crop_h=crop_h,
        crop_w=crop_w,
    )

    if psnrs.size == 0:
        raise RuntimeError("No valid image pairs were found for metric computation.")

    psnr = psnrs.mean()
    print(f"PSNR mean {psnr}", flush=True)
    ssim = ssims.mean()
    print(f"SSIM mean {ssim}", flush=True)

    lpips = LPIPS(net='alex', version='0.1')
    if torch.cuda.is_available():
        lpips = lpips.cuda()

    g_files = sorted(glob.glob(tmp_ours + '/*_source.png'))
    t_files = sorted(glob.glob(tmp_gt + '/*_target.png'))

    lpipses = []
    for i in range(len(g_files)):
        g = imageio.imread(g_files[i]).astype('float32') / 255.
        t = imageio.imread(t_files[i]).astype('float32') / 255.
        g = 2 * torch.from_numpy(g).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        t = 2 * torch.from_numpy(t).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        if torch.cuda.is_available():
            g = g.cuda()
            t = t.cuda()
        lpipses += [lpips(g, t).item()]
    lpips = np.mean(lpipses)
    print(f"LPIPS Alex Mean {lpips}", flush=True)

    lpips = LPIPS(net='vgg', version='0.1')
    if torch.cuda.is_available():
        lpips = lpips.cuda()

    g_files = sorted(glob.glob(tmp_ours + '/*_source.png'))
    t_files = sorted(glob.glob(tmp_gt + '/*_target.png'))

    lpipses = []
    for i in range(len(g_files)):
        g = imageio.imread(g_files[i]).astype('float32') / 255.
        t = imageio.imread(t_files[i]).astype('float32') / 255.
        g = 2 * torch.from_numpy(g).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        t = 2 * torch.from_numpy(t).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        if torch.cuda.is_available():
            g = g.cuda()
            t = t.cuda()
        lpipses += [lpips(g, t).item()]
    lpips = np.mean(lpipses)
    print(f"LPIPS VGG mean {lpips}", flush=True)

    os.system('python -m pytorch_fid --device cuda {} {}'.format(tmp_ours, tmp_gt))


def _load_metrics_config(config_path):
    cfg = HumanConfig()
    cfg.load(config_path)
    return cfg.get_cfg().metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument('--config-path', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--preds-root', default=None, help='Root directory of predictions')
    parser.add_argument('--target-root', default=None, help='Root directory of GT RGB images')
    parser.add_argument('--mask-root', default=None, help='Root directory of GT masks')
    parser.add_argument('--use-crop', action='store_true', default=None, help='Use AABB-centered crop')
    parser.add_argument('--no-crop', dest='use_crop', action='store_false', help='Use full-image comparison')
    args = parser.parse_args()

    metrics_cfg = _load_metrics_config(args.config_path)
    use_crop = metrics_cfg.use_crop if args.use_crop is None else args.use_crop
    image_size = int(metrics_cfg.image_size)
    crop_h = int(metrics_cfg.crop_height)
    crop_w = int(metrics_cfg.crop_width)

    target = args.target_root or './datasets/THuman/val/img/'
    mask = args.mask_root or './datasets/THuman/val/mask/'
    ours = args.preds_root or './outputs/eval/GHG/'

    print('###############################################', flush=True)

    if not os.path.exists(tmperr):
        os.makedirs(tmperr, exist_ok=True)
    if not os.path.exists(tmp_ours):
        os.makedirs(tmp_ours, exist_ok=True)
    if not os.path.exists(tmp_gt):
        os.makedirs(tmp_gt, exist_ok=True)

    evaluateErr(
        ours,
        target,
        mask,
        use_crop=use_crop,
        image_size=image_size,
        crop_h=crop_h,
        crop_w=crop_w,
    )

    print(ours)

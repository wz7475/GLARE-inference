import glob
import sys
from collections import OrderedDict
import tqdm
from natsort import natsort
import argparse

from torch.nn import Module

import options.options as option
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2

import os
from argparse import ArgumentParser
from typing import List, Tuple

from PIL import Image


def resize_bbox(
    bbox: List[int], original_size: Tuple[int, int], target_size: Tuple[int, int]
):
    """
    Converts and resizes from
    Args:
        bbox:
        original_size:
        target_size:

    Returns:

    """
    l = bbox[0]
    t = bbox[1]
    a = bbox[2]
    b = bbox[3]
    return [
        max((l / original_size[0]) * target_size[0], 0),
        max((t / original_size[1]) * target_size[1], 0),
        min((a / original_size[0]) * target_size[0], target_size[0]),
        min((b / original_size[1]) * target_size[1], target_size[1]),
    ]


def process_image(
    img_path: str,
    annotation_path_in: str,
    annotation_path_out: str,
    target_size: Tuple[int, int] = (640, 640),
) -> None:
    with Image.open(img_path) as img:
        img_orig_size = img.size
    new_annotation_lines = []
    print(f"read {annotation_path_in}")
    with open(annotation_path_in) as fp:
        split_lines = [line.split(",") for line in fp.readlines()]
        for elements in split_lines:
            bbox = [int(x) for x in elements[1:5]]
            resized_bbox = resize_bbox(bbox, img_orig_size, target_size)
            updated_elements = (
                elements[:1] + [str(x) for x in resized_bbox] + elements[5:]
            )
            new_annotation_lines.append(",".join(updated_elements))
    with open(annotation_path_out, "w") as fp:
        fp.writelines(new_annotation_lines)
    print(f"saved {annotation_path_out}")


def process_dir(input_dir: str, output_dir: str) -> None:
    img_paths = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.split(".")[-1].lower().lower() in ["jpg", "jpeg", "png", "ppm"]
    ]
    for img_path in img_paths:
        annotation_file_name = f"{os.path.basename(img_path).lower()}.txt"
        process_image(
            img_path,
            os.path.join(input_dir, annotation_file_name),
            os.path.join(output_dir, annotation_file_name)
        )



def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get('NORMAL'))


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result


def auto_padding(img, times=16):
    # img: numpy image with shape H*W*C

    h, w, _ = img.shape
    h1, w1 = (times - h % times) // 2, (times - w % times) // 2
    h2, w2 = (times - h % times) - h1, (times - w % times) - w1
    img = cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_REFLECT)
    return img, [h1, h2, w1, w2]

def inference_for_image(input_path: str, output_path: str, opt: dict, model: Module) -> None:
    img = imread(input_path)
    raw_shape = img.shape

    img, padding_params = auto_padding(img)
    his = hiseq_color_cv2_img(img)
    if opt.get("histeq_as_input", False):
        img = his

    lr_t = t(img)
    if opt["datasets"]["train"].get("log_low", False):
        lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
    if opt.get("concat_histeq", False):
        his = t(his)
        lr_t = torch.cat([lr_t, his], dim=1)
    heat = opt['heat']
    with torch.cuda.amp.autocast():
        sr_t = model.get_sr(lq=lr_t.cuda(), heat=None)

    sr = rgb(torch.clamp(sr_t, 0, 1)[:, :, padding_params[0]:sr_t.shape[2] - padding_params[1],
             padding_params[2]:sr_t.shape[3] - padding_params[3]])

    assert raw_shape == sr.shape
    imwrite(output_path, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default="./code/confs/LOL.yml")
    parser.add_argument("-o", "--output_dir", default="output")
    parser.add_argument("-i", "--input_dir", default="input")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    conf_path = args.opt
    model, opt = load_model(conf_path)
    device = 'cuda:0'
    model.netG = model.netG.to(device)
    model.net_hq=model.net_hq.to(device)
    
    lr_paths = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.split(".")[-1].lower().lower() in ["jpg", "jpeg", "png", "ppm"]
    ]

    os.makedirs(output_dir, exist_ok=True)
    os.system(f"cp {os.path.join(input_dir, '*.txt')} {output_dir}")

    for lr_path in tqdm.tqdm(lr_paths):
        path_out_sr = os.path.join(output_dir, os.path.basename(lr_path))
        inference_for_image(lr_path, path_out_sr, opt, model)


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.2f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


if __name__ == "__main__":
    main()

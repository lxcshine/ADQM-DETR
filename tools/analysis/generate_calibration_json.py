import os
import sys
import glob
import json
import argparse
from argparse import Namespace

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import xml.etree.ElementTree as ET

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))

from torch_inf_obb import load_model, letterbox, get_class_names, IMG_EXTS  # noqa: E402
from engine.data.dataset.mmeval import rbox_iou  # noqa: E402
from adqm.aar import compute_aar  # noqa: E402


# ---------------------------------------------------------------------------
# GT 解析
# ---------------------------------------------------------------------------

def _poly_to_rbox_le90(poly):
    """四边形顶点 (4,2) -> (cx, cy, w, h, theta_rad), le90 约定(w 为长轴, 角度 [0, 180))。
    与 engine/data/dataset/dior_dataset.py 的解析逻辑一致。"""
    (cx, cy), (w, h), angle = cv2.minAreaRect(poly)
    if w <= 1e-6 or h <= 1e-6:
        return None
    if w < h:
        w, h = h, w
        angle += 90.0
    angle = angle % 180.0
    return [float(cx), float(cy), float(w), float(h), float(np.deg2rad(angle))]


def parse_gt_diorr(xml_path, name2idx):
    """DIOR-R XML 标注 -> (boxes_rad (K,5), labels (K,)), 跳过 difficult 与未知类别。"""
    boxes, labels = [], []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall('object'):
        name_el = obj.find('name')
        if name_el is None:
            continue
        name = name_el.text.strip().lower()
        if name not in name2idx:
            continue
        diff_el = obj.find('difficult')
        if diff_el is not None and diff_el.text and int(diff_el.text) == 1:
            continue
        rb = obj.find('robndbox')
        if rb is None:
            continue
        keys = [('x_left_top', 'y_left_top'), ('x_right_top', 'y_right_top'),
                ('x_right_bottom', 'y_right_bottom'), ('x_left_bottom', 'y_left_bottom')]
        pts = []
        for kx, ky in keys:
            ex, ey = rb.find(kx), rb.find(ky)
            if ex is None or ey is None:
                pts = None
                break
            pts.append([float(ex.text), float(ey.text)])
        if pts is None:
            continue
        rbox = _poly_to_rbox_le90(np.array(pts, dtype=np.float32))
        if rbox is None:
            continue
        boxes.append(rbox)
        labels.append(name2idx[name])
    return np.array(boxes, np.float32).reshape(-1, 5), np.array(labels, np.int64)


def parse_gt_dota(txt_path, name2idx):
    """DOTA txt 标注 (x1 y1 ... x4 y4 class difficult) -> (boxes_rad (K,5), labels (K,))。"""
    boxes, labels = [], []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] in ('imagesource', 'gsd'):
                continue
            name = parts[8].lower()
            if name not in name2idx:
                continue
            difficult = parts[9] if len(parts) > 9 else '0'
            if difficult == '1':
                continue
            try:
                poly = np.array([float(v) for v in parts[:8]], dtype=np.float32).reshape(4, 2)
            except ValueError:
                continue
            rbox = _poly_to_rbox_le90(poly)
            if rbox is None:
                continue
            boxes.append(rbox)
            labels.append(name2idx[name])
    return np.array(boxes, np.float32).reshape(-1, 5), np.array(labels, np.int64)


def parse_gt(ann_path, name2idx):
    """按扩展名自动选择 DIOR-R XML / DOTA txt 解析。"""
    if ann_path.endswith('.xml'):
        return parse_gt_diorr(ann_path, name2idx)
    return parse_gt_dota(ann_path, name2idx)


# ---------------------------------------------------------------------------
# 预测-GT 匹配(按类别, 置信度降序贪心, SkewIoU)
# ---------------------------------------------------------------------------

def match_predictions(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, min_iou=0.5):
    """
    按类别贪心匹配: 预测按置信度降序, 每个预测分配 IoU 最大且未占用的 GT。
    Args: 均为 numpy; pred_boxes/gt_boxes 为 (N,5)/(K,5) 弧度 le90 像素坐标。
    Returns: (pred_idx, gt_idx, ious) 等长数组。
    """
    matched_p, matched_g, matched_i = [], [], []
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return (np.array([], np.int64), np.array([], np.int64), np.array([], np.float32))
    for c in np.unique(pred_labels):
        p_idx = np.where(pred_labels == c)[0]
        g_idx = np.where(gt_labels == c)[0]
        if len(g_idx) == 0:
            continue
        ious = rbox_iou(pred_boxes[p_idx], gt_boxes[g_idx])  # (n, K)
        taken = np.zeros(len(g_idx), bool)
        for pi in np.argsort(-pred_scores[p_idx]):
            row = ious[pi].copy()
            row[taken] = -1.0
            gj = int(np.argmax(row))
            if row[gj] >= min_iou:
                taken[gj] = True
                matched_p.append(int(p_idx[pi]))
                matched_g.append(int(g_idx[gj]))
                matched_i.append(float(row[gj]))
    return (np.array(matched_p, np.int64), np.array(matched_g, np.int64),
            np.array(matched_i, np.float32))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(args):
    # ---------- 模型 ----------
    model_args = Namespace(config=args.config, resume=args.resume, device=args.device,
                           input_size=args.input_size)
    cfg, model, input_size = load_model(model_args)
    class_names = get_class_names(cfg)
    name2idx = {c.lower(): i for i, c in enumerate(class_names)}
    use_adqm = getattr(model.model.decoder, 'axis_quality_head', None) is not None
    print(f"Generate calibration JSON: {'ADQM (axis fields included)' if use_adqm else 'baseline (p_cls only)'}")

    # ---------- 图像清单 ----------
    image_files = sorted(
        p for p in glob.glob(os.path.join(args.input, '**', '*'), recursive=True)
        if p.lower().endswith(IMG_EXTS))
    if args.max_images > 0:
        image_files = image_files[:args.max_images]
    if not image_files:
        raise FileNotFoundError(f"No images found under {args.input}")

    to_tensor = T.ToTensor()
    records = []
    n_pred_total, n_images_with_gt = 0, 0

    for i in range(0, len(image_files), args.batch_size):
        batch_files = image_files[i:i + args.batch_size]
        ims = [Image.open(p).convert('RGB') for p in batch_files]
        images = torch.stack([to_tensor(letterbox(im, input_size)) for im in ims]).to(args.device)
        orig_target_sizes = torch.tensor([[im.height, im.width] for im in ims], device=args.device)

        with torch.inference_mode():
            outputs = model(images, orig_target_sizes)
        labels, boxes, scores, axis_scores = (outputs if len(outputs) == 4 else (*outputs, None))

        for j, path in enumerate(batch_files):
            image_id = os.path.splitext(os.path.basename(path))[0]
            # GT 文件: auto 按实际存在优先 XML(DIOR-R), 否则 txt(DOTA)
            if args.gt_format == 'dota':
                ann_path = os.path.join(args.ann_folder, f'{image_id}.txt')
            elif args.gt_format == 'diorr':
                ann_path = os.path.join(args.ann_folder, f'{image_id}.xml')
            else:
                cand_xml = os.path.join(args.ann_folder, f'{image_id}.xml')
                ann_path = cand_xml if os.path.isfile(cand_xml) else \
                    os.path.join(args.ann_folder, f'{image_id}.txt')

            lab = labels[j].cpu().numpy()
            box = boxes[j].cpu().numpy()
            sco = scores[j].cpu().numpy()
            ax = axis_scores[j].cpu().numpy() if axis_scores is not None else None
            keep = sco >= args.thr
            lab, box, sco, ax = lab[keep], box[keep], sco[keep], (ax[keep] if ax is not None else None)
            n_pred_total += len(box)

            if not os.path.isfile(ann_path):
                continue
            gt_boxes, gt_labels = parse_gt(ann_path, name2idx)
            if len(gt_boxes) == 0:
                continue
            n_images_with_gt += 1

            p_idx, g_idx, ious = match_predictions(
                box, lab, sco, gt_boxes, gt_labels, args.min_iou)
            if len(p_idx) == 0:
                continue

            pred_m, gt_m = box[p_idx], gt_boxes[g_idx]
            if use_adqm:
                # AAR(论文公式 9): 仅对匹配成功的正样本计算
                aar_par, aar_perp = compute_aar(
                    torch.from_numpy(pred_m).float(), torch.from_numpy(gt_m).float())
                aar_par = aar_par.numpy()
                aar_perp = aar_perp.numpy()
                # p_cls 由公式 15 数值恢复: p_box / (p_parallel * p_perp)
                p_par, p_perp = ax[p_idx, 0], ax[p_idx, 1]
                p_cls = sco[p_idx] / np.clip(p_par * p_perp, 1e-6, None)
            else:
                aar_par = aar_perp = p_par = p_perp = p_cls = None

            for k in range(len(p_idx)):
                rec = {
                    'image_id': image_id,
                    'pred_box': [float(v) for v in pred_m[k]],
                    'pred_conf': float(sco[p_idx[k]]),
                    'gt_box': [float(v) for v in gt_m[k]],
                    'iou': float(ious[k]),
                }
                if use_adqm:
                    rec.update({
                        'p_cls': float(p_cls[k]),
                        'p_parallel': float(p_par[k]),
                        'p_perp': float(p_perp[k]),
                        'aar_parallel': float(aar_par[k]),
                        'aar_perp': float(aar_perp[k]),
                    })
                records.append(rec)

        if (i // args.batch_size) % 20 == 0 or i + args.batch_size >= len(image_files):
            print(f"[{min(i + args.batch_size, len(image_files))}/{len(image_files)}] "
                  f"positive samples so far: {len(records)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(records, f)

    print(f"\nDone. saved {len(records)} positive samples -> {args.output}")
    print(f"  images: {len(image_files)} (with GT: {n_images_with_gt})")
    print(f"  predictions(>= {args.thr}): {n_pred_total}, matched (IoU >= {args.min_iou}): {len(records)}")
    if records:
        mean_iou = float(np.mean([r['iou'] for r in records]))
        print(f"  mean IoU of matched pairs: {mean_iou:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate calibration JSON (paper Sec. 4.4)')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True, help='checkpoint (.pth)')
    parser.add_argument('-i', '--input', type=str, required=True, help='image directory')
    parser.add_argument('-a', '--ann-folder', type=str, required=True,
                        help='GT annotation folder (DIOR-R XML or DOTA txt)')
    parser.add_argument('-o', '--output', type=str, required=True, help='output JSON path')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-t', '--thr', type=float, default=0.25,
                        help='confidence threshold (p_box for ADQM / p_cls for baseline)')
    parser.add_argument('--min-iou', type=float, default=0.5, help='positive matching threshold')
    parser.add_argument('--input-size', type=int, default=800)
    parser.add_argument('--max-images', type=int, default=-1)
    parser.add_argument('--gt-format', type=str, default='auto', choices=['auto', 'diorr', 'dota'])
    args = parser.parse_args()
    main(args)

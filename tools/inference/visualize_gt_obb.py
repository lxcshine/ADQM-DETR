"""
DIOR-R GT（真值）可视化脚本。

读取 XML 标注（Oriented Bounding Boxes），在原始图像上绘制 GT OBB：
    - 绿色框：普通 GT 目标
    - 黄色框 + 类名后缀 '*'：difficult=1 的目标
输出文件名与推理结果图同名（如 11726.jpg），可直接与下面目录同名对照：
    outputs/infer_diorr_adqm/vis/11726.jpg      (ADQM 检测结果)
    outputs/infer_diorr_baseline/vis/11726.jpg  (baseline 检测结果)
    outputs/gt_diorr/11726.jpg                  (GT 本脚本)

用法:
    # 全量测试集 GT
    python tools/inference/visualize_gt_obb.py -i datasets/DIOR-R/JPEGImages-test -o outputs/gt_diorr

    # 指定清单 + 限量
    python tools/inference/visualize_gt_obb.py -i datasets/DIOR-R/JPEGImages-test -o outputs/gt_diorr \
        --ids-file datasets/DIOR-R/ImageSets/Main/test.txt --max-images 200
"""

import os
import glob
import argparse

import cv2
import numpy as np
import xml.etree.ElementTree as ET

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_xml_boxes(xml_path):
    """解析 DIOR-R XML 标注，返回 [(class_name, difficult, corners(4,2) int32), ...]"""
    boxes = []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall('object'):
        name_el = obj.find('name')
        if name_el is None:
            continue
        name = name_el.text.strip().lower()
        diff_el = obj.find('difficult')
        difficult = int(diff_el.text) if diff_el is not None and diff_el.text else 0
        rb = obj.find('robndbox')
        if rb is None:
            continue
        keys = [('x_left_top', 'y_left_top'), ('x_right_top', 'y_right_top'),
                ('x_right_bottom', 'y_right_bottom'), ('x_left_bottom', 'y_left_bottom')]
        pts = []
        valid = True
        for kx, ky in keys:
            ex, ey = rb.find(kx), rb.find(ky)
            if ex is None or ey is None:
                valid = False
                break
            pts.append([float(ex.text), float(ey.text)])
        if not valid:
            continue
        boxes.append((name, difficult, np.array(pts, dtype=np.int32)))
    return boxes


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 图像清单: --ids-file 优先，否则扫描目录
    if args.ids_file:
        with open(args.ids_file, 'r', encoding='utf-8') as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        ids = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(args.img_folder, '**', '*'), recursive=True)
            if p.lower().endswith(IMG_EXTS))
    if args.max_images > 0:
        ids = ids[:args.max_images]
    if not ids:
        raise FileNotFoundError(f'No images found under {args.img_folder}')

    n_done, n_missing_xml, n_boxes, n_difficult = 0, 0, 0, 0
    for img_id in ids:
        img_path = os.path.join(args.img_folder, f'{img_id}.jpg')
        xml_path = os.path.join(args.ann_folder, f'{img_id}.xml')
        if not os.path.isfile(img_path) or not os.path.isfile(xml_path):
            n_missing_xml += 1
            continue

        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im is None:
            n_missing_xml += 1
            continue

        for name, difficult, corners in parse_xml_boxes(xml_path):
            n_boxes += 1
            if difficult:
                n_difficult += 1
            # GT 约定: 绿色; difficult=1 用黄色并加 '*' 后缀
            color = (0, 255, 255) if difficult else (0, 255, 0)
            cv2.polylines(im, [corners], isClosed=True, color=color, thickness=2)
            label = f"{name}*" if difficult else name
            x, y = int(corners[0][0]), int(corners[0][1])
            cv2.putText(im, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(args.output_dir, f'{img_id}.jpg'), im)
        n_done += 1
        if n_done % 500 == 0:
            print(f'[{n_done}/{len(ids)}] gt boxes: {n_boxes}')

    print(f'\nDone. {n_done} images saved to {args.output_dir}')
    print(f'  GT boxes total: {n_boxes} (difficult: {n_difficult})')
    if n_missing_xml:
        print(f'  skipped (image/xml missing): {n_missing_xml}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DIOR-R GT visualization (OBB)')
    parser.add_argument('-i', '--img-folder', type=str, required=True)
    parser.add_argument('-a', '--ann-folder', type=str,
                        default='datasets/DIOR-R/Annotations/Oriented Bounding Boxes')
    parser.add_argument('-o', '--output-dir', type=str, default='outputs/gt_diorr')
    parser.add_argument('--ids-file', type=str, help='image id list, e.g. datasets/DIOR-R/ImageSets/Main/test.txt')
    parser.add_argument('--max-images', type=int, default=-1)
    args = parser.parse_args()
    main(args)

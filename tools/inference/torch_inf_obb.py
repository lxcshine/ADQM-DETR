import os
import sys
import time
import glob
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig
from engine.misc.vis_obb import get_rotated_rect

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# 20 色调色板（DIOR-R 20 类）
PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (128, 0, 255), (0, 128, 255), (255, 128, 0), (0, 255, 128),
] * 2


def rbox2qbox_np(cx, cy, w, h, angle_rad):
    """(cx, cy, w, h, theta) -> 8 顶点坐标 [x1, y1, x2, y2, x3, y3, x4, y4]"""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    wx, wy = w / 2 * cos_a, w / 2 * sin_a
    hx, hy = -h / 2 * sin_a, h / 2 * cos_a
    return [
        cx - wx - hx, cy - wy - hy,
        cx + wx - hx, cy + wy - hy,
        cx + wx + hx, cy + wy + hy,
        cx - wx + hx, cy - wy + hy,
    ]


def letterbox(im_pil, size):
    """keep-ratio 缩放 + 右下角 padding（与训练/验证 ResizeOBB(keep_ratio, pad_to_size) 一致）"""
    w, h = im_pil.size
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    im_resized = im_pil.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    canvas.paste(im_resized, (0, 0))
    return canvas


def get_class_names(cfg):
    """从配置的数据集类型解析类别名，失败时回退为 class{i}"""
    num_classes = int(getattr(cfg.postprocessor, 'num_classes', 80))
    try:
        ds_cfg = cfg.yaml_cfg.get('val_dataloader', {}).get('dataset', {})
        ds_type = ds_cfg.get('type', '')
        if ds_type:
            import engine.data.dataset as ds_mod
            ds_cls = getattr(ds_mod, ds_type, None)
            if ds_cls is not None and hasattr(ds_cls, 'CLASSES'):
                return [str(c) for c in ds_cls.CLASSES]
    except Exception:
        pass
    return [f'class{i}' for i in range(num_classes)]


class InferModel(nn.Module):
    """backbone -> encoder -> decoder(含 AxisQualityHead) -> PostProcessorOBB"""

    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def load_model(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    cfg.model.load_state_dict(state)

    model = InferModel(cfg).to(args.device)
    model.eval()

    input_shape = getattr(cfg.postprocessor, 'input_shape', None)
    input_size = int(input_shape[0]) if input_shape else args.input_size
    return cfg, model, input_size


def postprocess_one(labels, boxes, scores, axis_scores, thr):
    """阈值过滤（基于 p_box），返回 numpy 结果"""
    lab = labels.cpu().numpy()
    box = boxes.cpu().numpy()
    sco = scores.cpu().numpy()
    ax = axis_scores.cpu().numpy() if axis_scores is not None else None
    keep = sco >= thr
    return lab[keep], box[keep], sco[keep], (ax[keep] if ax is not None else None)


def visualize(im_cv, labels, boxes, scores, axis, class_names):
    """在原图上绘制 OBB 检测结果，标注 p_box / p_parallel / p_perp"""
    for j in range(len(boxes)):
        cx, cy, w, h, angle_rad = boxes[j]
        rect = get_rotated_rect(cx, cy, w, h, angle_rad)
        color = PALETTE[int(labels[j]) % len(PALETTE)]
        cv2.polylines(im_cv, [rect], isClosed=True, color=color, thickness=2)
        cls_name = class_names[int(labels[j])] if int(labels[j]) < len(class_names) else str(labels[j])
        if axis is not None:
            p_par, p_perp = axis[j, 0], axis[j, 1]
            text = f"{cls_name} {scores[j]:.2f}|{p_par:.2f}|{p_perp:.2f}"
        else:
            text = f"{cls_name} {scores[j]:.2f}"
        x, y = rect[0][0], rect[0][1]
        cv2.putText(im_cv, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return im_cv


def save_axis_txt(path, image_id, labels, boxes, scores, axis, class_names):
    """每个检测框: (cx, cy, w, h, theta, p_box, p_parallel, p_perp, s_parallel, s_perp)"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# image_id: {image_id}\n")
        f.write("# label cx cy w h theta(rad) p_box p_parallel p_perp s_parallel s_perp\n")
        for j in range(len(boxes)):
            cx, cy, w, h, angle_rad = boxes[j]
            cls_name = class_names[int(labels[j])] if int(labels[j]) < len(class_names) else str(labels[j])
            if axis is not None:
                p_par, p_perp, s_par, s_perp = axis[j]
                f.write(f"{cls_name} {cx:.2f} {cy:.2f} {w:.2f} {h:.2f} {angle_rad:.6f} "
                        f"{scores[j]:.6f} {p_par:.6f} {p_perp:.6f} {s_par:.6f} {s_perp:.6f}\n")
            else:
                f.write(f"{cls_name} {cx:.2f} {cy:.2f} {w:.2f} {h:.2f} {angle_rad:.6f} "
                        f"{scores[j]:.6f} nan nan nan nan\n")


def write_dota_lines(dota_dir, image_id, labels, boxes, scores, class_names):
    """追加 DOTA Task1 提交格式: image_id score x1 y1 x2 y2 x3 y3 x4 y4（每类一个文件）"""
    per_class = {}
    for j in range(len(boxes)):
        cx, cy, w, h, angle_rad = boxes[j]
        qbox = rbox2qbox_np(cx, cy, w, h, angle_rad)
        cls_name = class_names[int(labels[j])] if int(labels[j]) < len(class_names) else str(labels[j])
        per_class.setdefault(cls_name, []).append(
            f"{image_id} {scores[j]:.6f} " + " ".join(f"{v:.2f}" for v in qbox))
    for cls_name, lines in per_class.items():
        with open(os.path.join(dota_dir, f"Task1_{cls_name}.txt"), 'a', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")


def run_inference(args):
    cfg, model, input_size = load_model(args)
    class_names = get_class_names(cfg)

    vis_dir = os.path.join(args.output_dir, 'vis')
    axis_dir = os.path.join(args.output_dir, 'axis')
    dota_dir = os.path.join(args.output_dir, 'dota')
    for d in (vis_dir, axis_dir, dota_dir):
        os.makedirs(d, exist_ok=True)

    to_tensor = T.ToTensor()

    # ---------- 单张图像 ----------
    if os.path.isfile(args.input):
        image_files = [args.input]
    else:
        image_files = sorted(
            p for p in glob.glob(os.path.join(args.input, '**', '*'), recursive=True)
            if p.lower().endswith(IMG_EXTS))
        if args.max_images > 0:
            image_files = image_files[:args.max_images]
        if not image_files:
            raise FileNotFoundError(f"No images found under {args.input}")

    use_adqm = getattr(model.model.decoder, 'axis_quality_head', None) is not None
    print(f"Inference on {len(image_files)} image(s), threshold={args.thr}, "
          f"{'p_box = p_cls * p_parallel * p_perp (ADQM Eq.15)' if use_adqm else 'p_cls (baseline, no ADQM)'}")

    n_det_total = 0
    for i in range(0, len(image_files), args.batch_size):
        batch_files = image_files[i:i + args.batch_size]
        ims = [Image.open(p).convert('RGB') for p in batch_files]
        images = torch.stack([to_tensor(letterbox(im, input_size)) for im in ims]).to(args.device)
        orig_target_sizes = torch.tensor(
            [[im.height, im.width] for im in ims], device=args.device)  # [h, w] 与训练评估一致

        with torch.inference_mode():
            outputs = model(images, orig_target_sizes)

        labels, boxes, scores, axis_scores = (outputs if len(outputs) == 4 else (*outputs, None))

        for j, (path, im) in enumerate(zip(batch_files, ims)):
            lab, box, sco, ax = postprocess_one(
                labels[j], boxes[j], scores[j], axis_scores[j] if axis_scores is not None else None,
                args.thr)
            n_det_total += len(box)

            image_id = os.path.splitext(os.path.basename(path))[0]
            # 检测结果可视化（保存目录下每一张图像的检测结果图）
            im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            im_cv = visualize(im_cv, lab, box, sco, ax, class_names)
            cv2.imwrite(os.path.join(vis_dir, f"{image_id}.jpg"), im_cv)
            # 轴级信息
            save_axis_txt(os.path.join(axis_dir, f"{image_id}.txt"), image_id, lab, box, sco, ax, class_names)
            # DOTA 提交格式
            write_dota_lines(dota_dir, image_id, lab, box, sco, class_names)

        if (i // args.batch_size) % 10 == 0 or i + args.batch_size >= len(image_files):
            print(f"[{i + len(batch_files)}/{len(image_files)}] detections(p_box>={args.thr}): {n_det_total}")

    print(f"\nDone. {len(image_files)} images, {n_det_total} detections.")
    print(f"  visualization : {vis_dir}")
    print(f"  axis-level    : {axis_dir}  (cx, cy, w, h, theta, p_box, p_parallel, p_perp, s_parallel, s_perp)")
    print(f"  DOTA Task1    : {dota_dir}")


def benchmark(args):
    """对比添加 AxisQualityHead 前后的端到端推理延迟（预期差异 < 0.5 ms）"""
    cfg, model, input_size = load_model(args)
    device = args.device

    images = torch.rand(1, 3, input_size, input_size, device=device)
    sizes = torch.tensor([[input_size, input_size]], device=device)
    use_cuda = device.startswith('cuda') and torch.cuda.is_available()

    decoder = model.model.decoder
    axis_head = getattr(decoder, 'axis_quality_head', None)

    def measure(iters):
        with torch.inference_mode():
            for _ in range(args.bench_warmup):
                model(images, sizes)
            if use_cuda:
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    model(images, sizes)
                end.record()
                torch.cuda.synchronize()
                return start.elapsed_time(end) / iters
            t0 = time.perf_counter()
            for _ in range(iters):
                model(images, sizes)
            return (time.perf_counter() - t0) * 1000.0 / iters

    t_full = measure(args.bench_iters)

    if axis_head is not None:
        decoder.axis_quality_head = None  # 移除轴头 -> baseline
        t_base = measure(args.bench_iters)
        decoder.axis_quality_head = axis_head  # 恢复
        diff = t_full - t_base
    else:
        t_base = t_full
        diff = 0.0
        print("[Benchmark] model has no AxisQualityHead (adqm disabled?), diff = 0")

    print(f"\n[Benchmark] input 1x3x{input_size}x{input_size}, "
          f"warmup {args.bench_warmup}, iters {args.bench_iters}, device {device}")
    print(f"  with AxisQualityHead   : {t_full:.4f} ms")
    print(f"  without AxisQualityHead: {t_base:.4f} ms")
    print(f"  diff                   : {diff:.4f} ms (expected < 0.5 ms) -> "
          f"{'PASS' if diff < 0.5 else 'FAIL'}")


def main(args):
    if args.benchmark:
        benchmark(args)
    else:
        if not args.input:
            raise ValueError('-i/--input is required (image file or directory) unless --benchmark')
        run_inference(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OBB + ADQM inference (paper Eq.15)')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True, help='checkpoint (.pth)')
    parser.add_argument('-i', '--input', type=str, help='image file or image directory')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-o', '--output-dir', type=str, default='./outputs/infer_obb_adqm')
    parser.add_argument('-t', '--thr', type=float, default=0.35, help='p_box threshold')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=800,
                        help='letterbox size (ignored when PostProcessorOBB.input_shape is set)')
    parser.add_argument('--max-images', type=int, default=-1, help='limit number of images (dir mode)')
    parser.add_argument('--benchmark', action='store_true', help='latency comparison with/without AxisQualityHead')
    parser.add_argument('--bench-warmup', type=int, default=20)
    parser.add_argument('--bench-iters', type=int, default=100)
    args = parser.parse_args()
    main(args)

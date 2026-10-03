"""Split DOTA original images (and optional DOTA-format annotations) into patches.

Produces the layout expected by the released DOTA-v1.0 configs:

    split_ss_dota/
    └── test/  (or trainval/)
        ├── images/   {orig_stem}__{x}___{y}.png
        └── annfiles/ {orig_stem}__{x}___{y}.txt

The patch naming `{orig_stem}__{x}___{y}` is what the offline evaluator's
merge code (DotaAutoEvaluatorOffline.merge_results) parses back into the
patch origin (x, y) in the original image, so keep this exact format.

An object is assigned to the patch that contains its polygon center; the
polygon coordinates are shifted by the patch origin. Test images without
annotations get an empty matching .txt file per patch.

Usage (evaluation only, test split):
    python tools/dataset/split_dota.py ^
        --img-dir datasets/DOTA-v1.0/test/images ^
        --out-dir datasets/DOTA-v1.0/split_ss_dota/test ^
        --subsize 1024 --gap 200 --workers 8

Usage (trainval split with annotations):
    python tools/dataset/split_dota.py ^
        --img-dir datasets/DOTA-v1.0/train/images ^
        --ann-dir datasets/DOTA-v1.0/train/labelTxt-v1.0/labelTxt ^
        --out-dir datasets/DOTA-v1.0/split_ss_dota/trainval ^
        --subsize 1024 --gap 200 --workers 8
"""
import argparse
import os
from multiprocessing import Pool

from PIL import Image

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--img-dir', required=True, help='original full-size images')
    parser.add_argument('--ann-dir', default=None,
                        help='DOTA-format labelTxt dir (omit for unlabeled test split)')
    parser.add_argument('--out-dir', required=True,
                        help='output dir; creates <out-dir>/images and <out-dir>/annfiles')
    parser.add_argument('--subsize', type=int, default=1024, help='patch size')
    parser.add_argument('--gap', type=int, default=200, help='overlap between patches')
    parser.add_argument('--workers', type=int, default=8, help='process workers')
    return parser.parse_args()


def iter_origins(length, sub, gap):
    """Patch start positions with edge alignment: the last window is shifted
    back so it always has full `sub` size, guaranteeing full coverage."""
    stride = sub - gap
    if length <= sub:
        return [0]
    origins = list(range(0, length - sub + 1, stride))
    if origins[-1] != length - sub:
        origins.append(length - sub)
    return origins


def load_objects(ann_path):
    """Parse a DOTA labelTxt file into [(poly8, class, difficulty), ...]."""
    objects = []
    if not ann_path or not os.path.exists(ann_path):
        return objects
    with open(ann_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            # DOTA format: x1 y1 ... x4 y4 class difficulty (+ optional extras)
            if len(parts) < 9:
                continue  # 'imagesource:' / 'gsd:' headers etc.
            try:
                poly = [float(v) for v in parts[:8]]
            except ValueError:
                continue  # non-numeric line (header written differently)
            objects.append((poly, parts[8], parts[9] if len(parts) > 9 else '0'))
    return objects


def split_one(task):
    img_path, ann_path, out_img_dir, out_ann_dir, sub, gap = task
    stem = os.path.splitext(os.path.basename(img_path))[0]

    img = Image.open(img_path)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    w, h = img.size

    objects = load_objects(ann_path)

    # Images smaller than the patch size are padded (right/bottom) so every
    # patch is exactly sub x sub; origin stays (0, 0) so coordinates match.
    if w < sub or h < sub:
        canvas = Image.new(img.mode, (max(w, sub), max(h, sub)), 0)
        canvas.paste(img, (0, 0))
        img = canvas
        windows = [(0, 0)]
    else:
        windows = [(x, y) for y in iter_origins(h, sub, gap)
                          for x in iter_origins(w, sub, gap)]

    n_patch = 0
    for (x, y) in windows:
        patch_id = f'{stem}__{x}___{y}'
        patch = img.crop((x, y, x + sub, y + sub))
        patch.save(os.path.join(out_img_dir, patch_id + '.png'))

        lines = []
        for poly, cls, diff in objects:
            cx = sum(poly[0::2]) / 8.0
            cy = sum(poly[1::2]) / 8.0
            if x <= cx < x + sub and y <= cy < y + sub:
                shifted = [v - (x if i % 2 == 0 else y) for i, v in enumerate(poly)]
                lines.append(' '.join(f'{v:.2f}' for v in shifted)
                             + f' {cls} {diff}')
        with open(os.path.join(out_ann_dir, patch_id + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        n_patch += 1

    return stem, n_patch


def main():
    args = parse_args()

    out_img_dir = os.path.join(args.out_dir, 'images')
    out_ann_dir = os.path.join(args.out_dir, 'annfiles')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    images = sorted(f for f in os.listdir(args.img_dir)
                    if f.lower().endswith(IMG_EXTS))
    if not images:
        raise SystemExit(f'no images found in {args.img_dir}')

    tasks = []
    for fname in images:
        stem = os.path.splitext(fname)[0]
        ann_path = None
        if args.ann_dir:
            for ext in ('.txt',):
                cand = os.path.join(args.ann_dir, stem + ext)
                if os.path.exists(cand):
                    ann_path = cand
                    break
        tasks.append((os.path.join(args.img_dir, fname), ann_path,
                      out_img_dir, out_ann_dir, args.subsize, args.gap))

    print(f'splitting {len(tasks)} images -> {args.out_dir} '
          f'(subsize={args.subsize}, gap={args.gap}, workers={args.workers})')

    total = 0
    with Pool(args.workers) as pool:
        for i, (stem, n) in enumerate(pool.imap_unordered(split_one, tasks), 1):
            total += n
            if i % 50 == 0 or i == len(tasks):
                print(f'  [{i}/{len(tasks)}] images, {total} patches', flush=True)

    print(f'done: {len(tasks)} images -> {total} patches')


if __name__ == '__main__':
    main()

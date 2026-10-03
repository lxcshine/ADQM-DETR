import os
import json
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# 任务1: 数据准备与加载
# ---------------------------------------------------------------------------

def load_results(filepath):
    """加载测试结果 JSON(支持批量: 单个路径或路径列表), 仅保留正样本。

    Args:
        filepath: str 或 List[str], generate_calibration_json.py 的输出文件。
    Returns:
        List[dict]: 正样本记录列表(成功匹配到 GT, gt_box 与 iou 有效)。
    """
    paths = [filepath] if isinstance(filepath, str) else list(filepath)
    records = []
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for rec in data:
            # 仅保留正样本: 匹配到 GT 且 IoU 有效
            if rec.get('gt_box') is not None and rec.get('iou') is not None and rec['iou'] > 0:
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# 公共分箱统计
# ---------------------------------------------------------------------------

def _bin_statistics(conf, acc, num_bins=10):
    """均匀划分 J=num_bins 个区间 [0,0.1), ..., [0.9,1.0], 统计每区间均值。

    Args:
        conf: np.ndarray (N,), 预测置信度(或轴级质量分数), 范围 [0, 1]
        acc:  np.ndarray (N,), 实际准确率(IoU 或 AAR), 范围 [0, 1]
    Returns:
        bin_centers: (J,) 区间中心
        conf_j: (J,) conf(B_j), 空 bin 为 nan
        acc_j:  (J,) acc(B_j), 空 bin 为 nan
        count_j: (J,) |B_j| 样本数
    """
    conf = np.asarray(conf, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    assert conf.shape == acc.shape, f'conf/acc shape mismatch: {conf.shape} vs {acc.shape}'

    bin_idx = np.clip((conf * num_bins).astype(np.int64), 0, num_bins - 1)  # conf=1.0 -> 最后一个 bin
    bin_centers = (np.arange(num_bins) + 0.5) / num_bins

    conf_j = np.full(num_bins, np.nan)
    acc_j = np.full(num_bins, np.nan)
    count_j = np.zeros(num_bins, dtype=np.int64)
    for j in range(num_bins):
        mask = bin_idx == j
        n = int(mask.sum())
        count_j[j] = n
        if n > 0:
            conf_j[j] = conf[mask].mean()
            acc_j[j] = acc[mask].mean()
    return bin_centers, conf_j, acc_j, count_j


# ---------------------------------------------------------------------------
# 任务2: LaECE(论文公式: LaECE 定义)
# ---------------------------------------------------------------------------

def compute_laece(confs, ious, num_bins=10):
    """计算定位感知期望校准误差 LaECE。

    LaECE = Σ_{j=1}^{J} (|B_j| / N) * |acc(B_j) - conf(B_j)|
    其中 acc 采用 Localization-aware 定义(区间平均 IoU, 连续准确率)。

    Args:
        confs: Tensor/ndarray (N,), 正样本的 pred_conf
        ious:  Tensor/ndarray (N,), 正样本与 GT 的 SkewIoU
    Returns:
        float: LaECE(小数, 乘 100 得百分比)
    """
    confs = np.asarray(confs if not torch.is_tensor(confs) else confs.cpu().numpy(), dtype=np.float64)
    ious = np.asarray(ious if not torch.is_tensor(ious) else ious.cpu().numpy(), dtype=np.float64)
    n = len(confs)
    assert n > 0, 'compute_laece: 至少需要一个正样本'
    _, conf_j, acc_j, count_j = _bin_statistics(confs, ious, num_bins)
    weights = count_j / n
    laece = float(np.nansum(weights * np.abs(acc_j - conf_j)))  # 空 bin 权重 0, 贡献 0
    return laece


# ---------------------------------------------------------------------------
# 任务4: 轴级校准误差(论文 4.4 节最后一段)
# ---------------------------------------------------------------------------

def compute_axis_ece(pred_scores, gt_scores, num_bins=10):
    """计算轴级期望校准误差(p_parallel vs AAR_parallel 或 p_perp vs AAR_perp)。

    ECE_axis = Σ_{j=1}^{J} (|B_j| / N) * |acc_axis(B_j) - conf_axis(B_j)|
    其中 conf_axis 为轴级质量分数均值, acc_axis 为对应 AAR 均值。

    Args:
        pred_scores: Tensor/ndarray (N,), p_parallel 或 p_perp
        gt_scores:   Tensor/ndarray (N,), aar_parallel 或 aar_perp
    Returns:
        float: ECE(小数, 乘 100 得百分比)
    """
    return compute_laece(pred_scores, gt_scores, num_bins=num_bins)


# ---------------------------------------------------------------------------
# 任务3: Reliability Diagram(论文图X)
# ---------------------------------------------------------------------------

def plot_reliability_diagram(baseline_bins, adqm_bins, baseline_laece, adqm_laece,
                             output_path='reliability_diagram.pdf'):
    """绘制基线与 ADQM 的 Reliability Diagram(左右两个子图)。

    Args:
        baseline_bins / adqm_bins: (bin_centers, conf_j, acc_j, count_j),
            分别由 _bin_statistics(baseline conf/iou) 与 (adqm conf/iou) 得到
        baseline_laece / adqm_laece: float, 小数形式的 LaECE
        output_path: 输出文件(PDF, 300 DPI, Times New Roman 8pt)
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 8,
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42,  # TrueType 嵌入(投稿要求)
    })

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2))

    for ax, bins, laece, title, color in (
            (axes[0], baseline_bins, baseline_laece, 'Baseline', '#b8b8b8'),
            (axes[1], adqm_bins, adqm_laece, '+ADQM', '#1f77b4')):
        centers, conf_j, acc_j, count_j = bins
        valid = count_j > 0
        # 柱状图: 每 bin 实际准确率(宽度 0.08), 仅绘制非空 bin
        ax.bar(centers[valid], acc_j[valid], width=0.08, color=color,
               edgecolor='black', linewidth=0.4, label='Accuracy')
        # 完美校准对角线
        ax.plot([0, 1], [0, 1], color='gray', linewidth=0.7, label='Perfect calibration')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence', fontsize=8)
        ax.set_ylabel('Localization accuracy (IoU)', fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_aspect('equal')
        ax.text(0.04, 0.93, f'LaECE = {laece * 100:.1f}', transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='gray', linewidth=0.4, pad=1.5))

    # 基线子图: 红色箭头标注(论文: 预测 0.8, 实际 IoU 约 0.55)已按需求移除

    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Reliability diagram saved -> {output_path}")


# ---------------------------------------------------------------------------
# 任务5: 主流程与汇总表格
# ---------------------------------------------------------------------------

def main(args):
    # ---------- 任务1: 加载 ----------
    baseline_records = load_results(args.baseline_json)
    adqm_records = load_results(args.adqm_json)
    print(f"Loaded positive samples: baseline = {len(baseline_records)}, adqm = {len(adqm_records)}")

    base_confs = torch.tensor([r['pred_conf'] for r in baseline_records])
    base_ious = torch.tensor([r['iou'] for r in baseline_records])
    adqm_confs = torch.tensor([r['pred_conf'] for r in adqm_records])
    adqm_ious = torch.tensor([r['iou'] for r in adqm_records])

    # ---------- 任务2: LaECE ----------
    baseline_laece = compute_laece(base_confs, base_ious)
    adqm_laece = compute_laece(adqm_confs, adqm_ious)
    print(f"Baseline LaECE: {baseline_laece * 100:.1f}")   # 论文参考值: 12.7
    print(f"ADQM LaECE: {adqm_laece * 100:.1f}")           # 论文参考值: 4.3

    # ---------- 任务3: Reliability Diagram ----------
    baseline_bins = _bin_statistics(
        base_confs if not torch.is_tensor(base_confs) else base_confs.numpy(),
        base_ious.numpy())
    adqm_bins = _bin_statistics(adqm_confs.numpy(), adqm_ious.numpy())
    os.makedirs(args.output_dir, exist_ok=True)
    plot_reliability_diagram(
        baseline_bins, adqm_bins, baseline_laece, adqm_laece,
        output_path=os.path.join(args.output_dir, 'reliability_diagram.pdf'))

    # ---------- 任务4: 轴级校准误差(仅 ADQM 有轴级质量估计) ----------
    ece_parallel = ece_perp = None
    if len(adqm_records) > 0 and 'p_parallel' in adqm_records[0]:
        p_parallel = torch.tensor([r['p_parallel'] for r in adqm_records])
        aar_parallel = torch.tensor([r['aar_parallel'] for r in adqm_records])
        p_perp = torch.tensor([r['p_perp'] for r in adqm_records])
        aar_perp = torch.tensor([r['aar_perp'] for r in adqm_records])
        ece_parallel = compute_axis_ece(p_parallel, aar_parallel)
        ece_perp = compute_axis_ece(p_perp, aar_perp)
        print(f"Axis-level ECE (parallel): {ece_parallel * 100:.1f}")  # 论文参考值: 4.5
        print(f"Axis-level ECE (perp): {ece_perp * 100:.1f}")          # 论文参考值: 3.8

    # ---------- 任务5: 汇总表格 ----------
    fmt = lambda v: f"{v * 100:.1f}" if v is not None else '-'
    print("\n| 方法 | LaECE↓ | ECE_parallel↓ | ECE_perp↓ |")
    print("|------|--------|----------------|-----------|")
    print(f"| RiO-DETR (Baseline) | {fmt(baseline_laece)} | - | - |")
    print(f"| ADQM (Ours) | {fmt(adqm_laece)} | {fmt(ece_parallel)} | {fmt(ece_perp)} |")

    print("\n注: 论文参考值(DOTA-1.0 测试集, 全量训练): Baseline 12.7 / ADQM 4.3 / ECE_parallel 4.5 / ECE_perp 3.8")
    print("    在其他数据集或子集模型上数值会不同, 定性结论(ADQM 校准误差显著低于基线)应保持一致。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Confidence calibration analysis (paper Sec. 4.4)')
    parser.add_argument('--baseline-json', type=str, required=True,
                        help='baseline 结果 JSON(generate_calibration_json.py 输出)')
    parser.add_argument('--adqm-json', type=str, required=True,
                        help='ADQM 结果 JSON(generate_calibration_json.py 输出)')
    parser.add_argument('--output-dir', type=str, default='outputs/calibration')
    args = parser.parse_args()
    main(args)

"""
误差模式解耦实验（Error Mode Disentanglement）。

对应论文《Rethinking Localization Quality in Real-Time Oriented Detection》
4.4 节"置信度校准分析"中的误差模式解耦部分：验证 ADQM 的轴级质量分数能否区分
两种退化模式——
  1) 角度偏差导致的短轴崩溃（angular-error-dominant）：
     中心对齐良好但角度偏差较大（高宽比目标），短轴对齐质量骤降；
  2) 中心平移导致的均匀偏移（translation-error-dominant）：
     角度对齐良好但中心偏移较大，两轴对齐质量同步温和下降。

实验流程（与论文设置一致）：
  - 样本池：DIOR-R 测试集上 IoU ∈ [0.5, 0.7] 的匹配正样本（"share an
    identical IoU range"，即定位质量同档，仅误差模式不同）；
  - 平移误差主导：delta_theta < 2° 且 norm_center_disp > 0.15；
  - 角度误差主导：norm_center_disp < 0.05 且 5° ≤ delta_theta ≤ 15° 且 k > 2.0；
  - 每类随机抽取 200 个样本（不足则全部使用）；
  - 对比基线（单分类置信度）与 ADQM（p_box = p_cls × p_parallel × p_perp）
    在两类样本上的置信度表现。

输入：generate_calibration_json.py 产出的两个 JSON（同一测试集、同一筛选协议）。
输出：控制台统计表 + 显著性指标 + error_mode_disentanglement.pdf。

用法：
  python tools/analysis/error_mode_disentanglement.py \
      --baseline-json outputs/calibration/baseline_results.json \
      --adqm-json outputs/calibration/adqm_results.json \
      --output-dir outputs/calibration
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------- 实验常量（论文设置） -----------------------------
IOU_LO, IOU_HI = 0.5, 0.7   # 正样本匹配档位："identical IoU range"
NUM_SAMPLES = 200            # 每类抽取样本数
SEED = 42                    # 随机抽样种子（可复现）


# --------------------------------- 任务 1 -----------------------------------
def load_results(filepath):
    """加载测试结果，仅保留 IoU ∈ [0.5, 0.7] 的匹配正样本。

    对应论文 4.4 节：误差模式解耦在"相同 IoU 区间"的困难样本上进行，
    以排除整体定位质量差异的干扰，凸显误差模式本身的差异。

    Args:
        filepath: JSON 文件路径（generate_calibration_json.py 产出）。

    Returns:
        list[dict]: 满足 IoU 区间的正样本记录。
    """
    with open(filepath, 'r') as f:
        records = json.load(f)

    kept = [r for r in records if IOU_LO <= r['iou'] <= IOU_HI]
    print(f"[load] {Path(filepath).name}: 共 {len(records)} 条记录, "
          f"IoU∈[{IOU_LO},{IOU_HI}] 正样本 {len(kept)} 条")
    return kept


# --------------------------------- 任务 2 -----------------------------------
def compute_error_features(pred_box, gt_box):
    """计算单个样本的误差特征量（对应论文 4.4 节误差模式定义）。

    Args:
        pred_box: [cx, cy, w, h, theta]，原图像素坐标，theta 单位弧度。
        gt_box:   同上。

    Returns:
        dict:
            delta_theta      角度偏差（度, [0, 90]，处理 180° 周期性）
            delta_c          中心点偏移（像素）
            minor_axis       GT 短轴长度（像素）
            norm_center_disp 归一化中心偏移 = delta_c / minor_axis
            k                GT 宽高比 = max(w, h) / min(w, h)
    """
    cx_p, cy_p, _, _, theta_p = pred_box
    cx_g, cy_g, w_g, h_g, theta_g = gt_box

    # 角度偏差: |theta_p - theta_g|, 处理 OBB 的 180° 周期性 -> [0°, 90°]
    delta_theta = abs(theta_p - theta_g)          # rad
    delta_theta = min(delta_theta, np.pi - delta_theta)
    delta_theta = float(np.degrees(delta_theta))

    # 中心点偏移与归一化（相对 GT 短轴长度）
    delta_c = float(np.hypot(cx_p - cx_g, cy_p - cy_g))
    minor_axis = float(min(w_g, h_g))
    norm_center_disp = delta_c / max(minor_axis, 1e-6)

    # 目标宽高比
    k = float(max(w_g, h_g)) / max(minor_axis, 1e-6)

    return dict(delta_theta=delta_theta,
                delta_c=delta_c,
                minor_axis=minor_axis,
                norm_center_disp=norm_center_disp,
                k=k)


# --------------------------------- 任务 3 -----------------------------------
def filter_samples(samples, category, rng, num_samples=NUM_SAMPLES):
    """按论文条件筛选两类困难样本，并随机抽取固定数量。

    筛选条件（论文 4.4 节）：
      translation（平移误差主导）: delta_theta < 2.0° 且 norm_center_disp > 0.15
      angular   （角度误差主导）  : norm_center_disp < 0.05 且 5.0° ≤ delta_theta ≤ 15.0°
                                   且 k > 2.0（高宽比目标, 短轴对角度敏感）
      两类均要求 IoU ∈ [0.5, 0.7]（load_results 已保证, 此处再显式校验一次）。

    Args:
        samples:     load_results 返回的正样本列表。
        category:    'translation' 或 'angular'。
        rng:         random.Random 实例（可复现抽样）。
        num_samples: 每类抽取上限, 默认 200。

    Returns:
        list[dict]: 选中的样本（附加 'features' 字段），不足 num_samples 时全部返回。
    """
    candidates = []
    for s in samples:
        if not (IOU_LO <= s['iou'] <= IOU_HI):
            continue
        f = compute_error_features(s['pred_box'], s['gt_box'])
        if category == 'translation':
            ok = f['delta_theta'] < 2.0 and f['norm_center_disp'] > 0.15
        elif category == 'angular':
            ok = (f['norm_center_disp'] < 0.05
                  and 5.0 <= f['delta_theta'] <= 15.0
                  and f['k'] > 2.0)
        else:
            raise ValueError(f"未知样本类别: {category}")
        if ok:
            r = dict(s)
            r['features'] = f
            candidates.append(r)

    if len(candidates) < num_samples:
        print(f"[filter][{category}] 警告: 仅筛出 {len(candidates)} 个样本 "
              f"(不足 {num_samples}), 使用全部样本继续")
        selected = candidates
    else:
        selected = rng.sample(candidates, num_samples)

    if selected:
        mean_dt = np.mean([r['features']['delta_theta'] for r in selected])
        mean_nd = np.mean([r['features']['norm_center_disp'] for r in selected])
        mean_k = np.mean([r['features']['k'] for r in selected])
        print(f"[filter][{category}] 选中 {len(selected)} 个样本 | "
              f"mean delta_theta={mean_dt:.2f}°, "
              f"norm_center_disp={mean_nd:.3f}, k={mean_k:.2f}")
    return selected


# --------------------------------- 任务 4 -----------------------------------
def compute_statistics(samples_baseline, samples_adqm):
    """统计两类样本的置信度指标（对应论文表X）。

    Args:
        samples_baseline: 基线方法在某误差类别下的样本列表。
        samples_adqm:     ADQM 方法在同类别下的样本列表。

    Returns:
        dict: {'baseline': {...}, 'adqm': {...}, 'n_baseline': int, 'n_adqm': int}
    """
    def _mean(xs):
        return float(np.mean(xs)) if len(xs) > 0 else float('nan')

    stats = {
        'n_baseline': len(samples_baseline),
        'n_adqm': len(samples_adqm),
        'baseline': {'pred_conf': _mean([s['pred_conf'] for s in samples_baseline])},
        'adqm': {
            'pred_conf': _mean([s['pred_conf'] for s in samples_adqm]),
            'p_parallel': _mean([s['p_parallel'] for s in samples_adqm]),
            'p_perp': _mean([s['p_perp'] for s in samples_adqm]),
        },
    }
    return stats


# --------------------------------- 任务 5 -----------------------------------
def significance_analysis(stats_by_cat):
    """统计显著性验证：量化 ADQM 区分两类误差的能力。

    论文预期：
      - baseline_diff < 0.05: 基线单分类置信度无法区分两类误差模式；
      - adqm_diff     > 0.20: ADQM 聚合置信度能显著区分；
      - axis_gap     > 0.50 : 角度误差样本出现"短轴崩溃"（p_perp 骤降）。

    Args:
        stats_by_cat: {'translation': stats, 'angular': stats}

    Returns:
        dict: baseline_diff / adqm_diff / axis_gap
    """
    st, sa = stats_by_cat['translation'], stats_by_cat['angular']
    baseline_diff = abs(sa['baseline']['pred_conf'] - st['baseline']['pred_conf'])
    adqm_diff = abs(sa['adqm']['pred_conf'] - st['adqm']['pred_conf'])
    axis_gap = sa['adqm']['p_parallel'] - sa['adqm']['p_perp']

    print("\n===== 任务5: 统计显著性验证 =====")
    print(f"Baseline confidence gap: {baseline_diff:.3f}")   # 预期 < 0.05
    print(f"ADQM confidence gap: {adqm_diff:.3f}")           # 预期 > 0.20
    print(f"Axis gap (angular samples): {axis_gap:.3f}")     # 预期 > 0.50
    print(f"  (基线无法区分两类误差: {'PASS' if baseline_diff < 0.05 else 'FAIL'}; "
          f"ADQM 可区分: {'PASS' if adqm_diff > 0.20 else 'FAIL'}; "
          f"短轴崩溃: {'PASS' if axis_gap > 0.50 else 'FAIL'})")
    return dict(baseline_diff=baseline_diff, adqm_diff=adqm_diff, axis_gap=axis_gap)


# --------------------------------- 任务 6 -----------------------------------
def plot_disentanglement(data_by_cat, output_dir):
    """绘制误差模式解耦对比图（对应论文图X）。

    左子图: 平移误差主导样本 —— 两轴质量同步温和下降（p_parallel ≈ p_perp）；
    右子图: 角度误差主导样本 —— 短轴崩溃（p_perp 骤降, p_parallel 保持高位）。

    每个子图内容：
      - 基线 / ADQM 框级置信度分布直方图（半透明叠加）；
      - 均值竖直虚线；
      - 内嵌柱状图: Baseline conf / ADQM conf / p_parallel / p_perp 均值对比。

    Args:
        data_by_cat: {'translation': {'baseline': [...], 'adqm': [...]},
                      'angular':     {'baseline': [...], 'adqm': [...]}}
        output_dir: 图片输出目录。
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 8,
        'pdf.fonttype': 42,          # TrueType 嵌入, 期刊投稿要求
        'savefig.dpi': 300,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    titles = {'translation': 'Translation-Error-Dominant',
              'angular': 'Angular-Error-Dominant'}
    annotations = {'translation': 'Both axes drop\n(p$_{\\parallel}$ ≈ p$_{\\perp}$)',
                   'angular': 'Minor-axis collapse\n(p$_{\\perp}$ drops sharply)'}

    for ax, cat in zip(axes, ['translation', 'angular']):
        data = data_by_cat[cat]
        conf_b = [s['pred_conf'] for s in data['baseline']]
        conf_a = [s['pred_conf'] for s in data['adqm']]
        p_par = [s['p_parallel'] for s in data['adqm']]
        perp = [s['p_perp'] for s in data['adqm']]

        bins = np.linspace(0.0, 1.0, 21)
        ax.hist(conf_b, bins=bins, color='#b8b8b8', alpha=0.65,
                label='Baseline conf', edgecolor='none')
        ax.hist(conf_a, bins=bins, color='#1f77b4', alpha=0.55,
                label='ADQM conf', edgecolor='none')

        # 均值竖直虚线
        if conf_b:
            ax.axvline(np.mean(conf_b), color='#707070', ls='--', lw=0.9)
        if conf_a:
            ax.axvline(np.mean(conf_a), color='#1f47b4', ls='--', lw=0.9)

        # 内嵌柱状图: 四项均值对比
        inset = ax.inset_axes([0.58, 0.52, 0.39, 0.42])
        means = [np.mean(conf_b), np.mean(conf_a), np.mean(p_par), np.mean(perp)]
        bars = inset.bar(range(4), means, width=0.62,
                         color=['#b8b8b8', '#1f77b4', '#7fb3d5', '#c44e52'])
        inset.set_xticks(range(4))
        inset.set_xticklabels(['B', 'A', 'p$_{\\parallel}$', 'p$_{\\perp}$'],
                               fontsize=6)
        inset.tick_params(axis='y', labelsize=6)
        inset.set_ylim(0, 1.0)
        for b, v in zip(bars, means):
            inset.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.2f}',
                       ha='center', fontsize=5.5)
        inset.set_title('Mean', fontsize=6.5, pad=2)

        ax.set_title(titles[cat], fontsize=8.5)
        ax.set_xlabel('Box-level confidence', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.set_xlim(0, 1)
        # 误差模式标注
        ax.text(0.03, 0.72, annotations[cat], fontsize=7, color='#333333',
                transform=ax.transAxes,
                bbox=dict(fc='white', ec='#999999', lw=0.4, alpha=0.85))
        if cat == 'translation':
            ax.legend(fontsize=6.5, loc='upper right', framealpha=0.9)

    fig.tight_layout(pad=0.4)
    out = Path(output_dir) / 'error_mode_disentanglement.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[plot] 已保存: {out}")


# ---------------------------------- 主流程 -----------------------------------
def main():
    parser = argparse.ArgumentParser('Error Mode Disentanglement (论文 4.4 节)')
    parser.add_argument('--baseline-json', required=True, help='基线测试结果 JSON')
    parser.add_argument('--adqm-json', required=True, help='ADQM 测试结果 JSON')
    parser.add_argument('--output-dir', default='outputs/calibration')
    parser.add_argument('--num-samples', type=int, default=NUM_SAMPLES,
                        help='每类抽样数, 默认 200')
    parser.add_argument('--seed', type=int, default=SEED, help='随机抽样种子')
    args = parser.parse_args()

    # 任务 1: 加载（IoU ∈ [0.5, 0.7]）
    baseline_pool = load_results(args.baseline_json)
    adqm_pool = load_results(args.adqm_json)

    # 任务 3: 两方法分别筛选两类困难样本（同一 rng 保证可复现）
    rng = random.Random(args.seed)
    samples = {}
    for cat in ['translation', 'angular']:
        samples[cat] = {
            'baseline': filter_samples(baseline_pool, cat, rng, args.num_samples),
            'adqm': filter_samples(adqm_pool, cat, rng, args.num_samples),
        }

    # 任务 4: 统计（论文表X）
    stats_by_cat = {cat: compute_statistics(s['baseline'], s['adqm'])
                    for cat, s in samples.items()}
    print("\n===== 任务4: 置信度统计（对应论文表X） =====")
    header = f"{'样本类别':<12}{'方法':<10}{'框级置信度':<12}{'p_parallel':<12}{'p_perp':<12}{'N':>5}"
    print(header)
    print('-' * len(header))
    for cat, cat_cn in [('translation', '平移误差主导'), ('angular', '角度误差主导')]:
        st = stats_by_cat[cat]
        print(f"{cat_cn:<12}{'Baseline':<10}{st['baseline']['pred_conf']:<12.3f}{'-':<12}{'-':<12}{st['n_baseline']:>5}")
        print(f"{cat_cn:<12}{'ADQM':<10}{st['adqm']['pred_conf']:<12.3f}"
              f"{st['adqm']['p_parallel']:<12.3f}{st['adqm']['p_perp']:<12.3f}{st['n_adqm']:>5}")

    # 趋势自检（论文预期趋势）
    print("\n===== 趋势自检 =====")
    tr, ag = stats_by_cat['translation'], stats_by_cat['angular']
    gap_tr = abs(tr['adqm']['p_parallel'] - tr['adqm']['p_perp'])
    gap_ag = abs(ag['adqm']['p_parallel'] - ag['adqm']['p_perp'])
    print(f"平移样本两轴差 |p_par - p_perp| = {gap_tr:.3f} (预期: 接近, 同步下降)")
    print(f"角度样本两轴差 |p_par - p_perp| = {gap_ag:.3f} (预期: 显著, 短轴崩溃)")
    conf_ratio = (ag['adqm']['pred_conf'] / max(tr['adqm']['pred_conf'], 1e-6)
                  if tr['adqm']['pred_conf'] > 0 else float('nan'))
    print(f"角度/平移样本 ADQM 置信度比 = {conf_ratio:.3f} (预期: 显著更低)")

    # 任务 5: 显著性验证
    significance_analysis(stats_by_cat)

    # 任务 6: 可视化
    plot_disentanglement(samples, args.output_dir)


if __name__ == '__main__':
    main()

import math
import os
import sys

import torch

# 支持从任意目录直接运行：把项目根目录加入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adqm.aar import compute_aar, compute_direction_labels


def test_perfect_overlap():
    """测试 1: 预测框与 GT 框完全重合 -> AAR 双轴均为 1。

    重合时投影区间完全一致（l = w / h，d_bar = 0），任意角度下均应有 AAR = 1。
    """
    for theta in [0.0, 0.3, 0.7, 1.2, math.pi - 0.01]:
        box = torch.tensor([[10.0, 12.0, 5.0, 3.0, theta]])
        aar_p, aar_v = compute_aar(box, box.clone())
        assert torch.allclose(aar_p, torch.ones_like(aar_p), atol=1e-6), \
            f"theta={theta}: AAR_parallel={aar_p.item()} != 1"
        assert torch.allclose(aar_v, torch.ones_like(aar_v), atol=1e-6), \
            f"theta={theta}: AAR_perp={aar_v.item()} != 1"


def test_angle_offset():
    """测试 2: 中心重合、角度偏 10 度、宽高比 5:1 -> AAR_perp 显著低于 AAR_parallel。

    解析期望（按公式 6-9 精确手算）:
        l_parallel = 5, d_bar_parallel ≈ 0.086824 -> AAR_parallel ≈ 0.98296
        l_perp     = 1, d_bar_perp     ≈ 0.434121 -> AAR_perp     ≈ 0.69730
    细长框的角度偏差主要体现为短轴方向的顶点发散，因此 AAR_perp 被显著压低。
    """
    gt = torch.tensor([[10.0, 10.0, 5.0, 1.0, 0.0]])
    pred = torch.tensor([[10.0, 10.0, 5.0, 1.0, math.radians(10.0)]])
    aar_p, aar_v = compute_aar(pred, gt)
    assert abs(aar_p.item() - 0.98296) < 1e-3, \
        f"AAR_parallel={aar_p.item()}，期望 ≈ 0.98296"
    assert abs(aar_v.item() - 0.69730) < 1e-3, \
        f"AAR_perp={aar_v.item()}，期望 ≈ 0.69730"
    assert aar_v.item() < aar_p.item() - 0.2, \
        f"AAR_perp={aar_v.item()} 未显著低于 AAR_parallel={aar_p.item()}"


def test_no_intersection():
    """测试 3: 预测框与 GT 框完全不相交 -> 两轴 AAR 均为 0。

    投影交集长度被 clamp 为 0，而顶点偏差大于 0，故 AAR = 0 / d_bar = 0。
    """
    gt = torch.tensor([[5.0, 5.0, 2.0, 2.0, 0.4]])
    pred = torch.tensor([[80.0, 90.0, 2.0, 2.0, 0.4]])
    aar_p, aar_v = compute_aar(pred, gt)
    assert aar_p.item() == 0.0, f"AAR_parallel={aar_p.item()} != 0"
    assert aar_v.item() == 0.0, f"AAR_perp={aar_v.item()} != 0"


def test_translation_only():
    """测试 4: 仅中心平移（角度正确）-> 两轴 AAR 同步下降。

    使用方形框 (w=h=2) 沿对角线平移 (0.5, 0.5)，两轴几何完全对称：
        l_parallel = l_perp = 1.5, d_bar = 0.5 -> 双轴 AAR 均为 0.75（重合时为 1.0）
    """
    gt = torch.tensor([[5.0, 5.0, 2.0, 2.0, 0.0]])
    pred = torch.tensor([[5.5, 5.5, 2.0, 2.0, 0.0]])
    aar_p, aar_v = compute_aar(pred, gt)
    assert torch.allclose(aar_p, aar_v, atol=1e-6), \
        f"两轴应同步下降（相等）: AAR_parallel={aar_p.item()}, AAR_perp={aar_v.item()}"
    assert abs(aar_p.item() - 0.75) < 1e-6, f"AAR_parallel={aar_p.item()} != 0.75"
    assert aar_p.item() < 1.0, "相对完全重合 (AAR=1) 应下降"


def test_batch_and_robustness():
    """附加: 批量一致性 / N=0 空输入 / requires_grad 输入 / 值域 [0, 1]。"""
    gts = torch.tensor([
        [10.0, 10.0, 5.0, 3.0, 0.7],              # 重合
        [10.0, 10.0, 5.0, 1.0, 0.0],              # 角度偏移
        [5.0, 5.0, 2.0, 2.0, 0.4],               # 不相交
        [5.0, 5.0, 2.0, 2.0, 0.0],               # 平移
    ])
    preds = torch.tensor([
        [10.0, 10.0, 5.0, 3.0, 0.7],
        [10.0, 10.0, 5.0, 1.0, math.radians(10.0)],
        [80.0, 90.0, 2.0, 2.0, 0.4],
        [5.5, 5.5, 2.0, 2.0, 0.0],
    ])

    # 批量结果与逐条计算一致
    bp, bv = compute_aar(preds, gts)
    assert bp.shape == (4,) and bv.shape == (4,)
    for i in range(4):
        p_i, v_i = compute_aar(preds[i:i + 1], gts[i:i + 1])
        assert abs(bp[i].item() - p_i.item()) < 1e-6, f"样本 {i} 批量/单条结果不一致"
        assert abs(bv[i].item() - v_i.item()) < 1e-6, f"样本 {i} 批量/单条结果不一致"

    # N=0 空输入优雅返回
    empty = torch.zeros(0, 5)
    ep, ev = compute_aar(empty, empty)
    assert ep.shape == (0,) and ev.shape == (0,)

    # requires_grad=True 的预测框不报错（@torch.no_grad() 生效，输出不带梯度）
    grad_pred = preds.clone().requires_grad_(True)
    gp, gv = compute_aar(grad_pred, gts)
    assert not gp.requires_grad and not gv.requires_grad

    # 随机批量样本的值域检查: AAR ∈ [0, 1]
    torch.manual_seed(0)
    n = 1000
    scale = torch.tensor([20.0, 20.0, 10.0, 10.0, math.pi])
    rand_pred, rand_gt = torch.rand(n, 5) * scale, torch.rand(n, 5) * scale
    rp, rv = compute_aar(rand_pred, rand_gt)
    assert ((rp >= 0.0) & (rp <= 1.0)).all(), "AAR_parallel 超出 [0, 1]"
    assert ((rv >= 0.0) & (rv <= 1.0)).all(), "AAR_perp 超出 [0, 1]"


def test_direction_positive_parallel():
    """方向测试 1: GT 中心在预测框右侧（沿长轴正方向）-> s_parallel = 1。"""
    pred = torch.tensor([[5.0, 5.0, 4.0, 2.0, 0.0]])
    gt = torch.tensor([[8.0, 5.0, 4.0, 2.0, 0.0]])  # delta_c = (3, 0)，沿 +x（长轴正方向）
    s_p, s_v = compute_direction_labels(pred, gt)
    assert s_p.item() == 1.0, f"s_parallel={s_p.item()} != 1"
    assert s_v.item() == 0.0, f"纯长轴偏移，s_perp 应为 0，实际 {s_v.item()}"


def test_direction_negative_parallel():
    """方向测试 2: GT 中心在预测框左侧（沿长轴负方向）-> s_parallel = 0。"""
    pred = torch.tensor([[5.0, 5.0, 4.0, 2.0, 0.0]])
    gt = torch.tensor([[2.0, 5.0, 4.0, 2.0, 0.0]])  # delta_c = (-3, 0)，沿 -x（长轴负方向）
    s_p, s_v = compute_direction_labels(pred, gt)
    assert s_p.item() == 0.0, f"s_parallel={s_p.item()} != 0"
    assert s_v.item() == 0.0, f"纯长轴偏移，s_perp 应为 0，实际 {s_v.item()}"


def test_direction_coincident():
    """方向测试 3: 中心完全重合（投影为 0）-> 两个标签均为 0。"""
    for theta in [0.0, 0.5, 1.1, math.pi - 0.01]:
        box = torch.tensor([[5.0, 5.0, 4.0, 2.0, theta]])
        s_p, s_v = compute_direction_labels(box, box.clone())
        assert s_p.item() == 0.0, f"theta={theta}: s_parallel={s_p.item()} != 0"
        assert s_v.item() == 0.0, f"theta={theta}: s_perp={s_v.item()} != 0"


def test_direction_batch_and_robustness():
    """方向附加: 任意角度斜向偏移 / 批量一致 / N=0 / requires_grad / 值域 {0, 1}。"""
    # 任意角度（theta = pi/4）下的斜向偏移，投影远离 0，符号确定：
    #   delta_c = (2, 0.5):
    #     proj_parallel = 2.5*cos(pi/4)  ≈ +1.77 > 0 -> s_parallel = 1
    #     proj_perp     = -1.5*cos(pi/4) ≈ -1.06 < 0 -> s_perp = 0
    #   delta_c = (-2, 0.5):
    #     proj_parallel = -1.5*cos(pi/4) < 0 -> s_parallel = 0
    #     proj_perp     = 2.5*cos(pi/4)  > 0 -> s_perp = 1
    theta = math.pi / 4
    pred = torch.tensor([[5.0, 5.0, 3.0, 1.0, theta]])
    gt_a = torch.tensor([[7.0, 5.5, 3.0, 1.0, theta]])   # delta_c = (2, 0.5)
    gt_b = torch.tensor([[3.0, 5.5, 3.0, 1.0, theta]])   # delta_c = (-2, 0.5)
    s_a_p, s_a_v = compute_direction_labels(pred, gt_a)
    assert s_a_p.item() == 1.0 and s_a_v.item() == 0.0
    s_b_p, s_b_v = compute_direction_labels(pred, gt_b)
    assert s_b_p.item() == 0.0 and s_b_v.item() == 1.0

    # 批量结果与逐条计算一致（含重合 / 正方向 / 负方向 / 重合四种用例）
    preds = torch.cat([pred, pred, pred, pred])
    gts = torch.cat([pred, gt_a, gt_b, pred])
    bp, bv = compute_direction_labels(preds, gts)
    assert bp.shape == (4,) and bv.shape == (4,)
    for i in range(4):
        p_i, v_i = compute_direction_labels(preds[i:i + 1], gts[i:i + 1])
        assert bp[i].item() == p_i.item(), f"样本 {i} 批量/单条 s_parallel 不一致"
        assert bv[i].item() == v_i.item(), f"样本 {i} 批量/单条 s_perp 不一致"

    # N=0 空输入优雅返回
    empty = torch.zeros(0, 5)
    ep, ev = compute_direction_labels(empty, empty)
    assert ep.shape == (0,) and ev.shape == (0,)

    # requires_grad=True 的预测框不报错，输出无梯度
    grad_pred = preds.clone().requires_grad_(True)
    gp, gv = compute_direction_labels(grad_pred, gts)
    assert not gp.requires_grad and not gv.requires_grad

    # 随机批量样本的值域检查: 标签只能取 {0, 1}
    torch.manual_seed(0)
    n = 1000
    scale = torch.tensor([20.0, 20.0, 10.0, 10.0, math.pi])
    rand_pred, rand_gt = torch.rand(n, 5) * scale, torch.rand(n, 5) * scale
    sp, sv = compute_direction_labels(rand_pred, rand_gt)
    assert ((sp == 0.0) | (sp == 1.0)).all(), "s_parallel 存在非 {0, 1} 取值"
    assert ((sv == 0.0) | (sv == 1.0)).all(), "s_perp 存在非 {0, 1} 取值"


if __name__ == "__main__":
    tests = [
        test_perfect_overlap,
        test_angle_offset,
        test_no_intersection,
        test_translation_only,
        test_batch_and_robustness,
        test_direction_positive_parallel,
        test_direction_negative_parallel,
        test_direction_coincident,
        test_direction_batch_and_robustness,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")

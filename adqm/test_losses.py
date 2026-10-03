import math
import os
import sys

import torch

# 支持从任意目录直接运行：把项目根目录加入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adqm.losses import aar_loss, direction_loss, total_loss


def test_zero_loss_on_perfect_match():
    """测试 1: 预测值与 GT 完全一致 -> 损失为 0。"""
    # 硬标签 {0,1} 完全一致 -> 精确为 0
    one = torch.ones(2, 5)
    zero = torch.zeros(2, 5)
    assert aar_loss(one, one, one, one).item() == 0.0
    assert aar_loss(zero, zero, zero, zero).item() == 0.0
    assert direction_loss(one, one, one, one).item() == 0.0
    assert direction_loss(zero, zero, zero, zero).item() == 0.0
    # 混合硬标签完全一致 -> 0
    gt = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0]])
    assert aar_loss(gt, gt, gt, gt).item() == 0.0
    assert direction_loss(gt, gt, gt, gt).item() == 0.0

    # BCE 固有性质: 软标签 0<p<1 时 pred==gt 达到最小值 = 熵（非 0）
    p = torch.full((2, 3), 0.5)
    expected = 2.0 * math.log(2.0)  # 两轴各 H(0.5) = ln 2
    got = aar_loss(p, p, p, p).item()
    assert abs(got - expected) < 1e-6, f"软标签最小损失 {got} != 2*ln2 = {expected}"
    # pred 偏离 gt 会增大损失（验证上式确为最小值）
    p_off = torch.full((2, 3), 0.6)
    assert aar_loss(p_off, p_off, p, p).item() > got


def test_non_negative():
    """测试 2: 任意输入下损失值非负。"""
    torch.manual_seed(0)
    for _ in range(20):
        pred = torch.rand(2, 7)
        gt_soft = torch.rand(2, 7)
        gt_hard = (torch.rand(2, 7) > 0.5).float()
        mask = torch.rand(2, 7) > 0.3
        assert aar_loss(pred, pred, gt_soft, gt_soft, positive_mask=mask).item() >= 0.0
        assert direction_loss(pred, 1 - pred, gt_hard, 1 - gt_hard,
                              positive_mask=mask).item() >= 0.0

    # 极端输入: pred 与 gt 完全相反，单元素损失达到 clamp 上限（100/轴）
    one, zero = torch.ones(1, 4), torch.zeros(1, 4)
    worst = aar_loss(one, one, zero, zero).item()
    assert worst == 200.0, f"双轴全错元素损失应为 200（clamp 上限 100/轴），实际 {worst}"


def test_negative_samples_excluded():
    """测试 3: 负样本不参与损失计算（仅匈牙利匹配后的正样本受轴向监督）。"""
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, :3] = True  # 每批前 3 个 query 为正样本

    # 正样本 pred == gt（硬标签 -> 元素损失 0），负样本随意填充
    gt = torch.zeros(2, 6)
    pred = torch.zeros(2, 6)
    assert aar_loss(pred, pred, gt, gt, positive_mask=mask).item() == 0.0

    # 负样本位置完全错误（元素损失 200），masked 损失不变仍为 0
    pred_bad = pred.clone()
    pred_bad[:, 3:] = 1.0  # 负样本预测完全错误
    assert aar_loss(pred_bad, pred_bad, gt, gt, positive_mask=mask).item() == 0.0, \
        "负样本的错误不应影响损失"
    assert direction_loss(pred_bad, pred_bad, gt, gt, positive_mask=mask).item() == 0.0

    # 对照: 不加 mask 时同样的输入损失非零
    assert aar_loss(pred_bad, pred_bad, gt, gt).item() > 0.0

    # 精确均值验证: masked mean 只统计正样本元素
    #   par: pred=1 vs gt=[0,0,1,1] -> 元素损失 [100, 100, 0, 0]
    #   perp: 全对 -> 元素损失 0
    #   mask=[T,T,F,F] -> (100+100)/2 = 100；全样本 mean = 200/4 = 50
    pred_par = torch.ones(1, 4)
    gt_par = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    pred_perp = torch.ones(1, 4)
    gt_perp = torch.ones(1, 4)
    m = torch.tensor([[True, True, False, False]])
    assert direction_loss(pred_par, pred_perp, gt_par, gt_perp,
                          positive_mask=m).item() == 100.0
    assert direction_loss(pred_par, pred_perp, gt_par, gt_perp).item() == 50.0

    # 无正样本（mask 全 False）: 损失为 0 且不产生 NaN
    no_pos = torch.zeros(1, 4, dtype=torch.bool)
    empty = direction_loss(pred_par, pred_perp, gt_par, gt_perp, positive_mask=no_pos)
    assert empty.item() == 0.0 and not torch.isnan(empty)
    empty_aar = aar_loss(pred_par, pred_perp, gt_par, gt_perp, positive_mask=no_pos)
    assert empty_aar.item() == 0.0 and not torch.isnan(empty_aar)


def test_total_loss_and_gradients():
    """附加: 公式 16 总损失 + lambda 权重语义 + 梯度回传。"""
    det = torch.tensor(3.0)
    aar = torch.tensor(0.5)
    dir_ = torch.tensor(0.25)

    # 默认权重: 3 + 0.5 + 0.25 = 3.75
    assert abs(total_loss(det, aar, dir_).item() - 3.75) < 1e-6
    # 自定义权重: 3 + 2*0.5 + 4*0.25 = 5.0
    assert abs(total_loss(det, aar, dir_, lambda1=2.0, lambda2=4.0).item() - 5.0) < 1e-6

    # lambda 分别只作用于 aar / dir 分支（梯度验证）
    det_g = det.clone().requires_grad_(True)
    aar_g = aar.clone().requires_grad_(True)
    dir_g = dir_.clone().requires_grad_(True)
    total_loss(det_g, aar_g, dir_g, lambda1=0.5, lambda2=2.0).backward()
    assert det_g.grad.item() == 1.0
    assert aar_g.grad.item() == 0.5
    assert dir_g.grad.item() == 2.0

    # aar / direction 损失对预测可微（训练时梯度可回传到 AxisQualityHead）
    torch.manual_seed(1)
    pred = torch.rand(2, 4, requires_grad=True)
    gt = torch.rand(2, 4)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
    l = aar_loss(pred, pred, gt, gt, positive_mask=mask) \
        + direction_loss(pred, 1 - pred, gt, 1 - gt, positive_mask=mask)
    l.backward()
    assert pred.grad is not None and torch.isfinite(l).all()


if __name__ == "__main__":
    tests = [
        test_zero_loss_on_perfect_match,
        test_non_negative,
        test_negative_samples_excluded,
        test_total_loss_and_gradients,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")

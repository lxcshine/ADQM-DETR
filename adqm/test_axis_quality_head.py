import os
import sys

import torch
import torch.nn.functional as F

# 支持从任意目录直接运行：把项目根目录加入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adqm.axis_quality_head import AxisQualityHead

B, N = 2, 300  # 批大小 / query 数，与 RT-DETR 推理配置一致


def _make_head():
    """默认配置：embed_dim=256（解码器输出），num_heads=8（正交注意力）。"""
    return AxisQualityHead(embed_dim=256, num_heads=8)


def test_output_shapes():
    """测试 1: 输入随机张量 -> 输出维度正确。"""
    torch.manual_seed(0)
    head = _make_head()
    x = torch.randn(B, N, 256)
    p_cls = torch.rand(B, N)

    p_par, p_perp, s_par, s_perp, p_box = head(x, head_assignment={'p_cls': p_cls})
    assert p_par.shape == (B, N), f"p_parallel 形状 {tuple(p_par.shape)}"
    assert p_perp.shape == (B, N), f"p_perp 形状 {tuple(p_perp.shape)}"
    assert s_par.shape == (B, N), f"s_parallel 形状 {tuple(s_par.shape)}"
    assert s_perp.shape == (B, N), f"s_perp 形状 {tuple(s_perp.shape)}"
    assert p_box.shape == (B, N), f"p_box 形状 {tuple(p_box.shape)}"

    # 不提供 p_cls（训练路径）: p_box 为 None，其余输出不受影响
    p2_par, p2_perp, s2_par, s2_perp, p2_box = head(x)
    assert p2_box is None, "未提供 p_cls 时 p_box 应为 None"
    assert torch.equal(p2_par, p_par), "head_assignment 不应影响质量分数输出"

    # 分割语义: z_parallel 取前 embed_dim/2 通道（前 num_heads/2 个头）
    assert head.split == 128 and head.num_heads == 8


def test_output_range():
    """测试 2: 所有输出值在 [0, 1] 范围内 + 公式 15 聚合验证。"""
    torch.manual_seed(1)
    head = _make_head()
    # 常规、大幅值、全零等极端输入，sigmoid 输出都必须落在 [0, 1]
    for x in [torch.randn(B, N, 256),
              torch.randn(B, N, 256) * 10.0,
              torch.zeros(B, N, 256)]:
        p_cls = torch.rand(B, N)
        p_par, p_perp, s_par, s_perp, p_box = head(x, head_assignment={'p_cls': p_cls})
        for name, t in [('p_parallel', p_par), ('p_perp', p_perp),
                        ('s_parallel', s_par), ('s_perp', s_perp), ('p_box', p_box)]:
            assert (t >= 0.0).all() and (t <= 1.0).all(), f"{name} 存在超出 [0,1] 的值"

    # 公式 15 精确验证: p_box == p_cls * p_parallel * p_perp
    x = torch.randn(B, N, 256)
    p_cls = torch.rand(B, N)
    p_par, p_perp, _, _, p_box = head(x, head_assignment={'p_cls': p_cls})
    expected = p_cls * p_par * p_perp
    assert torch.allclose(p_box, expected, atol=1e-6), "p_box 聚合公式不符"


def test_param_count():
    """测试 3: 参数量不超过预期。

    精确计数（embed_dim=256, num_heads=8）:
        phi_parallel / phi_perp: Linear(128, 1) -> 128 + 1 = 129，各 129
        psi: Linear(256, 64) -> 256*64 + 64 = 16448
             Linear(64, 2)  -> 64*2 + 2   = 130
        总计 129 + 129 + 16448 + 130 = 16836
    RT-DETRv2-OBB-M 主干模型约 20M 参数，占比约 0.084% < 0.1%。
    """
    head = _make_head()
    n = sum(p.numel() for p in head.parameters())
    assert n == 16836, f"参数量 {n} != 精确期望 16836"
    assert n / 20e6 < 0.001, f"参数占比 {n / 20e6:.4%} 超过总模型的 0.1%"

    # 参数量随 embed_dim 仅线性/平方级增长（无额外注意力/卷积参数源）
    n_small = sum(p.numel() for p in AxisQualityHead(embed_dim=128, num_heads=4).parameters())
    # 128: phi 65*2 + Linear(128,32)=4128 + Linear(32,2)=66 -> 65+65+4128+66 = 4324
    assert n_small == 4324, f"小配置参数量 {n_small} != 4324"


def test_training_and_inference_semantics():
    """附加: 训练（sigmoid + BCE 损失可反传到头参数）与推理（阈值 0.5 二值化）。"""
    torch.manual_seed(2)
    head = _make_head()
    x = torch.randn(2, 10, 256, requires_grad=True)
    p_par, p_perp, s_par, s_perp, _ = head(x)

    # 训练: 方向分支用 BCE（标签 {0,1}，见 compute_direction_labels），
    # 质量分支用软标签（AAR）回归，均可反传
    loss = p_par.mean() + p_perp.mean() \
        + F.binary_cross_entropy(s_par, torch.zeros_like(s_par)) \
        + F.binary_cross_entropy(s_perp, torch.ones_like(s_perp))
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"参数 {name} 未收到梯度"
    assert x.grad is not None, "输入嵌入未收到梯度"

    # 推理: s > 0.5 -> 1（正方向），s <= 0.5 -> 0
    with torch.no_grad():
        head.eval()
        s_par2, s_perp2 = head(torch.randn(4, 50, 256))[2:4]
    s_bin = (torch.cat([s_par2, s_perp2]) > 0.5).float()
    assert ((s_bin == 0.0) | (s_bin == 1.0)).all(), "二值化结果非 {0, 1}"


if __name__ == "__main__":
    tests = [
        test_output_shapes,
        test_output_range,
        test_param_count,
        test_training_and_inference_semantics,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")

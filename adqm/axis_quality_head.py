import torch
import torch.nn as nn

__all__ = ["AxisQualityHead"]


class AxisQualityHead(nn.Module):
    """
    基于正交注意力的轴向质量预测头。

    输入：解码器输出嵌入和注意力头信息
    输出：轴级质量分数、方向预测、聚合后的框级置信度
    """

    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: 解码器输出嵌入的维度（如 256）
            num_heads: 正交注意力的总头数（如 8）。
                嵌入按头归属对半分割：前 num_heads/2 个头的输出通道
                属长轴组，后 num_heads/2 个头属短轴组。
        """
        super().__init__()
        assert embed_dim % 2 == 0, \
            f"embed_dim={embed_dim} 必须为偶数，才能按头对半分割"
        assert num_heads % 2 == 0, \
            f"num_heads={num_heads} 必须为偶数（正交注意力头两组对称）"
        assert embed_dim % num_heads == 0, \
            f"embed_dim={embed_dim} 不能被 num_heads={num_heads} 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # 分割点 = 前 num_heads // 2 个头的输出通道边界
        self.split = embed_dim // 2

        # (a) 两个单线性层映射器（论文公式 13）：长轴 / 短轴子向量 -> 标量质量分数
        self.phi_parallel = nn.Linear(embed_dim // 2, 1)
        self.phi_perp = nn.Linear(embed_dim // 2, 1)

        # (b) 两层感知机用于方向预测（论文公式 14）：输出 2 个方向 logits
        self.psi = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 2),
        )

    def forward(self, decoder_output, head_assignment=None):
        """
        输入：
          decoder_output: Tensor, shape (B, N_queries, embed_dim)
          head_assignment: dict 或 None。
            - 嵌入分割直接从 decoder_output 按头归属对半切（论文公式 12）；
            - dict 可携带 'p_cls'（分类头输出的分类置信度，(B, N_queries)），
              用于推理时的框级置信度聚合（论文公式 15）；未提供时 p_box 为 None。

        输出：
          p_parallel: Tensor, (B, N_queries), 长轴质量分数, [0, 1]
          p_perp: Tensor, (B, N_queries), 短轴质量分数, [0, 1]
          s_parallel: Tensor, (B, N_queries), 长轴方向预测, [0, 1]
          s_perp: Tensor, (B, N_queries), 短轴方向预测, [0, 1]
                     （推理时通过阈值 0.5 二值化为方向标签）
          p_box: Tensor, (B, N_queries), 聚合后的框级置信度
                    = p_cls * p_parallel * p_perp；未提供 p_cls 时为 None
        """
        z_parallel = decoder_output[:, :, :self.split]
        z_perp = decoder_output[:, :, self.split:]

        # ---------- 公式 13：轴级质量分数 ----------
        p_parallel = self.phi_parallel(z_parallel).squeeze(-1).sigmoid()  # (B, N)
        p_perp = self.phi_perp(z_perp).squeeze(-1).sigmoid()              # (B, N)

        # ---------- 公式 14：双轴方向预测 ----------
        s_parallel, s_perp = self.psi(decoder_output).unbind(-1)          # 各 (B, N)
        s_parallel = s_parallel.sigmoid()
        s_perp = s_perp.sigmoid()

        # ---------- 公式 15：框级置信度聚合（仅推理使用，训练不影响分类损失） ----------
        p_box = None
        if head_assignment is not None and 'p_cls' in head_assignment:
            p_cls = head_assignment['p_cls']
            p_box = p_cls * p_parallel * p_perp

        return p_parallel, p_perp, s_parallel, s_perp, p_box

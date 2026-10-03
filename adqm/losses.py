import torch
import torch.nn.functional as F

__all__ = ["aar_loss", "direction_loss", "total_loss"]


def _masked_mean(element_loss, positive_mask=None):
    """对逐元素损失应用正样本 mask 后求均值。

    Args:
        element_loss: (B, N) 逐元素损失张量
        positive_mask: (B, N) 布尔张量，标记匈牙利匹配后的正样本；
                       None 表示全部元素参与

    Returns:
        标量损失。无正样本时返回 0（保持计算图连接，不产生 NaN）。
    """
    if positive_mask is None:
        return element_loss.mean()
    masked = element_loss[positive_mask]
    if masked.numel() == 0:
        # 无正样本: 轴向监督为 0；sum() * 0.0 保持梯度路径安全
        return element_loss.sum() * 0.0
    return masked.mean()


def aar_loss(pred_p_parallel, pred_p_perp, gt_aar_parallel, gt_aar_perp, positive_mask=None):
    """AAR 损失（论文公式 18）。

    输入：
      pred_p_parallel: (B, N), 模型预测的长轴质量分数, [0, 1]
      pred_p_perp: (B, N), 模型预测的短轴质量分数, [0, 1]
      gt_aar_parallel: (B, N), GT 的长轴 AAR 值, [0, 1]
      gt_aar_perp: (B, N), GT 的短轴 AAR 值, [0, 1]
      positive_mask: (B, N) 布尔张量或 None，仅正样本参与损失

    输出：
      loss: 标量
    """
    element = F.binary_cross_entropy(pred_p_parallel, gt_aar_parallel, reduction='none') \
        + F.binary_cross_entropy(pred_p_perp, gt_aar_perp, reduction='none')  # (B, N)
    return _masked_mean(element, positive_mask)


def direction_loss(pred_s_parallel, pred_s_perp, gt_s_parallel, gt_s_perp, positive_mask=None):
    """方向损失（论文公式 19）。

    输入：
      pred_s_parallel: (B, N), 模型预测的长轴方向, [0, 1]
      pred_s_perp: (B, N), 模型预测的短轴方向, [0, 1]
      gt_s_parallel: (B, N), GT 的长轴方向标签, {0, 1}
      gt_s_perp: (B, N), GT 的短轴方向标签, {0, 1}
      positive_mask: (B, N) 布尔张量或 None，仅正样本参与损失

    输出：
      loss: 标量
    """
    element = F.binary_cross_entropy(pred_s_parallel, gt_s_parallel, reduction='none') \
        + F.binary_cross_entropy(pred_s_perp, gt_s_perp, reduction='none')  # (B, N)
    return _masked_mean(element, positive_mask)


def total_loss(detection_loss, aar_loss, dir_loss, lambda1=1.0, lambda2=1.0):
    """总损失（论文公式 16）。

    输入：
      detection_loss: 标量，RiO-DETR 现有检测损失（focal + 角度 + KLD），保持不变
      aar_loss: 标量，本模块公式 18 的 AAR 损失
      dir_loss: 标量，本模块公式 19 的方向损失
      lambda1: AAR 损失权重（超参数，可配置）
      lambda2: 方向损失权重（超参数，可配置）

    输出：
      total = detection_loss + lambda1 * aar_loss + lambda2 * dir_loss
    """
    return detection_loss + lambda1 * aar_loss + lambda2 * dir_loss

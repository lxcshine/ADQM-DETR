import torch

__all__ = ["compute_aar", "compute_direction_labels"]


def _obb_corners(boxes):
    
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    cos_t = torch.cos(theta).unsqueeze(-1)  # (N, 1)
    sin_t = torch.sin(theta).unsqueeze(-1)  # (N, 1)
    # 四个局部角点偏移 d_i = (±w/2, ±h/2)
    dx = torch.stack([-w, w, w, -w], dim=-1) * 0.5  # (N, 4)
    dy = torch.stack([-h, -h, h, h], dim=-1) * 0.5  # (N, 4)
    # 旋转矩阵 R(theta) = [[cos, -sin], [sin, cos]]
    rx = dx * cos_t - dy * sin_t  # (N, 4)
    ry = dx * sin_t + dy * cos_t  # (N, 4)
    corners = torch.stack([cx.unsqueeze(-1) + rx, cy.unsqueeze(-1) + ry], dim=-1)  # (N, 4, 2)
    return corners


def _safe_ratio(l, d_bar):
    denom = l + d_bar
    # clamp 避免 0/0 产生 NaN（denom>0 的分支不受影响；denom=0 时由 where 置 0）
    safe = denom.clamp(min=torch.finfo(denom.dtype).tiny)
    return torch.where(denom > 0, l / safe, torch.zeros_like(denom))


@torch.no_grad()
def compute_aar(pred_boxes, gt_boxes):
    
    assert pred_boxes.dim() == 2 and pred_boxes.size(-1) == 5, \
        f"pred_boxes 应为 (N, 5)，实际为 {tuple(pred_boxes.shape)}"
    assert gt_boxes.shape == pred_boxes.shape, \
        f"pred_boxes {tuple(pred_boxes.shape)} 与 gt_boxes {tuple(gt_boxes.shape)} 形状不一致"

    # 空输入直接返回空张量
    if pred_boxes.shape[0] == 0:
        empty = pred_boxes.new_zeros((0,))
        return empty, empty.clone()

    # ---------- 步骤 1：预测框与 GT 框的四个顶点 ----------
    pred_corners = _obb_corners(pred_boxes)  # (N, 4, 2)
    gt_corners = _obb_corners(gt_boxes)       # (N, 4, 2)

    # ---------- 步骤 2：GT 框的局部正交基 ----------
    theta_gt = gt_boxes[:, 4]
    e_parallel = torch.stack([torch.cos(theta_gt), torch.sin(theta_gt)], dim=-1)  # (N, 2) 长轴
    e_perp = torch.stack([-torch.sin(theta_gt), torch.cos(theta_gt)], dim=-1)     # (N, 2) 短轴

    # ---------- 步骤 3：顶点投影到 GT 局部坐标系（论文公式 6） ----------
    gt_center = gt_boxes[:, :2]                       # (N, 2)
    pred_rel = pred_corners - gt_center.unsqueeze(1)  # (N, 4, 2)，p_i_pred - c_gt
    gt_rel = gt_corners - gt_center.unsqueeze(1)      # (N, 4, 2)

    # u_i = (p_i - c_gt)^T · e_parallel;  v_i = (p_i - c_gt)^T · e_perp
    u_pred = (pred_rel * e_parallel.unsqueeze(1)).sum(dim=-1)  # (N, 4)
    v_pred = (pred_rel * e_perp.unsqueeze(1)).sum(dim=-1)      # (N, 4)
    u_gt = (gt_rel * e_parallel.unsqueeze(1)).sum(dim=-1)     # (N, 4)
    v_gt = (gt_rel * e_perp.unsqueeze(1)).sum(dim=-1)          # (N, 4)

    # ---------- 步骤 4：轴向投影交集长度（论文公式 7） ----------
    # GT 投影到自身局部坐标系，范围恰为 [-w/2, w/2] 与 [-h/2, h/2]
    half_w = gt_boxes[:, 2] * 0.5  # (N,)
    half_h = gt_boxes[:, 3] * 0.5  # (N,)
    l_parallel = (torch.minimum(u_pred.max(dim=-1).values, half_w)
                  - torch.maximum(u_pred.min(dim=-1).values, -half_w)).clamp(min=0.0)  # (N,)
    l_perp = (torch.minimum(v_pred.max(dim=-1).values, half_h)
              - torch.maximum(v_pred.min(dim=-1).values, -half_h)).clamp(min=0.0)      # (N,)

    # ---------- 步骤 5：平均顶点偏差（论文公式 8） ----------
    d_bar_parallel = (u_pred - u_gt).abs().mean(dim=-1)  # (N,)
    d_bar_perp = (v_pred - v_gt).abs().mean(dim=-1)      # (N,)

    # ---------- 步骤 6：AAR = l / (l + d_bar)，分母为 0 时置 0（论文公式 9） ----------
    aar_parallel = _safe_ratio(l_parallel, d_bar_parallel)
    aar_perp = _safe_ratio(l_perp, d_bar_perp)

    return aar_parallel, aar_perp


@torch.no_grad()
def compute_direction_labels(pred_boxes, gt_boxes):
    
    assert pred_boxes.dim() == 2 and pred_boxes.size(-1) == 5, \
        f"pred_boxes 应为 (N, 5)，实际为 {tuple(pred_boxes.shape)}"
    assert gt_boxes.shape == pred_boxes.shape, \
        f"pred_boxes {tuple(pred_boxes.shape)} 与 gt_boxes {tuple(gt_boxes.shape)} 形状不一致"

    # 空输入直接返回空张量
    if pred_boxes.shape[0] == 0:
        empty = pred_boxes.new_zeros((0,))
        return empty, empty.clone()

    # ---------- 步骤 1：中心偏差向量 delta_c = c_gt - c_pred ----------
    delta_c = gt_boxes[:, :2] - pred_boxes[:, :2]  # (N, 2)

    # ---------- 步骤 2：GT 框的局部正交基 ----------
    theta_gt = gt_boxes[:, 4]
    e_parallel = torch.stack([torch.cos(theta_gt), torch.sin(theta_gt)], dim=-1)  # (N, 2) 长轴
    e_perp = torch.stack([-torch.sin(theta_gt), torch.cos(theta_gt)], dim=-1)     # (N, 2) 短轴

    # ---------- 步骤 3：投影并取方向 ----------
    proj_parallel = (delta_c * e_parallel).sum(dim=-1)  # (N,)
    proj_perp = (delta_c * e_perp).sum(dim=-1)          # (N,)

    # ---------- 步骤 4：二值方向标签（严格大于 0 才为 1；投影为 0 时标签为 0） ----------
    # 转浮点并保持输入 dtype（float32 输入下与 .float() 等价）
    s_parallel = (proj_parallel > 0).to(pred_boxes.dtype)
    s_perp = (proj_perp > 0).to(pred_boxes.dtype)

    return s_parallel, s_perp

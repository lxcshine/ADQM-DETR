"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register


__all__ = ['PostProcessor', 'PostProcessorOBB']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self


@register()
class PostProcessorOBB(nn.Module):
    __share__ = ['num_classes', 'num_top_queries']

    def __init__(self, num_classes=80, num_top_queries=300, input_shape=None) -> None:
        super().__init__()
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.deploy_mode = False
        self.input_shape = input_shape

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        # ADQM 轴级置信度分支（论文公式 15）：
        # 推理时 p_box = p_cls * p_parallel * p_perp，替代 p_cls 用于
        # topk 排序与阈值过滤；AAR / 方向标签仅在训练时由 criterion 计算，
        # 此处只消费 AxisQualityHead 的前向输出，全部为 TRT 兼容算子。
        adqm = outputs['adqm'] if isinstance(outputs, dict) and 'adqm' in outputs else None

        if self.input_shape is not None:
            target_h, target_w = self.input_shape
            h = orig_target_sizes[:, 0]
            w = orig_target_sizes[:, 1]
            scale_w = target_w / w
            scale_h = target_h / h
            scale = torch.min(scale_w, scale_h).unsqueeze(1).unsqueeze(1)
            target_wh = torch.tensor([target_w, target_h, target_w, target_h], device=boxes.device).reshape(1, 1, 4)

            boxes_xywh = boxes[..., :4] * target_wh / scale
            boxes_angle = boxes[..., 4:] * 3.141592653589793
            bbox_pred = torch.cat([boxes_xywh, boxes_angle], dim=-1)

        else:
            h = orig_target_sizes[:, 0]
            w = orig_target_sizes[:, 1]
            scale_fct = torch.stack([w, h, w, h], dim=1).unsqueeze(1)
            boxes_xywh = boxes[..., :4] * scale_fct
            boxes_angle = boxes[..., 4:] * 3.141592653589793
            bbox_pred = torch.cat([boxes_xywh, boxes_angle], dim=-1)

        scores = F.sigmoid(logits)

        if adqm is not None:
            # p_parallel / p_perp: (B, N_queries) -> 按类别维度扩展到 (B, N, K)
            # 后与展平的 p_cls 相乘，topk 排序即基于 p_box
            p_axis = (adqm['p_parallel'] * adqm['p_perp']).unsqueeze(-1)
            p_axis = p_axis.expand(-1, -1, scores.shape[-1])
            p_box = scores.flatten(1) * p_axis.flatten(1)
            scores, index = torch.topk(p_box, self.num_top_queries, dim=-1)
        else:
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)

        labels = mod(index, self.num_classes)
        index = index // self.num_classes
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, 5))

        # 轴级信息按 query 索引 gather: (B, topk, 4)
        # 通道顺序 [p_parallel, p_perp, s_parallel, s_perp]
        axis_scores = None
        if adqm is not None:
            axis_src = torch.stack([
                adqm['p_parallel'], adqm['p_perp'],
                adqm['s_parallel'], adqm['s_perp'],
            ], dim=-1)
            axis_scores = axis_src.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, 4))

        if self.deploy_mode:
            if axis_scores is not None:
                return labels, boxes, scores, axis_scores
            return labels, boxes, scores

        results = []
        for i, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            if axis_scores is not None:
                result['p_parallel'] = axis_scores[i, :, 0]
                result['p_perp'] = axis_scores[i, :, 1]
                result['s_parallel'] = axis_scores[i, :, 2]
                result['s_perp'] = axis_scores[i, :, 3]
            results.append(result)

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self

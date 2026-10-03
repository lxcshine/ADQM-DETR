"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from calflops import calculate_flops
from typing import Tuple

def _fmt_params(n: int) -> str:
    """参数量统一以 M（百万）为单位，保留 4 位小数以覆盖新增小模块（如 ADQM head）。"""
    return "%.4fM" % (n / 1e6)

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, str]:

    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()

    flops, macs, _ = calculate_flops(model=model_for_info,
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4,
                                        print_detailed=False)

    # 参数总量统计: 总参数 / 可训练参数 / 分模块细分（单位 M）
    params = sum(p.numel() for p in model_for_info.parameters())
    trainable_params = sum(p.numel() for p in model_for_info.parameters() if p.requires_grad)

    lines = [
        "Model FLOPs:%s   MACs:%s   Params:%s" % (flops, macs, _fmt_params(params)),
        "Total params: %s   Trainable params: %s   Non-trainable params: %s"
        % (_fmt_params(params), _fmt_params(trainable_params), _fmt_params(params - trainable_params)),
    ]

    # 模块级参数分解: backbone / encoder / decoder
    for name in ('backbone', 'encoder', 'decoder'):
        module = getattr(model_for_info, name, None)
        if module is not None:
            n = sum(p.numel() for p in module.parameters())
            lines.append("  %-8s params: %10s (%6.2f%% of total)"
                         % (name, _fmt_params(n), 100.0 * n / max(params, 1)))

    # ADQM 轴向质量头（论文新增模块，见 adqm/axis_quality_head.py）
    decoder = getattr(model_for_info, 'decoder', None)
    aqh = getattr(decoder, 'axis_quality_head', None)
    if aqh is not None:
        n = sum(p.numel() for p in aqh.parameters())
        lines.append("  ADQM AxisQualityHead params: %s (%.4f%% of total)"
                     % (_fmt_params(n), 100.0 * n / max(params, 1)))

    del model_for_info

    return params, "\n".join(lines)

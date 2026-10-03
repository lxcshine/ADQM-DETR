import argparse
import json
import os
import sys
import time
from pathlib import Path

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values, width=40, vmin=None, vmax=None):
    """把数值序列渲染为终端 sparkline 曲线。"""
    if not values:
        return ""
    vals = values[-width:]
    lo = min(vals) if vmin is None else vmin
    hi = max(vals) if vmax is None else vmax
    if hi <= lo:
        return SPARK_CHARS[-1] * len(vals)
    n = len(SPARK_CHARS)
    return "".join(SPARK_CHARS[int((v - lo) / (hi - lo) * (n - 1))] for v in vals)


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def read_log_txt(log_dir):
    """增量读取 log.txt，返回 epoch 级记录列表（每行一个 JSON dict）。"""
    path = log_dir / "log.txt"
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def render_epoch_section(rows, width=76):
    lines = []
    if not rows:
        lines.append("  （暂无 epoch 记录，训练可能尚未完成第一个 epoch）")
        return lines
    losses = [r.get("train_loss") for r in rows if "train_loss" in r]
    epochs = [r.get("epoch") for r in rows]
    lines.append(f"  已完成 epoch: {len(rows)} 个（最新 epoch {epochs[-1]}）")
    if losses:
        lines.append(f"  train_loss 趋势: {sparkline(losses, width)}  "
                      f"最新 {fmt(losses[-1])}  最优 {fmt(min(losses))}")
    for r in rows[-3:]:  # 最近 3 个 epoch 的评估指标
        evals = r.get("test_coco_eval")
        if evals and len(evals) >= 3:
            # COCO 顺序: [AP@.5:.95, AP@.5, AP@.75]
            lines.append(f"    epoch {r.get('epoch')}: mAP50 {evals[1]:.4f} | "
                         f"mAP75 {evals[2]:.4f} | mAP(.5:.95) {evals[0]:.4f}")
    maps = [r["test_coco_eval"][1] for r in rows
            if r.get("test_coco_eval") and len(r["test_coco_eval"]) >= 3]
    if maps:
        lines.append(f"  mAP50 趋势:  {sparkline(maps, width)}  "
                      f"最新 {fmt(maps[-1])}  最优 {fmt(max(maps))}")
    return lines


def find_event_dir(log_dir):
    """定位 TensorBoard event 目录（writer 写在 output_dir/summary 下）。"""
    for cand in [log_dir / "summary", log_dir]:
        if cand.is_dir() and any(cand.glob("events.out.tfevents.*")):
            return cand
        # SummaryWriter(log_dir/xxx) 会在子目录再建一层
        if cand.is_dir():
            for sub in cand.iterdir():
                if sub.is_dir() and any(sub.glob("events.out.tfevents.*")):
                    return sub
    return None


def render_event_section(event_dir, width=76):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 5000})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return ["  （TensorBoard 中暂无标量记录）"], None, ""

    lines = []
    step_now = None
    speed_str = ""
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        if not events:
            continue
        vals = [e.value for e in events]
        steps = [e.step for e in events]
        lines.append(f"  {tag:<24} step {steps[-1]:<7} 最新 {fmt(vals[-1]):<12} "
                     f"{sparkline(vals, width - 46)}")
    # 当前 iter 步数与速度以 Loss/total 为准（Test/* 的 step 是 epoch 级）
    if 'Loss/total' in tags:
        events = ea.Scalars('Loss/total')
        if events:
            step_now = events[-1].step
            win = events[-50:]
            if len(win) >= 2:
                dt = win[-1].wall_time - win[0].wall_time
                ds = win[-1].step - win[0].step
                if dt > 0 and ds > 0:
                    speed_str = f"，最近 {ds / dt:.2f} iter/s"
    return lines, step_now, speed_str


def render(log_dir, interval_start):
    rows = read_log_txt(log_dir)
    event_dir = find_event_dir(log_dir)

    out = []
    out.append("=" * 78)
    out.append(f" RiO-DETR 训练实时监控   log_dir: {log_dir}   "
               f"刷新时刻 {time.strftime('%H:%M:%S')}")
    out.append("=" * 78)
    out.append("── epoch 级（log.txt）────────────────────────────────────")
    out.extend(render_epoch_section(rows))
    out.append("── iter 级（TensorBoard，每 10 iter 更新）────────────────")
    if event_dir is not None:
        lines, step_now, speed_str = render_event_section(event_dir)
        out.extend(lines)
        if step_now is not None:
            out.append(f"  当前累计 iter step: {step_now}{speed_str}")
    else:
        out.append("  （未找到 TensorBoard event 文件，训练启动后自动出现）")
    out.append("=" * 78)
    out.append(" 浏览器曲线: tensorboard --logdir <log-dir>   退出监控: Ctrl+C")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="RiO-DETR 训练实时监控（只读）")
    parser.add_argument("--log-dir", type=str, required=True,
                        help="训练 output_dir（含 log.txt 与 summary/）")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="刷新间隔秒数（默认 5）")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 终端 UTF-8 输出

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"目录不存在: {log_dir}（训练启动后再运行本脚本）")
        sys.exit(1)

    t0 = time.time()
    try:
        while True:
            text = render(log_dir, t0)
            os.system("cls" if os.name == "nt" else "clear")
            print(text, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n监控已退出。")


if __name__ == "__main__":
    main()

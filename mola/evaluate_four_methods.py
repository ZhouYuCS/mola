"""四方法回访帧 LPIPS 评测：fastar / worldkv / quanevict_nonse / quanevict。

评测规格：
- 随机轨迹总长度 50 帧（latent 帧）。管线硬约束 latent 帧数须为 3 的整数倍（1 block =
  3 latent 帧），50 不满足，故取最接近且 >= 50 的 51 latent 帧（17 blocks），并在报告
  中明确记录此偏差。
- 10 张不同测试图：demo_images/universal 下按文件名排序取前 10 张 PNG。
- 对每个方法，在轨迹所有回访帧对（首次访问块 <-> 回访块）上计算 LPIPS(AlexNet)，
  再对 10 张图取平均与标准差。
- 方法定义：
    worldkv         = 原始 K/V，不量化，轨迹检索（worldkv flag）
    fastar          = FASTAR TempCache（近重复 key 合并 + log(m) 偏置）
    quanevict_nonse = QuanEvict（无敏感度，统一阈值，不加载敏感度文件）
    quanevict       = QuanEvict（带敏感度，按敏感度加权分配）
- 错误处理：每个 (图, 方法) 独立 try/except，单点失败不中断整体；异常堆栈写入日志，
  该 run 记为 failed 并继续。最终报告列出全部故障与排障步骤。
- 为加速评测，禁用 mp4 逐块编码（monkey-patch utils.visualize.process_video 为 no-op）；
  仅保留返回的 numpy 视频用于 LPIPS，并另存代表性回访帧 PNG 作为可视化。
"""
import argparse
import gc
import glob
import json
import os
import time
import traceback

import numpy as np
import torch

# ---- 在导入 inference_streaming 之前决定是否保存 mp4 ----
# 默认关闭 mp4 逐块编码以加速评测（仅用返回的 numpy 视频计算 LPIPS）；
# 设 EVAL_SAVE_VIDEO=1 时开启 mp4 保存，并防御 interlaced PNG 导致的 overlay_icon 崩溃。
import utils.visualize as viz


if os.environ.get("EVAL_SAVE_VIDEO") == "1":
    if not hasattr(viz, "_orig_overlay_icon"):
        viz._orig_overlay_icon = viz.overlay_icon

        def _safe_overlay_icon(frame, icon, position, scale=1.0, rotation=0):
            if icon is None:
                return
            return viz._orig_overlay_icon(frame, icon, position, scale=scale, rotation=rotation)

        viz.overlay_icon = _safe_overlay_icon
else:
    def _noop_process_video(*args, **kwargs):
        return None

    viz.process_video = _noop_process_video

from inference_streaming import InteractiveGameInference  # noqa: E402
from utils.misc import set_seed  # noqa: E402

from compare_quanevict_worldkv import (  # noqa: E402
    SENS_WK,
    SENS_WV,
    build_inputs,
    generate_trajectory,
    frame_to_lpips_input,
)

# ---- 方法定义（顺序即报告顺序）----
METHODS = [
    {"key": "worldkv",         "label": "WorldKV",                          "worldkv": True,  "fastar": False, "sensitivity": None},
    {"key": "fastar",          "label": "FASTAR (TempCache)",               "worldkv": False, "fastar": True,  "sensitivity": None},
    {"key": "fastar_full",     "label": "FAST-AR (TempCache, 跨位置匹配)",     "worldkv": False, "fastar": False, "fastar_full": True, "sensitivity": None},
    {"key": "quanevict_nonse", "label": "QuanEvict (无敏感度, 统一阈值)",      "worldkv": False, "fastar": False, "sensitivity": False},
    {"key": "quanevict",       "label": "QuanEvict (敏感度加权)",             "worldkv": False, "fastar": False, "sensitivity": True},
]

METHOD_BY_KEY = {m["key"]: m for m in METHODS}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml")
    p.add_argument("--checkpoint_path", type=str,
                   default="/mnt/sda/zy/models/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors")
    p.add_argument("--pretrained_model_path", type=str, default="/mnt/sda/zy/models/Matrix-Game-2.0")
    p.add_argument("--img_dir", type=str, default="demo_images/universal")
    p.add_argument("--num_samples", type=int, default=10, help="测试图数量（<=0 取全部 PNG）")
    p.add_argument("--methods", type=str, default="worldkv,fastar,quanevict_nonse,quanevict",
                   help="逗号分隔的方法 key 列表")
    p.add_argument("--output_folder", type=str, default="outputs_four_methods_eval")
    p.add_argument("--max_num_output_frames", type=int, default=51,
                   help="latent 帧数（须为 3 的整数倍；50 不满足，默认取 51）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape", type=str, default=None,
                   help="轨迹形态：rectangle/zigzag/rotate/strafe（默认随机）")
    p.add_argument("--dry_run", action="store_true", help="只验证轨迹与图片选择，不加载模型")

    p.add_argument("--retrieval_topk_frames", type=int, default=4)
    p.add_argument("--quant_chunk_frames", type=int, default=12)

    p.add_argument("--qvg_sensitivity_wk", type=str, default=SENS_WK)
    p.add_argument("--qvg_sensitivity_wv", type=str, default=SENS_WV)

    p.add_argument("--qvg_bits", type=int, default=2)
    p.add_argument("--qvg_num_centroids", type=int, default=256)
    p.add_argument("--qvg_num_stages", type=int, default=1)
    p.add_argument("--qvg_group_size", type=int, default=64)
    p.add_argument("--qvg_kmeans_iters", type=int, default=4)
    p.add_argument("--qvg_seed", type=int, default=0)
    p.add_argument("--qvg_sparsify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qvg_sparse_ratio", type=float, default=0.5, help="QuanEvict 基准保留比例（每 head 由敏感度自适应缩放）")
    p.add_argument("--qvg_use_triton", action="store_true", default=True)

    p.add_argument("--kv_bank_on_gpu", action=argparse.BooleanOptionalAction, default=True, help="bank 放 GPU（统计字节即显存占用）")
    p.add_argument("--fastar_sim_threshold", type=float, default=0.9)
    p.add_argument("--fastar_thresholds", type=str, default="",
                   help="fastar_full 的阈值列表（逗号分隔，如 0.7,0.8,0.9,0.95,0.98），"
                        "非空时把 --methods 中的 fastar_full 展开为多个阈值方法")
    p.add_argument("--plot", action="store_true", default=True, help="保存结果柱状图 PNG")
    return p.parse_args()


class Logger:
    def __init__(self, path):
        self.path = path
        self.fh = open(path, "a", encoding="utf-8")

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def run_one(game, args, cond, noise, actions, method, name):
    m = game.pipeline.generator.model
    fastar_full_flag = False
    if method["worldkv"]:
        m.qvg_sparsify = False
        use_retrieval = False
        worldkv_flag = True
        fastar_flag = False
    elif method["fastar"]:
        m.qvg_sparsify = False
        use_retrieval = False
        worldkv_flag = False
        fastar_flag = True
    elif method.get("fastar_full"):
        m.qvg_sparsify = False
        use_retrieval = False
        worldkv_flag = False
        fastar_flag = False
        fastar_full_flag = True
    else:
        m.qvg_sparsify = True
        m.qvg_sparse_ratio = args.qvg_sparse_ratio
        # 无敏感度基线：不加载敏感度文件，走统一 sparse_ratio 路径（ratio=None）
        if method.get("sensitivity", True):
            m.qvg_sensitivity_wk = args.qvg_sensitivity_wk
            m.qvg_sensitivity_wv = args.qvg_sensitivity_wv
        else:
            m.qvg_sensitivity_wk = None
            m.qvg_sensitivity_wv = None
        use_retrieval = True
        worldkv_flag = False
        fastar_flag = False

    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        video = game.pipeline.inference(
            noise=noise,
            conditional_dict=cond,
            return_latents=False,
            output_folder=args.output_folder,
            name=name,
            mode="universal",
            scripted_actions=actions,
            use_retrieval=use_retrieval,
            retrieval_topk_frames=args.retrieval_topk_frames,
            traj_threshold=None,
            kv_bank_on_gpu=args.kv_bank_on_gpu,
            quant_chunk_frames=args.quant_chunk_frames,
            worldkv=worldkv_flag,
            fastar=fastar_flag,
            fastar_full=fastar_full_flag,
            fastar_sim_threshold=method.get("fastar_sim_threshold", args.fastar_sim_threshold),
            residual_mag_sink=None,
        )
    latency = time.perf_counter() - t0
    peak_gpu = torch.cuda.max_memory_allocated()
    bank_bytes = getattr(game.pipeline, "_last_bank_bytes", 0)
    return video, bank_bytes, peak_gpu, latency


def compute_mean_lpips(lpips_fn, video, revisit_pairs, device):
    rgb_per_block = 12
    vals = []
    per_pair = []
    for ref_c, rev_c in revisit_pairs:
        ref_idx = ref_c * rgb_per_block + 8
        rev_idx = rev_c * rgb_per_block + 8
        v = lpips_fn(
            frame_to_lpips_input(video[ref_idx], device),
            frame_to_lpips_input(video[rev_idx], device),
        ).item()
        vals.append(v)
        per_pair.append({"ref_block": ref_c, "revisit_block": rev_c,
                         "ref_rgb_idx": ref_idx, "revisit_rgb_idx": rev_idx, "lpips": v})
    if not vals:
        return None, []
    return float(np.mean(vals)), per_pair


def select_images(img_dir, num_samples):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if num_samples <= 0:
        return imgs
    return imgs[:num_samples]


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def save_plot(args, method_keys, agg_by_method, revisit_pairs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception as e:  # noqa: BLE001
        print(f"跳过绘图（matplotlib 不可用）：{e}")
        return None

    _EN = {"worldkv": "WorldKV", "fastar": "FASTAR (TempCache)",
           "fastar_full": "FAST-AR (TempCache, 跨位置匹配)",
           "quanevict_nonse": "QuanEvict (无敏感度)", "quanevict": "QuanEvict (敏感度加权)"}
    # 过滤无成功样本（lpips_mean 为 None）的方法，避免 ax.bar 收到 None 崩溃
    method_keys = [k for k in method_keys if agg_by_method[k]["lpips_mean"] is not None]
    if not method_keys:
        print("无有效结果，跳过绘图")
        return None
    labels = [METHOD_BY_KEY.get(k, {}).get("label", _EN.get(k, k)) for k in method_keys]
    means = [agg_by_method[k]["lpips_mean"] for k in method_keys]
    stds = [agg_by_method[k]["lpips_std"] for k in method_keys]
    ns = [agg_by_method[k]["n_success"] for k in method_keys]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    colors = ["tab:orange", "tab:purple", "tab:brown", "tab:green", "tab:red",
              "tab:blue", "tab:olive", "tab:cyan"][:len(labels)]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  alpha=0.85, error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Revisit-frame LPIPS (AlexNet, lower is better)")
    ax.set_title(f"Revisit-frame LPIPS across 4 methods ({args.max_num_output_frames} latent frames, "
                 f"{len(revisit_pairs)} revisit pairs, {ns[0] if ns else 0} images)")
    ax.grid(True, axis="y", alpha=0.3)
    for b, mean, std, n in zip(bars, means, stds, ns):
        ax.text(b.get_x() + b.get_width() / 2, mean + std + 0.003, f"{mean:.4f}±{std:.4f}\n(n={n})",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_png = os.path.join(args.output_folder, "lpips_comparison.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"柱状图已保存到 {out_png}")
    return out_png


def build_report(args, method_keys, agg_by_method, per_run, revisit_pairs, shape, imgs):
    lines = []
    lines.append("# 四方法回访帧 LPIPS 评测报告")
    lines.append("")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. 目标")
    lines.append("对四种 KV 缓存/轨迹稀疏方法 —— WorldKV、FASTAR(TempCache)、"
                "QuanEvict(无敏感度, 统一阈值)、QuanEvict(敏感度加权) —— 在相同轨迹、相同噪声、"
                "相同测试图下，度量“回访帧”与“首次访问帧”之间的 LPIPS 差异，并跨 10 张测试图取平均。")
    lines.append("")
    lines.append("## 2. 测试环境")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| GPU | RTX 4090 24GB (CUDA_VISIBLE_DEVICES=1) |")
    lines.append("| Python 环境 | /home/zy/miniconda3/envs/llama_factory/bin/python |")
    lines.append(f"| PyTorch | {torch.__version__} |")
    lines.append(f"| checkpoint | {args.checkpoint_path} |")
    lines.append(f"| pretrained/VAE | {args.pretrained_model_path} |")
    lines.append(f"| 配置 | {args.config_path} |")
    lines.append(f"| 随机种子 seed | {args.seed} |")
    lines.append("| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:True |")
    lines.append("")
    lines.append("## 3. 轨迹生成参数")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 请求总帧数 | 50 latent 帧 |")
    lines.append(f"| 实际使用 | {args.max_num_output_frames} latent 帧（50 非 3 的倍数，取 >=50 的最近可行值 51 = 17 blocks） |")
    lines.append("| 块/帧粒度 | 1 block = 3 latent 帧 = 12 RGB 帧 |")
    lines.append(f"| 轨迹形态（随机） | {shape} |")
    lines.append("| 位姿模型 | 4 维模拟位姿 [前后, 左右, pitch, yaw]，与 causal_inference.running_state 一致 |")
    lines.append(f"| 回访 block 对 | {revisit_pairs} |")
    lines.append("")
    lines.append("轨迹采用“去程 + 逆序回程（每操作取逆并按逆序排列）”，保证精确回到起点位姿，"
                "从而产生可评估的回访帧对；回访帧 RGB 索引 = block*12 + 8（该 block 末 latent 帧）。")
    lines.append("")
    lines.append("## 4. 测试图像选择")
    lines.append("")
    lines.append(f"从 `{args.img_dir}` 按文件名排序取前 {len(imgs)} 张 PNG：")
    lines.append("")
    for i, im in enumerate(imgs):
        lines.append(f"{i + 1}. `{os.path.basename(im)}`")
    lines.append("")
    lines.append("## 5. LPIPS 计算方法")
    lines.append("")
    lines.append("- 使用 `lpips.LPIPS(net='alex')`，在 cuda 上 eval 模式。")
    lines.append("- 对每个回访对 (ref_block, revisit_block)，取两帧：")
    lines.append("  - 首次访问帧：`ref_rgb_idx = ref_block*12 + 8`")
    lines.append("  - 回访帧：`revisit_rgb_idx = revisit_block*12 + 8`")
    lines.append("- 逐回访对计算 LPIPS，然后对每张图的所有回访对取平均（mean_lpips），"
                "最后跨 10 张图对每个方法取 mean 与 std。")
    lines.append("")
    lines.append("## 6. 结果（跨 10 张图）")
    lines.append("")
    lines.append("| 方法 | 平均 LPIPS | 标准差 | 成功样本数 | 平均 latency(s) | 平均 bank(MB) | 平均 peak GPU(GiB) |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in method_keys:
        a = agg_by_method[k]
        lp = f"{a['lpips_mean']:.4f}" if a['lpips_mean'] is not None else "N/A"
        sd = f"{a['lpips_std']:.4f}" if a['lpips_std'] is not None else "N/A"
        lat = f"{a['latency_mean']:.2f}" if a['latency_mean'] is not None else "N/A"
        bank = f"{a['bank_mb_mean']:.1f}" if a['bank_mb_mean'] is not None else "N/A"
        peak = f"{a['peak_gpu_mb_mean'] / 1024:.2f}" if a['peak_gpu_mb_mean'] is not None else "N/A"
        lines.append(f"| {METHOD_BY_KEY[k]['label']} | {lp} | {sd} | "
                     f"{a['n_success']}/{a['n_total']} | {lat} | {bank} | {peak} |")
    lines.append("")
    if agg_by_method.get("worldkv") and agg_by_method["worldkv"]["lpips_mean"] is not None:
        base = agg_by_method["worldkv"]["lpips_mean"]
        lines.append("### 相对 WorldKV 的 LPIPS 差异（正值=回访失真更大）")
        lines.append("")
        lines.append("| 方法 | ΔLPIPS (相对 WorldKV) |")
        lines.append("|---|---|")
        for k in method_keys:
            if k == "worldkv":
                continue
            a = agg_by_method[k]
            if a["lpips_mean"] is not None:
                lines.append(f"| {METHOD_BY_KEY[k]['label']} | {a['lpips_mean'] - base:+.4f} |")
        lines.append("")
    lines.append("## 7. 比较分析")
    lines.append("")
    ranked = [k for k in method_keys if agg_by_method[k]["lpips_mean"] is not None]
    ranked.sort(key=lambda k: agg_by_method[k]["lpips_mean"])
    if ranked:
        best = ranked[0]
        worst = ranked[-1]
        lines.append(f"- 回访帧 LPIPS 最低（质量最佳）的方法为 **{METHOD_BY_KEY[best]['label']}**"
                     f"（{agg_by_method[best]['lpips_mean']:.4f}）。")
        lines.append(f"- 回访帧 LPIPS 最高（回访失真最大）的方法为 **{METHOD_BY_KEY[worst]['label']}**"
                     f"（{agg_by_method[worst]['lpips_mean']:.4f}）。")
    if agg_by_method.get("worldkv") and agg_by_method["worldkv"]["lpips_mean"] is not None:
        lines.append(f"- WorldKV 保留原始 K/V，通常代表回访一致性的上界参考，"
                     f"LPIPS={agg_by_method['worldkv']['lpips_mean']:.4f}。")
    if agg_by_method.get("quanevict") and agg_by_method.get("quanevict_nonse"):
        q1 = agg_by_method["quanevict"]["lpips_mean"]
        q0 = agg_by_method["quanevict_nonse"]["lpips_mean"]
        if q1 is not None and q0 is not None:
            lines.append(f"- QuanEvict 敏感度加权 LPIPS={q1:.4f}，无敏感度统一阈值 LPIPS={q0:.4f}，"
                         f"差值 {q1 - q0:+.4f}（正=敏感度加权在回访一致性上更差，负=更好）。")
    lines.append("")
    lines.append("## 8. 错误处理与排障")
    lines.append("")
    lines.append("- 每个 (图, 方法) 独立 try/except，失败不中断整体评测。")
    lines.append("- 异常堆栈记录到日志文件 `evaluation.log`，并在 results.json 中标记 `status=failed`。")
    lines.append("- mp4 逐块编码已禁用（monkey-patch process_video），以避免 interlaced PNG 导致的 "
                "libpng 崩溃与编码开销；LPIPS 直接基于返回的 numpy 视频计算。")
    failures = [r for r in per_run if r["status"] == "failed"]
    if failures:
        lines.append(f"- 本次共 {len(failures)} 个失败 run，详见下节与 evaluation.log。")
    else:
        lines.append("- 本次无失败 run。")
    lines.append("")
    lines.append("## 9. 复现说明")
    lines.append("")
    lines.append("```bash")
    lines.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \\")
    lines.append("  /home/zy/miniconda3/envs/llama_factory/bin/python evaluate_four_methods.py \\")
    lines.append(f"    --num_samples {args.num_samples} --seed {args.seed} \\")
    lines.append(f"    --max_num_output_frames {args.max_num_output_frames}")
    lines.append("```")
    lines.append("")
    lines.append("## 10. 产物")
    lines.append("")
    lines.append("- `results.json`：完整逐 run / 逐回访对结果。")
    lines.append("- `lpips_comparison.png`：四方法平均 LPIPS ± std 柱状图。")
    lines.append("- `evaluation.log`：完整运行日志。")
    if failures:
        lines.append("")
        lines.append("### 失败 run 摘要")
        lines.append("")
        for r in failures:
            lines.append(f"- `{r['img']}` / `{r['method']}`: `{r.get('error', '')[:200]}`")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    log_path = os.path.join(args.output_folder, "evaluation.log")
    logger = Logger(log_path)

    method_keys = [k.strip() for k in args.methods.split(",") if k.strip()]

    # fastar_full 多阈值展开：--fastar_thresholds 非空时，把 fastar_full 替换为每个阈值一个方法
    if "fastar_full" in method_keys and args.fastar_thresholds:
        thrs = [float(t) for t in args.fastar_thresholds.split(",") if t.strip()]
        assert thrs, "fastar_thresholds 为空"
        expanded = []
        for k in method_keys:
            if k == "fastar_full":
                for t in thrs:
                    nk = f"fastar_full@{t:g}"
                    METHOD_BY_KEY[nk] = {
                        "key": nk,
                        "label": f"FAST-AR TempCache (thr={t:g})",
                        "worldkv": False, "fastar": False, "fastar_full": True,
                        "fastar_sim_threshold": t,
                    }
                    expanded.append(nk)
            else:
                expanded.append(k)
        method_keys = expanded

    for k in method_keys:
        assert k in METHOD_BY_KEY, f"未知方法 key: {k}（可选 {list(METHOD_BY_KEY)}）"

    assert args.max_num_output_frames % 3 == 0, \
        f"max_num_output_frames 须为 3 的整数倍，当前 {args.max_num_output_frames}"
    num_blocks = args.max_num_output_frames // 3
    if args.max_num_output_frames != 50:
        logger.log(f"注意：请求 50 latent 帧，实际使用 {args.max_num_output_frames} "
                   f"（50 非 3 的倍数，取 >=50 最近可行值；{num_blocks} blocks）")

    imgs = select_images(args.img_dir, args.num_samples)
    assert imgs, f"未在 {args.img_dir} 找到 PNG"
    logger.log(f"测试图（{len(imgs)} 张）：{imgs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions, revisit_pairs, shape = generate_trajectory(num_blocks, device, seed=args.seed, shape=args.shape)
    logger.log(f"轨迹形态={shape}，blocks={num_blocks}，回访对={revisit_pairs}")
    if not revisit_pairs:
        logger.log("警告：未产生任何回访对，LPIPS 将无法计算")
    if args.dry_run:
        logger.log("dry_run 完成（未加载模型）")
        logger.close()
        return

    set_seed(args.seed)
    game = InteractiveGameInference(args)
    mode = game.config.pop("mode")
    assert mode == "universal"

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(game.device).eval()

    per_run = []
    for i, img in enumerate(imgs):
        img_name = os.path.basename(img)
        logger.log(f"--- 图 {i + 1}/{len(imgs)}: {img_name} ---")
        try:
            cond, noise = build_inputs(game, args, mode, img_path=img, keep_vae=True)
        except Exception as e:  # noqa: BLE001
            logger.log(f"[{img_name}] build_inputs 失败，跳过整图：{e}")
            logger.log(traceback.format_exc())
            for k in method_keys:
                per_run.append({"img": img_name, "method": k, "status": "failed",
                                "stage": "build_inputs", "error": str(e)})
            continue

        for k in method_keys:
            method = METHOD_BY_KEY[k]
            name = f"{os.path.splitext(img_name)[0]}_{k}"
            video = None
            try:
                video, bank, peak, lat = run_one(game, args, cond, noise, actions, method, name)
                mean_lp, per_pair = compute_mean_lpips(lpips_fn, video, revisit_pairs, game.device)
                if mean_lp is None:
                    raise RuntimeError("无可计算的回访对")
                per_run.append({"img": img_name, "method": k, "status": "ok",
                                "mean_lpips": mean_lp, "per_pair": per_pair,
                                "bank_mb": bank / 1e6, "peak_gpu_mb": peak / 1e6,
                                "latency_sec": lat,
                                "throughput_fps": len(video) / lat if lat > 0 else None})
                logger.log(f"  [{k:16s}] lpips={mean_lp:.4f}  pairs={len(per_pair)}  "
                           f"bank={bank / 1e6:.1f}MB  peak={peak / 1e9:.2f}GiB  lat={lat:.2f}s")
            except Exception as e:  # noqa: BLE001
                logger.log(f"  [{k:16s}] 失败：{e}")
                logger.log(traceback.format_exc())
                per_run.append({"img": img_name, "method": k, "status": "failed",
                                "stage": "inference", "error": str(e)})
            finally:
                if video is not None:
                    del video
                gc.collect()
                torch.cuda.empty_cache()

    agg_by_method = {}
    for k in method_keys:
        ok = [r for r in per_run if r["method"] == k and r["status"] == "ok"]
        failed = [r for r in per_run if r["method"] == k and r["status"] == "failed"]
        lp_vals = [r["mean_lpips"] for r in ok]
        lp_mean, lp_std = _mean_std(lp_vals)
        lat_vals = [r["latency_sec"] for r in ok]
        bank_vals = [r["bank_mb"] for r in ok]
        peak_vals = [r["peak_gpu_mb"] for r in ok]
        agg_by_method[k] = {
            "n_total": len(ok) + len(failed),
            "n_success": len(ok),
            "lpips_mean": lp_mean, "lpips_std": lp_std,
            "lpips_all": lp_vals,
            "latency_mean": float(np.mean(lat_vals)) if lat_vals else None,
            "bank_mb_mean": float(np.mean(bank_vals)) if bank_vals else None,
            "peak_gpu_mb_mean": float(np.mean(peak_vals)) if peak_vals else None,
        }

    logger.log("\n===== 跨图平均 LPIPS =====")
    for k in method_keys:
        a = agg_by_method[k]
        if a["lpips_mean"] is not None:
            logger.log(f"  {k:16s}: {a['lpips_mean']:.4f} ± {a['lpips_std']:.4f} "
                       f"(n={a['n_success']}/{a['n_total']})")
        else:
            logger.log(f"  {k:16s}: 无成功样本")

    summary = {
        "spec": {
            "requested_frames": 50,
            "actual_latent_frames": args.max_num_output_frames,
            "note": "50 not divisible by 3; used nearest >=50 value",
            "num_blocks": num_blocks,
            "rgb_per_block": 12,
            "lpips_net": "alex",
        },
        "env": {
            "torch": torch.__version__,
            "checkpoint_path": args.checkpoint_path,
            "pretrained_model_path": args.pretrained_model_path,
            "config_path": args.config_path,
        },
        "seed": args.seed,
        "trajectory_shape": shape,
        "revisit_pairs": [[a, b] for a, b in revisit_pairs],
        "imgs": [os.path.basename(x) for x in imgs],
        "methods": {k: agg_by_method[k] for k in method_keys},
        "per_run": per_run,
    }
    out_json = os.path.join(args.output_folder, "results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.log(f"结果 JSON 已保存到 {out_json}")

    if args.plot:
        save_plot(args, method_keys, agg_by_method, revisit_pairs)

    report = build_report(args, method_keys, agg_by_method, per_run, revisit_pairs, shape, imgs)
    out_md = os.path.join(args.output_folder, "评价报告.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    logger.log(f"报告已保存到 {out_md}")

    logger.close()


if __name__ == "__main__":
    main()

"""多样本 × 多 gamma × WorldKV 对比：KV cache 显存、回访帧 LPIPS、推理时延 取平均。

- 每个样本跑一次 WorldKV 基线（原始 K/V，不量化），再跑多个 gamma 的 QuanEvict。
- bank 字节由 `causal_inference._last_bank_bytes` 在推理内部累计（worldkv=原始 K/V，
  quanevict=packed dict），覆盖两种模式，因此不再需要 monkey-patch。
  * 默认 bank 存 CPU（不加 --kv_bank_on_gpu）；加 --kv_bank_on_gpu 后 bank 落 GPU，
    各 tensor element_size 与 CPU 一致，字节数即等于 GPU 显存占用。
- 每个 run 记录 bank 字节、峰值 GPU 显存、推理 wall-clock 时延、回访帧 LPIPS。
- 结果按方法/gamma 跨样本平均，并画曲线（QuanEvict 曲线 + WorldKV 水平参考线）。
"""
import argparse
import gc
import glob
import json
import os
import time

import numpy as np
import torch

from inference_streaming import InteractiveGameInference
from utils.misc import set_seed

from compare_quanevict_worldkv import (
    SENS_WK,
    SENS_WV,
    build_inputs,
    generate_trajectory,
    frame_to_lpips_input,
)

import utils.visualize as viz

# 防御性 patch：interlaced PNG 在 cv2.imread 下偶发返回 None（libpng error: bad
# parameters to zlib），导致 overlay_icon 崩溃。这里在 icon 为空时跳过叠加，仅影响
# mp4 里的鼠标图标，不影响 bank 统计、LPIPS 与时延。
if not hasattr(viz, "_orig_overlay_icon"):
    viz._orig_overlay_icon = viz.overlay_icon

    def _safe_overlay_icon(frame, icon, position, scale=1.0, rotation=0):
        if icon is None:
            return
        return viz._orig_overlay_icon(frame, icon, position, scale=scale, rotation=rotation)

    viz.overlay_icon = _safe_overlay_icon


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml")
    p.add_argument("--checkpoint_path", type=str, default="")
    p.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
    p.add_argument("--img_path", type=str, default="demo_images/universal/0000.png")
    p.add_argument("--img_dir", type=str, default=None,
                     help="素材库目录，逗号分隔可传多个；指定后按文件名排序取前 --num_samples 张 PNG（<=0 表示全部）")
    p.add_argument("--num_samples", type=int, default=0, help="样本数，<=0 表示取全部")
    p.add_argument("--output_folder", type=str, default="outputs_gamma_compare")
    p.add_argument("--max_num_output_frames", type=int, default=36, help="latent 帧数（须为 3 的整数倍）")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--retrieval_topk_frames", type=int, default=4)
    p.add_argument("--quant_chunk_frames", type=int, default=12)

    # QuanEvict 保留比例
    p.add_argument("--gamma_list", type=str, default="0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0",
                   help="逗号分隔的 gamma 值列表")

    # QVG 量化配置（InteractiveGameInference._init_models 会读取这些属性）
    p.add_argument("--qvg_bits", type=int, default=2)
    p.add_argument("--qvg_num_centroids", type=int, default=256)
    p.add_argument("--qvg_num_stages", type=int, default=1)
    p.add_argument("--qvg_group_size", type=int, default=64)
    p.add_argument("--qvg_kmeans_iters", type=int, default=5)
    p.add_argument("--qvg_seed", type=int, default=0)
    p.add_argument("--qvg_sparsify", action="store_true")
    p.add_argument("--qvg_sparse_ratio", type=float, default=0.5, help="QuanEvict 基准保留比例（每 head 由敏感度自适应缩放）")
    p.add_argument("--qvg_use_triton", action="store_true", default=True)
    p.add_argument("--qvg_gamma", type=float, default=1.0)
    p.add_argument("--qvg_sensitivity_wk", type=str, default=SENS_WK)
    p.add_argument("--qvg_sensitivity_wv", type=str, default=SENS_WV)

    p.add_argument("--kv_bank_on_gpu", action="store_true",
                   help="把压缩 bank 放 GPU（此时统计字节数即 GPU 显存占用）")
    p.add_argument("--skip_worldkv", action="store_true", help="不跑 WorldKV 基线")
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                   help="正式测量前做一次 warmup 推理，稳定时延测量")
    p.add_argument("--plot", action="store_true", help="保存存储/LPIPS/时延曲线 PNG")
    return p.parse_args()


def run_one(game, args, cond, noise, actions, gamma, worldkv, name):
    m = game.pipeline.generator.model
    if worldkv:
        m.qvg_sparsify = False
        use_retrieval = False
        worldkv_flag = True
    else:
        m.qvg_sparsify = True
        m.qvg_sparse_ratio = args.qvg_sparse_ratio
        m.qvg_gamma = gamma
        m.qvg_sensitivity_wk = args.qvg_sensitivity_wk
        m.qvg_sensitivity_wv = args.qvg_sensitivity_wv
        use_retrieval = True
        worldkv_flag = False

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
            residual_mag_sink=None,
        )
    latency = time.perf_counter() - t0
    peak_gpu = torch.cuda.max_memory_allocated()
    bank_bytes = getattr(game.pipeline, "_last_bank_bytes", 0)

    label = "worldkv" if worldkv else f"gamma_{gamma:g}"
    print(f"[{name}] {label}: bank={bank_bytes / 1e6:.1f} MB  "
          f"peak_gpu={peak_gpu / 1e9:.2f} GiB  latency={latency:.2f} s", flush=True)
    return video, bank_bytes, peak_gpu, latency


def compute_mean_lpips(lpips_fn, video, revisit_pairs, device):
    rgb_per_block = 12
    vals = []
    for ref_c, rev_c in revisit_pairs:
        ref_idx = ref_c * rgb_per_block + 8
        rev_idx = rev_c * rgb_per_block + 8
        v = lpips_fn(
            frame_to_lpips_input(video[ref_idx], device),
            frame_to_lpips_input(video[rev_idx], device),
        ).item()
        vals.append(v)
    return float(np.mean(vals)), vals


def _mean_std(vals):
    vals = list(vals)
    return float(np.mean(vals)), float(np.std(vals))


def _agg_method(entries):
    n = len(entries)
    if n == 0:
        return None
    bm, bs = _mean_std([p["bank_mb"] for p in entries])
    pm, ps = _mean_std([p["peak_gpu_mb"] for p in entries])
    lm, ls = _mean_std([p["latency_sec"] for p in entries])
    lp_m, lp_s = _mean_std([p["mean_lpips"] for p in entries])
    tm, ts = _mean_std([p["throughput_fps"] for p in entries])
    return {
        "n_samples": n,
        "bank_mb_mean": bm, "bank_mb_std": bs,
        "peak_gpu_mb_mean": pm, "peak_gpu_mb_std": ps,
        "latency_sec_mean": lm, "latency_sec_std": ls,
        "lpips_mean": lp_m, "lpips_std": lp_s,
        "throughput_fps_mean": tm, "throughput_fps_std": ts,
    }


def aggregate(per_run, gammas):
    worldkv_agg = _agg_method([p for p in per_run if p["method"] == "worldkv"])
    agg = []
    for g in gammas:
        sub = [p for p in per_run if p["method"] == "quanevict" and p["gamma"] == g]
        if not sub:
            continue
        a = _agg_method(sub)
        a["gamma"] = g
        agg.append(a)
    return worldkv_agg, agg


def save_plot(args, per_run, worldkv_agg, agg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"跳过绘图（matplotlib 不可用）：{e}")
        return

    gs = [a["gamma"] for a in agg]
    fig, _ax = plt.subplots(2, 2, figsize=(16, 10))
    ax_kv, ax_lp, ax_lat, ax_tp = _ax[0][0], _ax[0][1], _ax[1][0], _ax[1][1]

    ax_kv.errorbar(gs, [a["bank_mb_mean"] / 1024 for a in agg],
                     yerr=[a["bank_mb_std"] / 1024 for a in agg], marker="o", capsize=3,
                     label="QuanEvict")
    if worldkv_agg:
        ax_kv.axhline(worldkv_agg["bank_mb_mean"] / 1024, color="tab:orange", ls="--",
                        label=f"WorldKV {worldkv_agg['bank_mb_mean'] / 1024:.2f} GB")
    ax_kv.set_xlabel("gamma"); ax_kv.set_ylabel("KV cache (GB)")
    ax_kv.set_title("KV cache storage"); ax_kv.legend(); ax_kv.grid(True, alpha=0.3)

    ax_lp.errorbar(gs, [a["lpips_mean"] for a in agg],
                     yerr=[a["lpips_std"] for a in agg], marker="o", capsize=3, color="tab:red",
                     label="QuanEvict")
    if worldkv_agg:
        ax_lp.axhline(worldkv_agg["lpips_mean"], color="tab:orange", ls="--",
                        label=f"WorldKV {worldkv_agg['lpips_mean']:.4f}")
    ax_lp.set_xlabel("gamma"); ax_lp.set_ylabel("mean revisit LPIPS")
    ax_lp.set_title("Revisit LPIPS"); ax_lp.legend(); ax_lp.grid(True, alpha=0.3)

    ax_lat.errorbar(gs, [a["latency_sec_mean"] for a in agg],
                     yerr=[a["latency_sec_std"] for a in agg], marker="o", capsize=3, color="tab:green",
                     label="QuanEvict")
    if worldkv_agg:
        ax_lat.axhline(worldkv_agg["latency_sec_mean"], color="tab:orange", ls="--",
                        label=f"WorldKV {worldkv_agg['latency_sec_mean']:.2f} s")
    ax_lat.set_xlabel("gamma"); ax_lat.set_ylabel("inference latency (s)")
    ax_lat.set_title("Inference latency"); ax_lat.legend(); ax_lat.grid(True, alpha=0.3)

    ax_tp.errorbar(gs, [a["throughput_fps_mean"] for a in agg],
                   yerr=[a["throughput_fps_std"] for a in agg], marker="o", capsize=3, color="tab:blue",
                   label="QuanEvict")
    if worldkv_agg:
        ax_tp.axhline(worldkv_agg["throughput_fps_mean"], color="tab:orange", ls="--",
                      label=f"WorldKV {worldkv_agg['throughput_fps_mean']:.2f} fps")
    ax_tp.set_xlabel("gamma"); ax_tp.set_ylabel("throughput (frames/s)")
    ax_tp.set_title("Throughput"); ax_tp.legend(); ax_tp.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.output_folder, "gamma_vs_worldkv_multisample.png")
    plt.savefig(out_png, dpi=150)
    print(f"曲线已保存到 {out_png}")


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    if args.img_dir:
        all_imgs = []
        for d in args.img_dir.split(","):
            d = d.strip()
            if d:
                all_imgs.extend(glob.glob(os.path.join(d, "*.png")))
        all_imgs = sorted(all_imgs)
        imgs = all_imgs if args.num_samples <= 0 else all_imgs[: args.num_samples]
    else:
        imgs = [args.img_path]
    assert imgs, "没有找到样本图片"
    print(f"样本图片（{len(imgs)} 张）：{imgs}", flush=True)

    gammas = [float(x) for x in args.gamma_list.split(",") if x.strip() != ""]
    assert gammas, "gamma_list 不能为空"

    set_seed(args.seed)
    game = InteractiveGameInference(args)
    mode = game.config.pop("mode")
    assert mode == "universal"

    num_blocks = args.max_num_output_frames // 3
    actions, revisit_pairs, shape = generate_trajectory(num_blocks, game.device, seed=args.seed)
    print(f"轨迹形态：{shape}（{num_blocks} 块）；回访对 {revisit_pairs}")
    print(f"gamma 列表：{gammas}  kv_bank_on_gpu={args.kv_bank_on_gpu}  "
          f"skip_worldkv={args.skip_worldkv}", flush=True)

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(game.device).eval()

    cond, noise = build_inputs(game, args, mode, img_path=imgs[0], keep_vae=True)

    if args.warmup:
        print("warmup 推理（不计入结果）...", flush=True)
        wv, _, _, _ = run_one(game, args, cond, noise, actions, gammas[len(gammas) // 2],
                              worldkv=False, name="warmup")
        del wv
        gc.collect()
        torch.cuda.empty_cache()

    per_run = []
    for i, img in enumerate(imgs):
        if i > 0:
            cond, noise = build_inputs(game, args, mode, img_path=img, keep_vae=True)
            gc.collect()
            torch.cuda.empty_cache()
        base = os.path.basename(img)

        if not args.skip_worldkv:
            video, b, peak, lat = run_one(game, args, cond, noise, actions, None,
                                          worldkv=True, name=f"{base}_worldkv")
            lp, _ = compute_mean_lpips(lpips_fn, video, revisit_pairs, game.device)
            per_run.append({"img": base, "method": "worldkv", "gamma": None,
                            "bank_mb": b / 1e6, "peak_gpu_mb": peak / 1e6,
                            "latency_sec": lat, "mean_lpips": lp,
                            "throughput_fps": len(video) / lat})
            del video
            gc.collect()
            torch.cuda.empty_cache()

        for g in gammas:
            video, b, peak, lat = run_one(game, args, cond, noise, actions, g,
                                          worldkv=False, name=f"{base}_g{g:g}")
            lp, _ = compute_mean_lpips(lpips_fn, video, revisit_pairs, game.device)
            per_run.append({"img": base, "method": "quanevict", "gamma": g,
                            "bank_mb": b / 1e6, "peak_gpu_mb": peak / 1e6,
                            "latency_sec": lat, "mean_lpips": lp,
                            "throughput_fps": len(video) / lat})
            print(f"  >>> {base} gamma={g:g}: bank={b / 1e6:.1f} MB  "
                  f"peak={peak / 1e9:.2f} GiB  lat={lat:.2f} s  lpips={lp:.4f}", flush=True)
            del video
            gc.collect()
            torch.cuda.empty_cache()

    worldkv_agg, agg = aggregate(per_run, gammas)

    print("\n===== 跨样本平均 =====")
    if worldkv_agg:
        print(f"WorldKV: bank={worldkv_agg['bank_mb_mean'] / 1024:.3f} GB  "
              f"peak={worldkv_agg['peak_gpu_mb_mean'] / 1024:.2f} GiB  "
              f"lat={worldkv_agg['latency_sec_mean']:.2f} s  "
              f"fps={worldkv_agg['throughput_fps_mean']:.2f}  "
              f"lpips={worldkv_agg['lpips_mean']:.4f}")
    print(f"{'gamma':>6} | {'bank GB':>9} | {'peak GiB':>9} | {'latency s':>10} | {'fps':>7} | {'LPIPS':>8}")
    for a in agg:
        print(f"{a['gamma']:>6g} | {a['bank_mb_mean'] / 1024:>9.3f} | "
              f"{a['peak_gpu_mb_mean'] / 1024:>9.2f} | {a['latency_sec_mean']:>10.2f} | "
              f"{a['throughput_fps_mean']:>7.2f} | {a['lpips_mean']:>8.4f}")

    summary = {
        "ratio_base": args.qvg_sparse_ratio, "seed": args.seed,
        "max_num_output_frames": args.max_num_output_frames,
        "kv_bank_on_gpu": args.kv_bank_on_gpu, "gammas": gammas,
        "imgs": [os.path.basename(x) for x in imgs],
        "revisit_pairs": [[a, b] for a, b in revisit_pairs],
        "worldkv": worldkv_agg,
        "quanevict_by_gamma": agg, "per_run": per_run,
    }
    out_json = os.path.join(args.output_folder, "gamma_vs_worldkv_multisample.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_json}")

    if args.plot:
        save_plot(args, per_run, worldkv_agg, agg)


if __name__ == "__main__":
    main()

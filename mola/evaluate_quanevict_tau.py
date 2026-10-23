"""QuanEvict 多 tau（ratio_base）× gamma 回访帧 LPIPS 扫描评测。

规格与 evaluate_four_methods.py 一致：10 张测试图；请求 50 latent 帧，实际取 51（17 blocks）。
对每个 (ratio_base, gamma) 组合，在轨迹回访块对上算 LPIPS(AlexNet)，跨 10 图取 mean/std。
worldkv 为可选参考基线；默认禁用 mp4 编码（EVAL_SAVE_VIDEO=1 恢复）。
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml")
    p.add_argument("--checkpoint_path", type=str,
                   default="/mnt/sda/zy/models/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors")
    p.add_argument("--pretrained_model_path", type=str, default="/mnt/sda/zy/models/Matrix-Game-2.0")
    p.add_argument("--img_dir", type=str, default="demo_images/universal")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--output_folder", type=str, default="outputs_quanevict_tau_scan")
    p.add_argument("--max_num_output_frames", type=int, default=51,
                   help="latent 帧数（须为 3 的整数倍；50 不满足，默认取 51）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape", type=str, default=None)
    p.add_argument("--dry_run", action="store_true", help="只验证轨迹与图片选择，不加载模型")

    p.add_argument("--retrieval_topk_frames", type=int, default=4)
    p.add_argument("--quant_chunk_frames", type=int, default=12)

    p.add_argument("--ratio_base_list", type=str, default="0.3,0.5,0.7,0.9")
    p.add_argument("--gamma_list", type=str, default="0,1")
    p.add_argument("--skip_worldkv", action="store_true", help="不跑 WorldKV 基线")

    p.add_argument("--qvg_sensitivity_wk", type=str, default=SENS_WK)
    p.add_argument("--qvg_sensitivity_wv", type=str, default=SENS_WV)

    p.add_argument("--qvg_bits", type=int, default=2)
    p.add_argument("--qvg_num_centroids", type=int, default=256)
    p.add_argument("--qvg_num_stages", type=int, default=1)
    p.add_argument("--qvg_group_size", type=int, default=64)
    p.add_argument("--qvg_kmeans_iters", type=int, default=5)
    p.add_argument("--qvg_seed", type=int, default=0)
    p.add_argument("--qvg_sparsify", action="store_true")
    p.add_argument("--qvg_sparse_ratio", type=float, default=0.5)
    p.add_argument("--qvg_gamma", type=float, default=1.0)
    p.add_argument("--qvg_use_triton", action="store_true", default=True)

    p.add_argument("--kv_bank_on_gpu", action="store_true")
    p.add_argument("--plot", action="store_true", default=True)
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


def run_one(game, args, cond, noise, actions, ratio, gamma, worldkv, name):
    m = game.pipeline.generator.model
    if worldkv:
        m.qvg_sparsify = False
        use_retrieval = False
        worldkv_flag = True
    else:
        m.qvg_sparsify = True
        m.qvg_sparse_ratio = ratio
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
            fastar_full=False,
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
    return imgs if num_samples <= 0 else imgs[:num_samples]


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def save_plot(args, ratios, gammas, agg_by_key, worldkv_agg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print(f"跳过绘图（matplotlib 不可用）：{e}")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for g in gammas:
        ys = [agg_by_key[(r, g)]["lpips_mean"] for r in ratios]
        ys = [y if y is not None else float("nan") for y in ys]
        errs = [agg_by_key[(r, g)]["lpips_std"] if agg_by_key[(r, g)]["lpips_std"] is not None else 0.0
                for r in ratios]
        ax.errorbar(ratios, ys, yerr=errs, marker="o", capsize=4, label=f"gamma={g:g}")
    if worldkv_agg and worldkv_agg["lpips_mean"] is not None:
        ax.axhline(worldkv_agg["lpips_mean"], color="tab:orange", ls="--",
                   label=f"WorldKV {worldkv_agg['lpips_mean']:.4f}")
    ax.set_xlabel("ratio_base (tau)")
    ax.set_ylabel("mean revisit LPIPS (AlexNet, lower is better)")
    ax.set_title(f"QuanEvict vs tau ({args.max_num_output_frames} latent frames)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_png = os.path.join(args.output_folder, "quanevict_tau_scan.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"曲线已保存到 {out_png}")


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    logger = Logger(os.path.join(args.output_folder, "evaluation.log"))

    ratios = [float(x) for x in args.ratio_base_list.split(",") if x.strip() != ""]
    gammas = [float(x) for x in args.gamma_list.split(",") if x.strip() != ""]
    assert ratios and gammas, "ratio_base_list / gamma_list 不能为空"

    assert args.max_num_output_frames % 3 == 0, \
        f"max_num_output_frames 须为 3 的整数倍，当前 {args.max_num_output_frames}"
    num_blocks = args.max_num_output_frames // 3
    if args.max_num_output_frames != 50:
        logger.log(f"注意：请求 50 latent 帧，实际使用 {args.max_num_output_frames} "
                   f"（50 非 3 的倍数，取 >=50 最近可行值；{num_blocks} blocks）")

    imgs = select_images(args.img_dir, args.num_samples)
    assert imgs, f"未在 {args.img_dir} 找到 PNG"
    logger.log(f"测试图（{len(imgs)} 张）：{imgs}")
    logger.log(f"ratio_base 列表：{ratios}  gamma 列表：{gammas}  "
               f"skip_worldkv={args.skip_worldkv}")

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
        except Exception as e:
            logger.log(f"[{img_name}] build_inputs 失败，跳过整图：{e}")
            logger.log(traceback.format_exc())
            per_run.append({"img": img_name, "method": "build_inputs", "status": "failed",
                            "stage": "build_inputs", "error": str(e)})
            continue

        def _record(key, method_key, ratio, gamma, video, bank, peak, lat):
            mean_lp, per_pair = compute_mean_lpips(lpips_fn, video, revisit_pairs, game.device)
            if mean_lp is None:
                raise RuntimeError("无可计算的回访对")
            per_run.append({"img": img_name, "key": key, "ratio": ratio, "gamma": gamma,
                            "status": "ok", "mean_lpips": mean_lp, "per_pair": per_pair,
                            "bank_mb": bank / 1e6, "peak_gpu_mb": peak / 1e6,
                            "latency_sec": lat,
                            "throughput_fps": len(video) / lat if lat > 0 else None})
            logger.log(f"  [{method_key:24s}] lpips={mean_lp:.4f}  pairs={len(per_pair)}  "
                       f"bank={bank / 1e6:.1f}MB  peak={peak / 1e9:.2f}GiB  lat={lat:.2f}s")

        if not args.skip_worldkv:
            video = None
            try:
                name = f"{os.path.splitext(img_name)[0]}_worldkv"
                video, bank, peak, lat = run_one(game, args, cond, noise, actions,
                                                 None, None, worldkv=True, name=name)
                _record("worldkv", "worldkv", None, None, video, bank, peak, lat)
            except Exception as e:
                logger.log(f"  [worldkv] 失败：{e}")
                logger.log(traceback.format_exc())
                per_run.append({"img": img_name, "key": "worldkv", "status": "failed",
                                "stage": "inference", "error": str(e)})
            finally:
                if video is not None:
                    del video
                gc.collect()
                torch.cuda.empty_cache()

        for ratio in ratios:
            for gamma in gammas:
                key = f"r{ratio:g}_g{gamma:g}"
                name = f"{os.path.splitext(img_name)[0]}_{key}"
                video = None
                try:
                    video, bank, peak, lat = run_one(game, args, cond, noise, actions,
                                                     ratio, gamma, worldkv=False, name=name)
                    _record(key, f"r={ratio:g} g={gamma:g}", ratio, gamma, video, bank, peak, lat)
                except Exception as e:
                    logger.log(f"  [{key}] 失败：{e}")
                    logger.log(traceback.format_exc())
                    per_run.append({"img": img_name, "key": key, "ratio": ratio, "gamma": gamma,
                                    "status": "failed", "stage": "inference", "error": str(e)})
                finally:
                    if video is not None:
                        del video
                    gc.collect()
                    torch.cuda.empty_cache()

    def _agg(key_entries):
        ok = [r for r in key_entries if r["status"] == "ok"]
        lp_mean, lp_std = _mean_std([r["mean_lpips"] for r in ok])
        lat_vals = [r["latency_sec"] for r in ok]
        bank_vals = [r["bank_mb"] for r in ok]
        peak_vals = [r["peak_gpu_mb"] for r in ok]
        return {
            "n_total": len(key_entries),
            "n_success": len(ok),
            "lpips_mean": lp_mean, "lpips_std": lp_std,
            "lpips_all": [r["mean_lpips"] for r in ok],
            "latency_mean": float(np.mean(lat_vals)) if lat_vals else None,
            "bank_mb_mean": float(np.mean(bank_vals)) if bank_vals else None,
            "peak_gpu_mb_mean": float(np.mean(peak_vals)) if peak_vals else None,
        }

    worldkv_agg = None
    if not args.skip_worldkv:
        worldkv_agg = _agg([r for r in per_run if r.get("key") == "worldkv"])
    agg_by_key = {}
    for ratio in ratios:
        for gamma in gammas:
            key = f"r{ratio:g}_g{gamma:g}"
            agg_by_key[(ratio, gamma)] = _agg([r for r in per_run if r.get("key") == key])

    logger.log("\n===== 跨图平均 =====")
    if worldkv_agg and worldkv_agg["lpips_mean"] is not None:
        logger.log(f"  worldkv: lpips={worldkv_agg['lpips_mean']:.4f} ± {worldkv_agg['lpips_std']:.4f}")
    logger.log(f"{'ratio':>6} | {'gamma':>5} | {'LPIPS':>8} | {'latency s':>10} | {'bank MB':>9} | {'peak GiB':>9} | n")
    for ratio in ratios:
        for gamma in gammas:
            a = agg_by_key[(ratio, gamma)]
            lp = f"{a['lpips_mean']:.4f}" if a['lpips_mean'] is not None else "N/A"
            lat = f"{a['latency_mean']:.1f}" if a['latency_mean'] is not None else "N/A"
            bank = f"{a['bank_mb_mean']:.1f}" if a['bank_mb_mean'] is not None else "N/A"
            peak = f"{a['peak_gpu_mb_mean'] / 1024:.2f}" if a['peak_gpu_mb_mean'] is not None else "N/A"
            logger.log(f"{ratio:>6g} | {gamma:>5g} | {lp:>8} | {lat:>10} | {bank:>9} | {peak:>9} | "
                       f"{a['n_success']}/{a['n_total']}")

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
        "ratios": ratios,
        "gammas": gammas,
        "worldkv": worldkv_agg,
        "quanevict": {f"r{r:g}_g{g:g}": agg_by_key[(r, g)] for r in ratios for g in gammas},
        "per_run": per_run,
    }
    out_json = os.path.join(args.output_folder, "results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.log(f"结果 JSON 已保存到 {out_json}")

    if args.plot:
        save_plot(args, ratios, gammas, agg_by_key, worldkv_agg)

    logger.close()


if __name__ == "__main__":
    main()

"""QuanEvict QVG 压缩超参扫描（num_stages / kmeans_iters / num_centroids）。

固定 sparse_ratio=0.5、gamma=1.0、group_size=64、bits=2，其余按配置变化。
对比不同 QVG 压缩超参下的回访帧 LPIPS(AlexNet)、延迟、bank 字节、峰值显存，
找出延迟/质量的 Pareto 前沿。默认禁用 mp4 编码。
"""
import gc
import json
import os
import time
import traceback

import numpy as np
import torch

import evaluate_quanevict_tau as eq  # 复用 run_one / LPIPS / 轨迹 / Logger / 输入构建
from inference_streaming import InteractiveGameInference  # noqa: E402
from utils.misc import set_seed  # noqa: E402

# 扫描配置：num_stages(S) / kmeans_iters(I) / num_centroids(K)。第一个为当前基线。
CONFIGS = [
    ("S3_I10_K256", dict(num_stages=3, kmeans_iters=10, num_centroids=256)),
    ("S1_I5_K256", dict(num_stages=1, kmeans_iters=5, num_centroids=256)),
    ("S1_I5_K128", dict(num_stages=1, kmeans_iters=5, num_centroids=128)),
]


def main():
    args = eq.parse_args()
    args.output_folder = args.output_folder if args.output_folder != "outputs_quanevict_tau_scan" else "outputs_qvg_hp_scan"
    os.makedirs(args.output_folder, exist_ok=True)
    logger = eq.Logger(os.path.join(args.output_folder, "evaluation.log"))

    assert args.max_num_output_frames % 3 == 0, \
        f"max_num_output_frames 须为 3 的整数倍，当前 {args.max_num_output_frames}"
    num_blocks = args.max_num_output_frames // 3

    imgs = eq.select_images(args.img_dir, args.num_samples)
    assert imgs, f"未在 {args.img_dir} 找到 PNG"
    logger.log(f"测试图（{len(imgs)} 张）：{imgs}")
    logger.log(f"配置列表：{[k for k, _ in CONFIGS]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions, revisit_pairs, shape = eq.generate_trajectory(
        num_blocks, device, seed=args.seed, shape=args.shape)
    logger.log(f"轨迹形态={shape}，blocks={num_blocks}，回访对={revisit_pairs}")
    if not revisit_pairs:
        logger.log("警告：未产生任何回访对，LPIPS 将无法计算")

    set_seed(args.seed)
    game = InteractiveGameInference(args)
    mode = game.config.pop("mode")
    assert mode == "universal"
    m = game.pipeline.generator.model

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(game.device).eval()

    per_run = []
    for i, img in enumerate(imgs):
        img_name = os.path.basename(img)
        logger.log(f"--- 图 {i + 1}/{len(imgs)}: {img_name} ---")
        try:
            cond, noise = eq.build_inputs(game, args, mode, img_path=img, keep_vae=True)
        except Exception as e:
            logger.log(f"[{img_name}] build_inputs 失败，跳过整图：{e}")
            logger.log(traceback.format_exc())
            continue

        for key, cfg in CONFIGS:
            m.qvg_num_stages = cfg["num_stages"]
            m.qvg_kmeans_iters = cfg["kmeans_iters"]
            m.qvg_num_centroids = cfg["num_centroids"]

            name = f"{os.path.splitext(img_name)[0]}_{key}"
            video = None
            try:
                video, bank, peak, lat = eq.run_one(
                    game, args, cond, noise, actions,
                    args.qvg_sparse_ratio, args.qvg_gamma, worldkv=False, name=name)
                mean_lp, per_pair = eq.compute_mean_lpips(lpips_fn, video, revisit_pairs, game.device)
                if mean_lp is None:
                    raise RuntimeError("无可计算的回访对")
                per_run.append({"img": img_name, "key": key, "status": "ok",
                                "mean_lpips": mean_lp, "bank_mb": bank / 1e6,
                                "peak_gpu_mb": peak / 1e6, "latency_sec": lat})
                logger.log(f"  [{key:12s}] lpips={mean_lp:.4f}  pairs={len(per_pair)}  "
                           f"bank={bank / 1e6:.1f}MB  peak={peak / 1e9:.2f}GiB  lat={lat:.2f}s")
            except Exception as e:
                logger.log(f"  [{key:12s}] 失败：{e}")
                logger.log(traceback.format_exc())
                per_run.append({"img": img_name, "key": key, "status": "failed",
                                "stage": "inference", "error": str(e)})
            finally:
                if video is not None:
                    del video
                gc.collect()
                torch.cuda.empty_cache()

    agg = {}
    for key, cfg in CONFIGS:
        ok = [r for r in per_run if r["key"] == key and r["status"] == "ok"]
        lp_mean, lp_std = eq._mean_std([r["mean_lpips"] for r in ok])
        lat_vals = [r["latency_sec"] for r in ok]
        bank_vals = [r["bank_mb"] for r in ok]
        peak_vals = [r["peak_gpu_mb"] for r in ok]
        agg[key] = {
            "num_stages": cfg["num_stages"],
            "kmeans_iters": cfg["kmeans_iters"],
            "num_centroids": cfg["num_centroids"],
            "n_total": len(ok) + len([r for r in per_run if r["key"] == key and r["status"] == "failed"]),
            "n_success": len(ok),
            "lpips_mean": lp_mean, "lpips_std": lp_std,
            "lpips_all": [r["mean_lpips"] for r in ok],
            "latency_mean": float(np.mean(lat_vals)) if lat_vals else None,
            "bank_mb_mean": float(np.mean(bank_vals)) if bank_vals else None,
            "peak_gpu_mb_mean": float(np.mean(peak_vals)) if peak_vals else None,
        }

    logger.log("\n===== 跨图平均 =====")
    logger.log(f"{'key':>12} | {'S':>2} {'I':>2} {'K':>4} | {'LPIPS':>8} | {'latency s':>10} | {'bank MB':>9} | {'peak GiB':>9} | n")
    for key, cfg in CONFIGS:
        a = agg[key]
        lp = f"{a['lpips_mean']:.4f}" if a['lpips_mean'] is not None else "N/A"
        lat = f"{a['latency_mean']:.1f}" if a['latency_mean'] is not None else "N/A"
        bank = f"{a['bank_mb_mean']:.1f}" if a['bank_mb_mean'] is not None else "N/A"
        peak = f"{a['peak_gpu_mb_mean'] / 1024:.2f}" if a['peak_gpu_mb_mean'] is not None else "N/A"
        logger.log(f"{key:>12} | {cfg['num_stages']:>2} {cfg['kmeans_iters']:>2} {cfg['num_centroids']:>4} | "
                   f"{lp:>8} | {lat:>10} | {bank:>9} | {peak:>9} | {a['n_success']}/{a['n_total']}")

    summary = {
        "spec": {
            "requested_frames": 50,
            "actual_latent_frames": args.max_num_output_frames,
            "num_blocks": num_blocks,
            "rgb_per_block": 12,
            "lpips_net": "alex",
            "sparse_ratio": args.qvg_sparse_ratio,
            "gamma": args.qvg_gamma,
            "group_size": args.qvg_group_size,
            "bits": args.qvg_bits,
        },
        "seed": args.seed,
        "trajectory_shape": shape,
        "revisit_pairs": [[a, b] for a, b in revisit_pairs],
        "imgs": [os.path.basename(x) for x in imgs],
        "configs": {key: cfg for key, cfg in CONFIGS},
        "agg": agg,
        "per_run": per_run,
    }
    out_json = os.path.join(args.output_folder, "results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.log(f"结果 JSON 已保存到 {out_json}")
    logger.close()


if __name__ == "__main__":
    main()

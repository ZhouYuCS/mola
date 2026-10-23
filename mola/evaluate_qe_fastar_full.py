"""QuanEvict vs FAST-AR(full) vs full-kv 综合评测。

方法：
  - full_kv     : 基线（不量化、不轨迹检索、不压缩，滚动窗口 local_attn_size=12），仅作质量参考，不测速度。
  - fastar_full : FAST-AR 忠实 TempCache（跨位置匹配，thr=0.7）。
  - quanevict   : QuanEvict（带敏感度 gamma=1，默认 sparsify=True + kv_bank_on_gpu=True）。
  - worldkv     : WorldKV 位姿稀疏轨迹检索 + key-key 压缩（top-4 帧，原始 K/V 不量化，bank 放 CPU）。

帧数：10 张测试图，请求 100 latent 帧（管线硬约束须为 3 的整数倍，实际 102 = 34 blocks）。

指标：
  1) 回访帧对（首次访问块末帧 vs 回访块末帧）：LPIPS / PSNR / SSIM / FID（三方法各算）。
  2) 相同编号全帧 vs full_kv：LPIPS / PSNR / SSIM / FID（quanevict、fastar_full）。
  3) 速度/显存：quanevict、fastar_full（full_kv 不测）。

产物：results.json + 评价报告.md + evaluation.log。
"""
import argparse
import gc
import glob
import json
import math
import os
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

# ---- 在导入 inference_streaming 之前决定是否保存 mp4 ----
import sys

import utils.visualize as viz


def _noop_process_video(*args, **kwargs):
    return None


if "--save_video" not in sys.argv:
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
    {"key": "full_kv",     "label": "full-kv (baseline)",          "mode": "full_kv",     "fastar_sim_threshold": None},
    {"key": "fastar_full", "label": "FAST-AR (TempCache, thr=0.8)", "mode": "fastar_full", "fastar_sim_threshold": 0.8},
    {"key": "quanevict",   "label": "QuanEvict", "mode": "quanevict", "fastar_sim_threshold": None},
    {"key": "quanevict_plus", "label": "QuanEvict+", "mode": "quanevict_plus", "fastar_sim_threshold": None},
    {"key": "worldkv",     "label": "WorldKV (位姿检索 + key-key 压缩)",  "mode": "worldkv",     "fastar_sim_threshold": None},
    {"key": "flowcache",   "label": "FlowCache (importance-redundancy KV 驱逐)", "mode": "flowcache", "fastar_sim_threshold": None},
]
METHOD_BY_KEY = {m["key"]: m for m in METHODS}


def _effective_methods(args):
    """根据 --methods 与 --fastar_thresholds 组装最终方法列表。

    fastar_full 在 --fastar_thresholds 非空时展开为每个阈值一个方法，便于一次扫描。
    """
    keys = [k.strip() for k in args.methods.split(",") if k.strip()]
    methods = []
    for k in keys:
        if k == "fastar_full" and getattr(args, "fastar_thresholds", ""):
            thrs = [float(t) for t in args.fastar_thresholds.split(",") if t.strip()]
            assert thrs, "fastar_thresholds 为空"
            for t in thrs:
                nk = f"fastar_full@{t:g}"
                methods.append({
                    "key": nk,
                    "label": f"FAST-AR (TempCache, thr={t:g})",
                    "mode": "fastar_full",
                    "fastar_sim_threshold": t,
                })
        else:
            assert k in METHOD_BY_KEY, f"未知方法 key: {k}（可选 {list(METHOD_BY_KEY)}）"
            methods.append(METHOD_BY_KEY[k])
    return methods


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml")
    p.add_argument("--checkpoint_path", type=str,
                   default="/mnt/sda/zy/models/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors")
    p.add_argument("--pretrained_model_path", type=str, default="/mnt/sda/zy/models/Matrix-Game-2.0")
    p.add_argument("--img_dir", type=str, default="demo_images/universal")
    p.add_argument("--num_samples", type=int, default=10, help="测试图数量（<=0 取全部 PNG）")
    p.add_argument("--output_folder", type=str, default="outputs_qe_fastar_full_eval")
    p.add_argument("--max_num_output_frames", type=int, default=102,
                   help="latent 帧数（须为 3 的整数倍；请求 100，默认取 102）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape", type=str, default=None,
                   help="轨迹形态：rectangle/zigzag/rotate/strafe（默认随机）")
    p.add_argument("--dry_run", action="store_true", help="只验证轨迹与图片选择，不加载模型")

    p.add_argument("--retrieval_topk_frames", type=int, default=4)
    p.add_argument("--quant_chunk_frames", type=int, default=12)

    p.add_argument("--qvg_gamma", type=float, default=1.0)
    p.add_argument("--qvg_sensitivity_wk", type=str, default=SENS_WK)
    p.add_argument("--qvg_sensitivity_wv", type=str, default=SENS_WV)
    p.add_argument("--batch_size", type=int, default=1,
                   help="并行批大小；>1 时同一输入重复到该 batch size 运行")

    p.add_argument("--qvg_bits", type=int, default=2)
    p.add_argument("--qvg_num_centroids", type=int, default=256)
    p.add_argument("--qvg_num_stages", type=int, default=1)
    p.add_argument("--qvg_group_size", type=int, default=64)
    p.add_argument("--qvg_kmeans_iters", type=int, default=4)
    p.add_argument("--qvg_seed", type=int, default=0)
    p.add_argument("--qvg_sparsify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qvg_sparse_ratio", type=float, default=0.5,
                   help="QuanEvict 基准保留比例（每 head 由敏感度自适应缩放）")
    p.add_argument("--qvg_use_triton", action="store_true", default=True)

    p.add_argument("--kv_bank_on_gpu", action=argparse.BooleanOptionalAction, default=True,
                   help="KV bank 放 GPU（默认 True）")
    p.add_argument("--cache_transfer", action=argparse.BooleanOptionalAction, default=False,
                   help="将 KV cache 统一搬迁到 CPU，适用于 OOM 保护；False 时保持 GPU 全程运行")
    p.add_argument("--fastar_sim_threshold", type=float, default=0.8)
    p.add_argument("--fastar_lsh_tables", type=int, default=1,
                   help="fastar_full 的 LSH 哈希表数量（SimHash 多表探测；0 表示关闭 LSH，退回精确暴力 top-1）")
    p.add_argument("--fastar_lsh_bits", type=int, default=10,
                   help="fastar_full 的 LSH 每张表哈希位数（SimHash sign random projection 桶编码）")
    p.add_argument("--fastar_lsh_seed", type=int, default=0,
                   help="fastar_full 的 LSH 随机投影矩阵种子")
    p.add_argument("--fastar_lsh_bucket_cap", type=int, default=128,
                   help="FAST-AR LSH 候选桶过滤上限；越小越省显存，默认 128（过大桶会触发中间张量 OOM）")
    p.add_argument("--flowcache_budget", type=int, default=10000,
                   help="FlowCache KV 驱逐 bank 的 token 预算（importance-redundancy 打分后保留数）")
    p.add_argument("--flowcache_lambda", type=float, default=0.07,
                   help="FlowCache 打分权重 λ（importance 权重，1-λ 为 redundancy 权重）")
    p.add_argument("--flowcache_maxpool_kernel", type=int, default=3,
                   help="FlowCache importance 的 MaxPool1D 平滑核大小")
    p.add_argument("--worldkv_compress", action=argparse.BooleanOptionalAction, default=True,
                   help="WorldKV 是否开启 World Compression（key-key 相似度驱逐 non-anchor token）")
    p.add_argument("--worldkv_retain_ratio", type=float, default=0.25,
                   help="WorldKV 压缩时 non-anchor token 保留比例（论文 P=25%，chunk 3T->1.5T）")
    p.add_argument("--methods", type=str,
                   default="full_kv,fastar_full,quanevict,quanevict_plus,worldkv,flowcache",
                   help="逗号分隔的方法 key 列表（full_kv / fastar_full / quanevict / quanevict_plus / worldkv / flowcache）")
    p.add_argument("--fastar_thresholds", type=str, default="0.8",
                   help="fastar_full 的阈值列表（逗号分隔，如 0.8,0.9,0.95）；非空时把 fastar_full 展开为多个阈值方法")
    p.add_argument("--skip_full_kv", action="store_true",
                   help="跳过 full_kv 基线运行（只测各方法端到端延迟 WAN/VAE，不计算 all_frame 相对指标）")
    p.add_argument("--save_video", action="store_true",
                   help="保存推理视频（默认用 noop 替换 process_video 以节省时间/磁盘）")
    p.add_argument("--fid_batch_size", type=int, default=64)
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


# ============================ 指标实现 ============================

def psnr_uint8(a, b):
    """a, b: uint8 [H,W,C] numpy。返回 PSNR(dB)。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _gaussian_window(window_size, channels):
    def gaussian(w, sigma):
        g = torch.tensor(
            [math.exp(-(x - w // 2) ** 2 / (2.0 * sigma ** 2)) for x in range(w)],
            dtype=torch.float32)
        return g / g.sum()

    g1 = gaussian(window_size, 1.5).unsqueeze(1)
    g2 = g1 @ g1.t()
    return g2.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()


def ssim_batch(ta, tb):
    """ta, tb: [B, C, H, W] float，范围 [0,1]。返回每图 SSIM [B]。"""
    C = ta.size(1)
    window = _gaussian_window(11, C).to(ta.device)
    pad = 11 // 2
    mu1 = F.conv2d(ta, window, padding=pad, groups=C)
    mu2 = F.conv2d(tb, window, padding=pad, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(ta * ta, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(tb * tb, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(ta * tb, window, padding=pad, groups=C) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=(1, 2, 3))


def _uint8_to_01_batch(frames, device):
    """frames: uint8 [N,H,W,C] -> [N,C,H,W] float [0,1] on device."""
    t = torch.from_numpy(np.ascontiguousarray(frames)).to(device).permute(0, 3, 1, 2).float()
    return t / 255.0


def _uint8_to_lpips_batch(frames, device):
    """frames: uint8 [N,H,W,C] -> [N,C,H,W] float [-1,1] on device."""
    t = _uint8_to_01_batch(frames, device)
    return t * 2.0 - 1.0


class FIDComputer:
    """用 torchvision InceptionV3 的 pool3(2048) 特征计算 Frechet Distance。"""

    def __init__(self, device):
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.device = device
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(self, frames, batch_size=64):
        """frames: uint8 [N,H,W,C] numpy。返回 [N,2048] float64 numpy。"""
        feats = []
        for i in range(0, len(frames), batch_size):
            x = _uint8_to_01_batch(frames[i:i + batch_size], self.device)  # [B,3,H,W]
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            out = self.model(x)
            feats.append(out.detach().cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float64)

    @staticmethod
    def fid(feats1, feats2):
        from scipy.linalg import sqrtm
        mu1 = feats1.mean(axis=0)
        mu2 = feats2.mean(axis=0)
        sig1 = np.cov(feats1, rowvar=False)
        sig2 = np.cov(feats2, rowvar=False)
        d = sig1.shape[0]
        eps = 1e-6 * np.eye(d)
        sig1 = sig1 + eps
        sig2 = sig2 + eps
        diff = mu1 - mu2
        covmean = sqrtm(sig1 @ sig2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean))


# ============================ 推理 ============================

def run_one(game, args, cond, noise, actions, method, name):
    m = game.pipeline.generator.model
    mode = method["mode"]
    use_retrieval = False
    worldkv_flag = False
    fastar_full_flag = False
    flowcache_flag = False
    full_kv_flag = False
    kv_bank_on_gpu = args.kv_bank_on_gpu and not args.cache_transfer
    if mode == "full_kv":
        m.qvg_sparsify = False
        full_kv_flag = True
    elif mode == "fastar_full":
        m.qvg_sparsify = False
        fastar_full_flag = True
    elif mode == "worldkv":
        # WorldKV：位姿稀疏轨迹检索 + key-key 相似度压缩（World Compression），
        # bank 逐帧存原始 K/V（不量化）；cache_transfer=True 时强制 bank 留在 CPU 以降 OOM。
        m.qvg_sparsify = False
        worldkv_flag = True
        kv_bank_on_gpu = False
    elif mode == "flowcache":
        m.qvg_sparsify = False
        flowcache_flag = True
    else:  # quanevict / quanevict_plus
        m.qvg_sparsify = True
        m.qvg_sparse_ratio = args.qvg_sparse_ratio
        m.qvg_gamma = args.qvg_gamma
        if mode == "quanevict_plus":
            m.qvg_sensitivity_wk = args.qvg_sensitivity_wk
            m.qvg_sensitivity_wv = args.qvg_sensitivity_wv
        else:
            m.qvg_sensitivity_wk = None
            m.qvg_sensitivity_wv = None
        use_retrieval = True

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
            kv_bank_on_gpu=kv_bank_on_gpu,
            quant_chunk_frames=args.quant_chunk_frames,
            worldkv=worldkv_flag,
            worldkv_compress=args.worldkv_compress,
            worldkv_retain_ratio=args.worldkv_retain_ratio,
            fastar_full=fastar_full_flag,
            fastar_sim_threshold=method.get("fastar_sim_threshold") or args.fastar_sim_threshold,
            fastar_lsh_tables=args.fastar_lsh_tables,
            fastar_lsh_bits=args.fastar_lsh_bits,
            fastar_lsh_seed=args.fastar_lsh_seed,
            fastar_lsh_bucket_cap=args.fastar_lsh_bucket_cap,
            residual_mag_sink=None,
            flowcache=flowcache_flag,
            flowcache_budget=args.flowcache_budget,
            flowcache_lambda=args.flowcache_lambda,
            flowcache_maxpool_kernel=args.flowcache_maxpool_kernel,
            full_kv=full_kv_flag,
            cache_transfer=args.cache_transfer,
        )
    wan_latency = getattr(game.pipeline, "_last_wan_latency", time.perf_counter() - t0)
    vae_latency = getattr(game.pipeline, "_last_vae_latency", 0.0)
    peak_gpu = torch.cuda.max_memory_allocated()
    bank_bytes = getattr(game.pipeline, "_last_bank_bytes", 0)
    return video, bank_bytes, peak_gpu, wan_latency, vae_latency


def compute_revisit_metrics(lpips_fn, video, revisit_pairs, device):
    """逐回访对计算 LPIPS/PSNR/SSIM，返回 (mean_dict, per_pair)。"""
    rgb_per_block = 12
    per_pair = []
    for ref_c, rev_c in revisit_pairs:
        ref_idx = ref_c * rgb_per_block + 8
        rev_idx = rev_c * rgb_per_block + 8
        ref = video[ref_idx]
        rev = video[rev_idx]
        lp = lpips_fn(
            frame_to_lpips_input(ref, device),
            frame_to_lpips_input(rev, device),
        ).item()
        ps = psnr_uint8(ref, rev)
        ss = float(ssim_batch(
            _uint8_to_01_batch(ref[None], device),
            _uint8_to_01_batch(rev[None], device),
        ).item())
        per_pair.append({
            "ref_block": ref_c, "revisit_block": rev_c,
            "ref_rgb_idx": ref_idx, "revisit_rgb_idx": rev_idx,
            "lpips": lp, "psnr": ps, "ssim": ss,
        })
    if not per_pair:
        return None, []
    mean = {
        "lpips": float(np.mean([x["lpips"] for x in per_pair])),
        "psnr": float(np.mean([x["psnr"] for x in per_pair])),
        "ssim": float(np.mean([x["ssim"] for x in per_pair])),
    }
    return mean, per_pair


def compute_all_frame_metrics(lpips_fn, video_m, video_full, device, chunk=64):
    """相同编号全帧对比：返回逐帧 LPIPS/PSNR/SSIM 列表。"""
    T = len(video_m)
    lpips_vals = []
    psnr_vals = []
    ssim_vals = []
    for i in range(0, T, chunk):
        a = video_m[i:i + chunk]
        b = video_full[i:i + chunk]
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2, axis=(1, 2, 3))
        # 完全相同的帧（mse=0）PSNR 截断为 100 dB，避免 inf 污染逐帧均值
        psnr_vals.extend([100.0 if m == 0 else 20.0 * np.log10(255.0 / np.sqrt(m)) for m in mse])
        lp = lpips_fn(
            _uint8_to_lpips_batch(a, device),
            _uint8_to_lpips_batch(b, device),
        ).reshape(-1).detach().cpu().numpy()
        lpips_vals.extend([float(x) for x in lp])
        ss = ssim_batch(_uint8_to_01_batch(a, device), _uint8_to_01_batch(b, device)).detach().cpu().numpy()
        ssim_vals.extend([float(x) for x in ss])
    return {
        "lpips": lpips_vals,
        "psnr": psnr_vals,
        "ssim": ssim_vals,
    }


def select_images(img_dir, num_samples):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if num_samples <= 0:
        return imgs
    return imgs[:num_samples]


def _mean_std(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


# ============================ 报告 ============================

def build_report(args, agg, per_run, revisit_pairs, shape, imgs):
    L = []
    L.append("# QuanEvict vs FAST-AR(full) vs full-kv 综合评测报告")
    L.append("")
    L.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append("## 1. 目标")
    L.append("")
    L.append("在相同轨迹、相同噪声、相同测试图下，评测 QuanEvict（带敏感度）与 FAST-AR(full) "
             "的回访一致性（回访帧对 LPIPS/PSNR/SSIM/FID），并以 full-kv 为质量参考基线，"
             "度量两方法在所有相同编号帧上相对 full-kv 的 LPIPS/PSNR/SSIM/FID。")
    L.append("")
    L.append("## 2. 方法定义")
    L.append("")
    L.append("| 方法 | 说明 | 是否测速度 |")
    L.append("|---|---|---|")
    for m in METHODS:
        speed = "否" if m["mode"] == "full_kv" else "是"
        L.append(f"| {m['label']} | key=`{m['key']}` | {speed} |")
    L.append("")
    L.append("- **full-kv**：不量化、不轨迹检索、不压缩，滚动窗口 `local_attn_size=12`（sink 3 + recent 9）。")
    L.append("- **fastar_full**：FAST-AR 忠实 TempCache，跨空间位置 top-1 匹配，阈值 0.9；用 SimHash LSH（多表探测）加速 top-1 近邻检索。")
    L.append("- **QuanEvict**：QVG 压缩 + 残差稀疏化（`sparsify=True`），无敏感度加权；相当于统一保留比例。")
    L.append("- **QuanEvict+**：QVG 压缩 + 残差稀疏化 + per-(layer,head) 敏感度自适应比例（γ=1）。")
    L.append("- **worldkv**：位姿稀疏轨迹检索（translation/rotation 分离归一化，top-4 帧）+ key-key 相似度压缩（anchor 帧全保留，non-anchor 帧按 Eq.5 保留最低相似度 25% token，chunk 3T→1.5T），bank 逐帧存原始 K/V 不量化（放 CPU）。")
    L.append("- **flowcache**：importance-redundancy 打分的 KV 驱逐，固定预算内选 top-B token（贴 FlowCache 论文 Sec.3.3）。")
    L.append("")
    L.append("## 3. 测试环境")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| GPU | RTX 4090 24GB (CUDA_VISIBLE_DEVICES=1) |")
    L.append(f"| PyTorch | {torch.__version__} |")
    L.append(f"| checkpoint | {args.checkpoint_path} |")
    L.append(f"| 配置 | {args.config_path} |")
    L.append(f"| seed | {args.seed} |")
    L.append(f"| kv_bank_on_gpu | {args.kv_bank_on_gpu} |")
    L.append(f"| cache_transfer | {args.cache_transfer} |")
    L.append(f"| batch_size | {args.batch_size} |")
    L.append(f"| qvg_sparsify | {args.qvg_sparsify} |")
    L.append(f"| qvg_kmeans_iters | {args.qvg_kmeans_iters} |")
    L.append(f"| qvg_num_stages | {args.qvg_num_stages} |")
    L.append(f"| qvg_sparse_ratio | {args.qvg_sparse_ratio} |")
    L.append("")
    L.append("## 4. 轨迹与帧数")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 请求总帧数 | 100 latent 帧 |")
    L.append(f"| 实际使用 | {args.max_num_output_frames} latent 帧（100 非 3 的倍数，取 >=100 最近可行值 = 34 blocks） |")
    L.append("| 块/帧粒度 | 1 block = 3 latent 帧 = 12 RGB 帧 |")
    L.append(f"| 轨迹形态（随机） | {shape} |")
    L.append(f"| 回访 block 对 | {revisit_pairs} |")
    L.append("")
    L.append("回访帧 RGB 索引 = `block*12 + 8`（该 block 末 latent 帧）。")
    L.append("")
    L.append("## 5. 测试图像")
    L.append("")
    for i, im in enumerate(imgs):
        L.append(f"{i + 1}. `{os.path.basename(im)}`")
    L.append("")
    L.append("## 6. 指标计算方法")
    L.append("")
    L.append("- **LPIPS**：`lpips.LPIPS(net='alex')`，AlexNet 特征。")
    L.append("- **PSNR**：`20*log10(255/sqrt(MSE))`，逐 RGB 帧。")
    L.append("- **SSIM**：11×11 高斯窗（σ=1.5），逐 RGB 通道平均，逐帧标量。")
    L.append("- **FID**：torchvision InceptionV3（IMAGENET1K_V1）pool3(2048) 特征，Fréchet Distance；"
             "回访 FID 为“首次访问帧分布 vs 回访帧分布”，全帧 FID 为“方法全帧分布 vs full-kv 全帧分布”。")
    L.append("")
    L.append("> 注：回访帧数量较少（约百帧量级），FID 协方差为秩亏，回访 FID 仅作参考，稳定性弱于全帧 FID。")
    L.append("")
    L.append("## 7. 回访帧对指标（跨 10 图）")
    L.append("")
    L.append("| 方法 | 平均 LPIPS↓ | 平均 PSNR↑ | 平均 SSIM↑ | 回访 FID↓ | 成功样本 |")
    L.append("|---|---|---|---|---|---|")
    for m in METHODS:
        a = agg[m["key"]]["revisit"]
        lp = f"{a['lpips_mean']:.4f}±{a['lpips_std']:.4f}" if a["lpips_mean"] is not None else "N/A"
        ps = f"{a['psnr_mean']:.2f}±{a['psnr_std']:.2f}" if a["psnr_mean"] is not None else "N/A"
        ss = f"{a['ssim_mean']:.4f}±{a['ssim_std']:.4f}" if a["ssim_mean"] is not None else "N/A"
        fid = f"{a['fid']:.2f}" if a["fid"] is not None else "N/A"
        L.append(f"| {m['label']} | {lp} | {ps} | {ss} | {fid} | {a['n_success']}/{a['n_total']} |")
    L.append("")
    L.append("## 8. 全帧 vs full-kv 指标（跨 10 图，相同编号帧）")
    L.append("")
    L.append("| 方法 | 平均 LPIPS↓ | 平均 PSNR↑ | 平均 SSIM↑ | 全帧 FID↓ |")
    L.append("|---|---|---|---|---|")
    for m in METHODS:
        if m["mode"] == "full_kv":
            continue
        a = agg[m["key"]]["all_frame"]
        lp = f"{a['lpips_mean']:.4f}±{a['lpips_std']:.4f}" if a["lpips_mean"] is not None else "N/A"
        ps = f"{a['psnr_mean']:.2f}±{a['psnr_std']:.2f}" if a["psnr_mean"] is not None else "N/A"
        ss = f"{a['ssim_mean']:.4f}±{a['ssim_std']:.4f}" if a["ssim_mean"] is not None else "N/A"
        fid = f"{a['fid']:.2f}" if a["fid"] is not None else "N/A"
        L.append(f"| {m['label']} | {lp} | {ps} | {ss} | {fid} |")
    L.append("")
    L.append("## 9. 速度 / 显存（跨 10 图平均，含 full-kv）")
    L.append("")
    L.append("| 方法 | 平均 WAN latency(s) | 平均 VAE latency(s) | 平均 bank(MB) | 平均 peak GPU(GiB) |")
    L.append("|---|---|---|---|---|")
    for m in METHODS:
        s = agg[m["key"]]["speed"]
        wan = f"{s['wan_latency_mean']:.2f}" if s["wan_latency_mean"] is not None else "N/A"
        vae = f"{s['vae_latency_mean']:.2f}" if s["vae_latency_mean"] is not None else "N/A"
        bank = f"{s['bank_mb_mean']:.1f}" if s["bank_mb_mean"] is not None else "N/A"
        peak = f"{s['peak_gpu_mb_mean'] / 1024:.2f}" if s["peak_gpu_mb_mean"] is not None else "N/A"
        L.append(f"| {m['label']} | {wan} | {vae} | {bank} | {peak} |")
    L.append("")
    L.append("## 10. 复现命令")
    L.append("")
    L.append("```bash")
    L.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \\")
    L.append("  /home/zy/miniconda3/envs/llama_factory/bin/python evaluate_qe_fastar_full.py \\")
    L.append(f"    --num_samples {args.num_samples} --seed {args.seed} \\")
    L.append(f"    --max_num_output_frames {args.max_num_output_frames}")
    L.append("```")
    L.append("")
    L.append("## 11. 产物")
    L.append("")
    L.append("- `results.json`：完整逐 run / 逐回访对 / 逐帧结果。")
    L.append("- `evaluation.log`：完整运行日志。")
    L.append("")
    failures = [r for r in per_run if r["status"] == "failed"]
    if failures:
        L.append("### 失败 run 摘要")
        L.append("")
        for r in failures:
            L.append(f"- `{r['img']}` / `{r['method']}`: `{r.get('error', '')[:200]}`")
        L.append("")
    return "\n".join(L)


def main():
    args = parse_args()
    global METHODS, METHOD_BY_KEY
    METHODS = _effective_methods(args)
    METHOD_BY_KEY = {m["key"]: m for m in METHODS}

    os.makedirs(args.output_folder, exist_ok=True)
    log_path = os.path.join(args.output_folder, "evaluation.log")
    logger = Logger(log_path)

    assert args.max_num_output_frames % 3 == 0, \
        f"max_num_output_frames 须为 3 的整数倍，当前 {args.max_num_output_frames}"
    num_blocks = args.max_num_output_frames // 3

    imgs = select_images(args.img_dir, args.num_samples)
    assert imgs, f"未在 {args.img_dir} 找到 PNG"
    logger.log(f"测试图（{len(imgs)} 张）：{imgs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions, revisit_pairs, shape = generate_trajectory(num_blocks, device, seed=args.seed, shape=args.shape)
    logger.log(f"轨迹形态={shape}，blocks={num_blocks}，回访对={revisit_pairs}")
    if not revisit_pairs:
        logger.log("警告：未产生任何回访对，回访指标将无法计算")

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
    fid_computer = FIDComputer(game.device)

    revisit_ref_frames = {m["key"]: [] for m in METHODS}
    revisit_frames = {m["key"]: [] for m in METHODS}
    all_feats = {m["key"]: [] for m in METHODS}

    per_run = []

    for i, img in enumerate(imgs):
        img_name = os.path.basename(img)
        logger.log(f"--- 图 {i + 1}/{len(imgs)}: {img_name} ---")
        try:
            cond, noise = build_inputs(game, args, mode, img_path=img, keep_vae=True)
        except Exception as e:  # noqa: BLE001
            logger.log(f"[{img_name}] build_inputs 失败，跳过整图：{e}")
            logger.log(traceback.format_exc())
            for m in METHODS:
                per_run.append({"img": img_name, "method": m["key"], "status": "failed",
                                "stage": "build_inputs", "error": str(e)})
            continue

        video_full = None
        if not args.skip_full_kv:
            try:
                video_full, fbank, fpeak, fwan, fvae = run_one(game, args, cond, noise, actions, METHOD_BY_KEY["full_kv"],
                                              f"{os.path.splitext(img_name)[0]}_full_kv")
                r_mean, r_pairs = compute_revisit_metrics(lpips_fn, video_full, revisit_pairs, game.device)
                if r_mean is not None:
                    for pp in r_pairs:
                        revisit_ref_frames["full_kv"].append(video_full[pp["ref_rgb_idx"]])
                        revisit_frames["full_kv"].append(video_full[pp["revisit_rgb_idx"]])
                all_feats["full_kv"].append(fid_computer.features(video_full, args.fid_batch_size))
                per_run.append({"img": img_name, "method": "full_kv", "status": "ok",
                                "revisit": r_mean, "per_pair": r_pairs,
                                "speed": {"bank_mb": fbank / 1e6, "peak_gpu_mb": fpeak / 1e6,
                                          "wan_latency_sec": fwan, "vae_latency_sec": fvae}})
                rlp = f"{r_mean['lpips']:.4f}" if r_mean else "N/A"
                rps = f"{r_mean['psnr']:.2f}" if r_mean else "N/A"
                rss = f"{r_mean['ssim']:.4f}" if r_mean else "N/A"
                logger.log(f"  [full_kv] revisit_lpips={rlp} psnr={rps} ssim={rss} "
                           f"wan={fwan:.2f}s vae={fvae:.2f}s")
            except Exception as e:  # noqa: BLE001
                logger.log(f"  [full_kv] 失败：{e}")
                logger.log(traceback.format_exc())
                per_run.append({"img": img_name, "method": "full_kv", "status": "failed",
                                "stage": "inference", "error": str(e)})

        for m in METHODS:
            if m["mode"] == "full_kv":
                continue
            video_m = None
            try:
                video_m, bank, peak, wan_lat, vae_lat = run_one(game, args, cond, noise, actions, m,
                                                   f"{os.path.splitext(img_name)[0]}_{m['key']}")
                r_mean, r_pairs = compute_revisit_metrics(lpips_fn, video_m, revisit_pairs, game.device)
                if r_mean is not None:
                    for pp in r_pairs:
                        revisit_ref_frames[m["key"]].append(video_m[pp["ref_rgb_idx"]])
                        revisit_frames[m["key"]].append(video_m[pp["revisit_rgb_idx"]])
                if video_full is not None:
                    af = compute_all_frame_metrics(lpips_fn, video_m, video_full, game.device)
                    af_mean = {
                        "lpips": float(np.mean(af["lpips"])),
                        "psnr": float(np.mean(af["psnr"])),
                        "ssim": float(np.mean(af["ssim"])),
                    }
                    all_feats[m["key"]].append(fid_computer.features(video_m, args.fid_batch_size))
                else:
                    af_mean = None
                per_run.append({"img": img_name, "method": m["key"], "status": "ok",
                                "revisit": r_mean, "per_pair": r_pairs,
                                "all_frame": af_mean,
                                "speed": {"bank_mb": bank / 1e6, "peak_gpu_mb": peak / 1e6,
                                          "wan_latency_sec": wan_lat, "vae_latency_sec": vae_lat}})
                rlp = f"{r_mean['lpips']:.4f}" if r_mean else "N/A"
                rps = f"{r_mean['psnr']:.2f}" if r_mean else "N/A"
                alp = f"{af_mean['lpips']:.4f}" if af_mean else "N/A"
                logger.log(f"  [{m['key']:12s}] rev_lpips={rlp} rev_psnr={rps} "
                           f"all_lpips={alp} bank={bank / 1e6:.1f}MB peak={peak / 1e9:.2f}GiB "
                           f"wan={wan_lat:.2f}s vae={vae_lat:.2f}s")
            except Exception as e:  # noqa: BLE001
                logger.log(f"  [{m['key']:12s}] 失败：{e}")
                logger.log(traceback.format_exc())
                per_run.append({"img": img_name, "method": m["key"], "status": "failed",
                                "stage": "inference", "error": str(e)})
            finally:
                if video_m is not None:
                    del video_m
                gc.collect()
                torch.cuda.empty_cache()

        if video_full is not None:
            del video_full
        gc.collect()
        torch.cuda.empty_cache()

    # ============================ 聚合 ============================
    agg = {m["key"]: {} for m in METHODS}
    for m in METHODS:
        k = m["key"]
        ok = [r for r in per_run if r["method"] == k and r["status"] == "ok"]
        failed = [r for r in per_run if r["method"] == k and r["status"] == "failed"]

        rev = {
            "n_total": len(ok) + len(failed),
            "n_success": len(ok),
        }
        lp_vals = [r["revisit"]["lpips"] for r in ok if r.get("revisit")]
        ps_vals = [r["revisit"]["psnr"] for r in ok if r.get("revisit")]
        ss_vals = [r["revisit"]["ssim"] for r in ok if r.get("revisit")]
        rev["lpips_mean"], rev["lpips_std"] = _mean_std(lp_vals)
        rev["psnr_mean"], rev["psnr_std"] = _mean_std(ps_vals)
        rev["ssim_mean"], rev["ssim_std"] = _mean_std(ss_vals)
        if len(revisit_ref_frames[k]) > 1 and len(revisit_frames[k]) > 1:
            f1 = fid_computer.features(np.stack(revisit_ref_frames[k]), args.fid_batch_size)
            f2 = fid_computer.features(np.stack(revisit_frames[k]), args.fid_batch_size)
            rev["fid"] = FIDComputer.fid(f1, f2)
        else:
            rev["fid"] = None
        agg[k]["revisit"] = rev

        af = {"lpips_mean": None, "lpips_std": None, "psnr_mean": None, "psnr_std": None,
              "ssim_mean": None, "ssim_std": None, "fid": None}
        if k != "full_kv":
            af_lp = [r["all_frame"]["lpips"] for r in ok if r.get("all_frame")]
            af_ps = [r["all_frame"]["psnr"] for r in ok if r.get("all_frame")]
            af_ss = [r["all_frame"]["ssim"] for r in ok if r.get("all_frame")]
            af["lpips_mean"], af["lpips_std"] = _mean_std(af_lp)
            af["psnr_mean"], af["psnr_std"] = _mean_std(af_ps)
            af["ssim_mean"], af["ssim_std"] = _mean_std(af_ss)
            if len(all_feats[k]) > 0 and len(all_feats.get("full_kv", [])) > 0:
                fm = np.concatenate(all_feats[k], axis=0)
                ff = np.concatenate(all_feats["full_kv"], axis=0)
                af["fid"] = FIDComputer.fid(fm, ff)
        agg[k]["all_frame"] = af

        spd = {"wan_latency_mean": None, "vae_latency_mean": None,
               "bank_mb_mean": None, "peak_gpu_mb_mean": None}
        wan_vals = [r["speed"]["wan_latency_sec"] for r in ok if r.get("speed")]
        vae_vals = [r["speed"]["vae_latency_sec"] for r in ok if r.get("speed")]
        bank_vals = [r["speed"]["bank_mb"] for r in ok if r.get("speed")]
        peak_vals = [r["speed"]["peak_gpu_mb"] for r in ok if r.get("speed")]
        spd["wan_latency_mean"] = float(np.mean(wan_vals)) if wan_vals else None
        spd["vae_latency_mean"] = float(np.mean(vae_vals)) if vae_vals else None
        spd["bank_mb_mean"] = float(np.mean(bank_vals)) if bank_vals else None
        spd["peak_gpu_mb_mean"] = float(np.mean(peak_vals)) if peak_vals else None
        agg[k]["speed"] = spd

    logger.log("\n===== 回访帧对指标（跨图平均）=====")
    for m in METHODS:
        r = agg[m["key"]]["revisit"]
        logger.log(f"  {m['key']:12s}: lpips={r['lpips_mean']} psnr={r['psnr_mean']} "
                   f"ssim={r['ssim_mean']} fid={r['fid']} (n={r['n_success']}/{r['n_total']})")

    summary = {
        "spec": {
            "requested_frames": 100,
            "actual_latent_frames": args.max_num_output_frames,
            "note": "100 not divisible by 3; used nearest >=100 value",
            "num_blocks": num_blocks,
            "rgb_per_block": 12,
            "lpips_net": "alex",
            "fid_features": "inceptionv3_pool3_2048",
        },
        "env": {
            "torch": torch.__version__,
            "checkpoint_path": args.checkpoint_path,
            "pretrained_model_path": args.pretrained_model_path,
            "config_path": args.config_path,
            "kv_bank_on_gpu": args.kv_bank_on_gpu,
            "qvg_sparsify": args.qvg_sparsify,
            "qvg_kmeans_iters": args.qvg_kmeans_iters,
            "qvg_num_stages": args.qvg_num_stages,
            "qvg_sparse_ratio": args.qvg_sparse_ratio,
            "qvg_gamma": args.qvg_gamma,
        },
        "seed": args.seed,
        "trajectory_shape": shape,
        "revisit_pairs": [[a, b] for a, b in revisit_pairs],
        "imgs": [os.path.basename(x) for x in imgs],
        "methods": agg,
        "per_run": per_run,
    }
    out_json = os.path.join(args.output_folder, "results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.log(f"结果 JSON 已保存到 {out_json}")

    report = build_report(args, agg, per_run, revisit_pairs, shape, imgs)
    out_md = os.path.join(args.output_folder, "评价报告.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    logger.log(f"报告已保存到 {out_md}")

    logger.close()


if __name__ == "__main__":
    main()

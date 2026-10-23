"""QuanEvict(ratio=0.5) vs 纯 WorldKV 回访帧 LPIPS 对比。

- 两次运行使用相同初始图像、相同噪声、相同运动轨迹。
- 轨迹：四种真实形态（rectangle/zigzag/rotate/strafe，随机取一）+ 逆序回程；回访对基于 4 维模拟位姿（与 WorldKV 的 running_state 一致）。strafe 额外引入 A/D 左右平移。
- 回访帧 = 某个 chunk 的末 latent 帧（RGB index = chunk*12 + 8），其位姿与某个
  更早 chunk 的末 latent 帧位姿相同。
- 指标 = 回访帧与首次访问帧之间的 LPIPS(AlexNet)。

为避免单进程跑两遍导致的显存叠加/碎片化，支持分方法运行：
    python compare_quanevict_worldkv.py --method worldkv   # 只跑 WorldKV，存 worldkv_video.npy
    python compare_quanevict_worldkv.py --method quanevict # 只跑 QuanEvict，存 quanevict_video.npy
    python compare_quanevict_worldkv.py --method compare   # 不加载模型，只算 LPIPS
    python compare_quanevict_worldkv.py --method both      # 单进程连续跑两者（默认）
"""
import argparse
import gc
import json
import math
import os
import random

import numpy as np
import torch
from diffusers.utils import load_image

from inference_streaming import InteractiveGameInference
from utils.conditions import Bench_actions_universal
from utils.misc import set_seed

SENS_WK = "/home/zy/Matrix-Game/Matrix-Game-2/outputs/ratequant_sensitivity/sensitivity_wK.npy"
SENS_WV = "/home/zy/Matrix-Game/Matrix-Game-2/outputs/ratequant_sensitivity/sensitivity_wV.npy"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml")
    p.add_argument("--checkpoint_path", type=str, default="")
    p.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
    p.add_argument("--img_path", type=str, default="demo_images/universal/0000.png")
    p.add_argument("--output_folder", type=str, default="outputs_quanevict_vs_worldkv")
    p.add_argument("--max_num_output_frames", type=int, default=60, help="latent 帧数（须为 3 的整数倍）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", type=str, default="both",
                   choices=["both", "worldkv", "quanevict", "compare"])

    # 检索 / 压缩
    p.add_argument("--retrieval_topk_frames", type=int, default=4)
    p.add_argument("--quant_chunk_frames", type=int, default=12)

    # QuanEvict
    p.add_argument("--gamma", "--qvg_gamma", dest="qvg_gamma", type=float, default=1.0)
    p.add_argument("--qvg_sensitivity_wk", type=str, default=SENS_WK)
    p.add_argument("--qvg_sensitivity_wv", type=str, default=SENS_WV)

    # QVG 量化配置（InteractiveGameInference._init_models 会读取这些属性）
    p.add_argument("--qvg_bits", type=int, default=2)
    p.add_argument("--qvg_num_centroids", type=int, default=256)
    p.add_argument("--qvg_num_stages", type=int, default=1)
    p.add_argument("--qvg_group_size", type=int, default=64)
    p.add_argument("--qvg_kmeans_iters", type=int, default=4)
    p.add_argument("--qvg_seed", type=int, default=0)
    p.add_argument("--qvg_sparsify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qvg_sparse_ratio", type=float, default=0.5, help="QuanEvict 基准保留比例（每 head 由敏感度自适应缩放）")
    p.add_argument("--qvg_use_triton", action="store_true", default=True)
    return p.parse_args()


# ---- 运动轨迹生成（参照 wkv/generate_traj.py 的操作序列 + 模拟位姿）----
# 粒度：1 个 action = 1 个 block = 3 latent 帧 = 12 RGB 帧。
# 操作：R/L = 相机左右转（yaw，用 mouse 表达）；W/S/A/D = 前后左右平移（keyboard）；
#       _ = no-op（停留）。每种形态都采用“去程 + 逆序回程”，保证回到起点位姿，
#       从而产生可评估的回访帧对。

CAM_VALUE = 0.1          # 相机旋转的鼠标值（与 get_current_action 一致）
BLOCK_SIZE = 3           # 每 block 的 latent 帧数（num_frame_per_block）

_KB = {"W": [1, 0, 0, 0], "S": [0, 1, 0, 0], "A": [0, 0, 1, 0], "D": [0, 0, 0, 1], "_": [0, 0, 0, 0]}
_MOUSE = {"R": [0.0, CAM_VALUE], "L": [0.0, -CAM_VALUE], "_": [0.0, 0.0]}
_INV = {"W": "S", "S": "W", "A": "D", "D": "A", "R": "L", "L": "R", "_": "_"}


def _expand_ops_to_actions(ops, device):
    """把操作序列展开成 block 粒度 actions 与每块结束时的 4 维模拟位姿。

    位姿定义与 causal_inference.py 的 running_state 完全一致（WorldKV Sec 5.1）：
        state = [前后, 左右, pitch, yaw]
        前后 += BLOCK_SIZE * (W - S)
        左右 += BLOCK_SIZE * (D - A)
        pitch, yaw += mouse 原始值（±CAM_VALUE）
    """
    actions = []
    states = []
    state = [0.0, 0.0, 0.0, 0.0]
    for op in ops:
        kb = _KB.get(op, _KB["_"])
        ms = _MOUSE.get(op, _MOUSE["_"])
        state[0] += BLOCK_SIZE * (kb[0] - kb[1])
        state[1] += BLOCK_SIZE * (kb[3] - kb[2])
        state[2] += ms[0]
        state[3] += ms[1]
        actions.append({
            "keyboard": torch.tensor(kb, device=device, dtype=torch.int64),
            "mouse": torch.tensor(ms, device=device, dtype=torch.float32),
        })
        states.append(tuple(round(float(s), 4) for s in state))
    return actions, states


def _out_and_back(forward_ops):
    """去程 + 逆序回程：回程是去程每个操作的逆并按逆序排列，精确回到起点位姿。"""
    return forward_ops + [_INV[op] for op in reversed(forward_ops)]


def _find_revisit_pairs(states):
    """基于 4 维模拟位姿 [前后, 左右, pitch, yaw] 找 block 回访对，过滤原点。"""
    first_visit = {}
    pairs = []
    origin = (0.0, 0.0, 0.0, 0.0)
    for i, s in enumerate(states):
        if s == origin:
            continue
        if s in first_visit:
            pairs.append((first_visit[s], i))
        else:
            first_visit[s] = i
    return pairs


def _build_ops(shape, num_blocks):
    """按形态生成操作序列，总长度尽量不超过 num_blocks（不足由调用方填 no-op）。"""
    if shape == "rectangle":
        seg = max(1, (num_blocks - 4) // 4)          # 2 段前进 + 2 次右转（L 形去程）
        fwd = []
        for _ in range(2):
            fwd += ["W"] * seg + ["R"]
    elif shape == "zigzag":
        seg = max(1, (num_blocks - 6) // 6)          # 3 段前进 + 3 次交替转弯（之字）
        fwd = []
        for i in range(3):
            fwd += ["W"] * seg
            fwd += ["R"] if i % 2 == 0 else ["L"]
    elif shape == "rotate":
        seg = max(1, (num_blocks - 6) // 6)          # 3 段前进 + 3 次同向右转（弧线/绕圈）
        fwd = []
        for _ in range(3):
            fwd += ["W"] * seg + ["R"]
    elif shape == "strafe":
        # 带横向平移的折线：前进(W)与左右平移(D/A)交替，左右维度非零；
        # 4 操作一组重复，去程≈num_blocks//2，逆序回程精确回到原点。
        rep = max(1, (num_blocks - 1) // 8)
        fwd = ["W", "D", "W", "A"] * rep
    else:
        raise ValueError(f"未知轨迹形态: {shape}")
    return _out_and_back(fwd)


def generate_trajectory(num_blocks, device, seed=None, shape=None):
    """生成真实运动轨迹（三种形态随机），返回 (actions, revisit_pairs, shape)。

    actions: list[dict]，长度 = num_blocks，每个 dict = {keyboard[4], mouse[2]}。
    revisit_pairs: list[(ref_block, revisit_block)]，基于模拟位姿。
    """
    rng = random.Random(seed)
    if shape is None:
        shape = rng.choice(["rectangle", "zigzag", "rotate", "strafe"])
    ops = _build_ops(shape, num_blocks)
    if len(ops) > num_blocks:
        ops = ops[:num_blocks]                       # 过长截断（一般不会发生）
    ops = ops + ["_"] * (num_blocks - len(ops))      # 剩余填 no-op
    actions, states = _expand_ops_to_actions(ops, device)
    revisit_pairs = _find_revisit_pairs(states)
    return actions, revisit_pairs, shape


def build_inputs(game, args, mode, img_path=None, keep_vae=False):
    image = load_image(img_path or args.img_path)
    image = game._resizecrop(image, 352, 640)
    image = game.frame_process(image)[None, :, None, :, :].to(
        dtype=game.weight_dtype, device=game.device)

    padding_video = torch.zeros_like(image).repeat(
        1, 1, 4 * (args.max_num_output_frames - 1), 1, 1)
    img_cond = torch.concat([image, padding_video], dim=2)
    tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
    with torch.no_grad():
        img_cond = game.vae.encode(img_cond, device=game.device, **tiler_kwargs).to(game.device)
        visual_context = game.vae.clip.encode_video(image)
    mask_cond = torch.ones_like(img_cond)
    mask_cond[:, :, 1:] = 0
    cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
    if not keep_vae:
        del game.vae
        gc.collect()
        torch.cuda.empty_cache()

    sampled_noise = torch.randn(
        [1, 16, args.max_num_output_frames, 44, 80],
        device=game.device, dtype=game.weight_dtype)

    num_frames = (args.max_num_output_frames - 1) * 4 + 1
    cond_data = Bench_actions_universal(num_frames)
    conditional_dict = {
        "cond_concat": cond_concat.to(device=game.device, dtype=game.weight_dtype),
        "visual_context": visual_context.to(device=game.device, dtype=game.weight_dtype),
        "mouse_cond": cond_data["mouse_condition"].unsqueeze(0).to(
            device=game.device, dtype=game.weight_dtype),
        "keyboard_cond": cond_data["keyboard_condition"].unsqueeze(0).to(
            device=game.device, dtype=game.weight_dtype),
    }
    return conditional_dict, sampled_noise


def run_inference(game, args, conditional_dict, sampled_noise, scripted_actions, worldkv, name):
    m = game.pipeline.generator.model
    if worldkv:
        m.qvg_sparsify = False
        use_retrieval = False
    else:
        m.qvg_sparsify = True
        m.qvg_sparse_ratio = args.qvg_sparse_ratio
        m.qvg_gamma = args.qvg_gamma
        m.qvg_sensitivity_wk = args.qvg_sensitivity_wk
        m.qvg_sensitivity_wv = args.qvg_sensitivity_wv
        use_retrieval = True

    set_seed(args.seed)
    with torch.no_grad():
        video = game.pipeline.inference(
            noise=sampled_noise,
            conditional_dict=conditional_dict,
            return_latents=False,
            output_folder=args.output_folder,
            name=name,
            mode="universal",
            scripted_actions=scripted_actions,
            use_retrieval=use_retrieval,
            retrieval_topk_frames=args.retrieval_topk_frames,
            traj_threshold=None,
            kv_bank_on_gpu=False,
            quant_chunk_frames=args.quant_chunk_frames,
            worldkv=worldkv,
            residual_mag_sink=None,
        )
    print(f"[{name}] video.shape={video.shape}  "
          f"peak_alloc={torch.cuda.max_memory_allocated()/1e9:.2f} GiB  "
          f"peak_reserved={torch.cuda.max_memory_reserved()/1e9:.2f} GiB", flush=True)
    torch.cuda.reset_peak_memory_stats()
    return video


def frame_to_lpips_input(frame_uint8, device):
    t = torch.from_numpy(frame_uint8.astype(np.float32)).to(device)
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t / 127.5 - 1.0


def compute_and_report(args, revisit_pairs, video_wkv, video_qe, device):
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    rgb_per_block = 12
    results = []
    for ref_c, rev_c in revisit_pairs:
        ref_idx = ref_c * rgb_per_block + 8
        rev_idx = rev_c * rgb_per_block + 8
        l_wkv = lpips_fn(
            frame_to_lpips_input(video_wkv[ref_idx], device),
            frame_to_lpips_input(video_wkv[rev_idx], device),
        ).item()
        l_qe = lpips_fn(
            frame_to_lpips_input(video_qe[ref_idx], device),
            frame_to_lpips_input(video_qe[rev_idx], device),
        ).item()
        results.append({
            "ref_chunk": ref_c, "revisit_chunk": rev_c,
            "ref_rgb_idx": ref_idx, "revisit_rgb_idx": rev_idx,
            "lpips_worldkv": l_wkv, "lpips_quanevict": l_qe, "delta": l_qe - l_wkv,
        })

    mean_wkv = float(np.mean([r["lpips_worldkv"] for r in results]))
    mean_qe = float(np.mean([r["lpips_quanevict"] for r in results]))
    summary = {
        "ratio_base": args.qvg_sparse_ratio, "gamma": args.qvg_gamma, "seed": args.seed,
        "max_num_output_frames": args.max_num_output_frames,
        "revisit_pairs": [[a, b] for a, b in revisit_pairs],
        "mean_lpips_worldkv": mean_wkv, "mean_lpips_quanevict": mean_qe,
        "mean_delta": mean_qe - mean_wkv, "per_pair": results,
    }

    print("\n===== 回访帧 LPIPS 对比 =====")
    for r in results:
        print(f"  ref=chunk{r['ref_chunk']:2d} -> revisit=chunk{r['revisit_chunk']:2d} | "
              f"WorldKV={r['lpips_worldkv']:.4f}  QuanEvict={r['lpips_quanevict']:.4f}  "
              f"delta={r['delta']:+.4f}")
    print(f"\n平均 LPIPS  WorldKV={mean_wkv:.4f}  QuanEvict={mean_qe:.4f}  delta={mean_qe - mean_wkv:+.4f}")

    out_json = os.path.join(args.output_folder, "lpips_compare.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到 {out_json}")


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    num_blocks = args.max_num_output_frames // 3
    assert args.max_num_output_frames % 3 == 0

    if args.method == "compare":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, revisit_pairs, _ = generate_trajectory(num_blocks, device, seed=args.seed)
        wkv_path = os.path.join(args.output_folder, "worldkv_video.npy")
        qe_path = os.path.join(args.output_folder, "quanevict_video.npy")
        assert os.path.exists(wkv_path), f"缺少 {wkv_path}"
        assert os.path.exists(qe_path), f"缺少 {qe_path}"
        video_wkv = np.load(wkv_path)
        video_qe = np.load(qe_path)
        compute_and_report(args, revisit_pairs, video_wkv, video_qe, device)
        return

    set_seed(args.seed)
    game = InteractiveGameInference(args)
    mode = game.config.pop("mode")
    assert mode == "universal"

    chunk_size = game.pipeline.num_frame_per_block
    assert chunk_size == 3 and args.max_num_output_frames // chunk_size == num_blocks

    conditional_dict, sampled_noise = build_inputs(game, args, mode)
    scripted_actions, revisit_pairs, shape = generate_trajectory(num_blocks, game.device, seed=args.seed)

    print(f"轨迹形态：{shape}（共 {num_blocks} 块）")
    print(f"回访 block 对（ref, revisit）：{revisit_pairs}", flush=True)

    video_wkv = None
    video_qe = None

    if args.method in ("both", "worldkv"):
        print("\n=== 纯 WorldKV ===", flush=True)
        video_wkv = run_inference(game, args, conditional_dict, sampled_noise,
                                  scripted_actions, worldkv=True, name="worldkv")
        np.save(os.path.join(args.output_folder, "worldkv_video.npy"), video_wkv)
        if args.method == "worldkv":
            print("已保存 worldkv_video.npy，可稍后跑 --method quanevict 与 --method compare")
            return

    if args.method in ("both", "quanevict"):
        gc.collect()
        torch.cuda.empty_cache()
        print("\n=== QuanEvict (ratio_base=%.3f, gamma=%.3f) ===" % (args.qvg_sparse_ratio, args.qvg_gamma), flush=True)
        video_qe = run_inference(game, args, conditional_dict, sampled_noise,
                                 scripted_actions, worldkv=False, name="quanevict")
        np.save(os.path.join(args.output_folder, "quanevict_video.npy"), video_qe)
        if args.method == "quanevict":
            print("已保存 quanevict_video.npy，可稍后跑 --method compare")
            return

    compute_and_report(args, revisit_pairs, video_wkv, video_qe, game.device)


if __name__ == "__main__":
    main()

"""用真实 KV-cache residual 验证 k-means 不同 iters 的聚类差异。

流程：加载模型跑一次 QuanEvict 推理，monkey-patch `_kmeans_batched` 捕获
真实输入 residual（即 KV-cache 张量），再对捕获到的若干 chunk 用固定 init 种子
跑不同 iters，对比 SSE / 质心 L2 差 / assign 一致率。
"""
import os
import sys

os.environ.pop("EVAL_SAVE_VIDEO", None)

import torch

import utils.visualize as viz
viz.process_video = lambda *a, **k: None  # 禁用 mp4 编码

import evaluate_four_methods as efm
import wan.utils.qvg as qvg
from inference_streaming import InteractiveGameInference
from utils.misc import set_seed
from wan.utils.qvg import _gather_centroids

captured = []
_orig_kmeans = qvg._kmeans_batched


def _patched_kmeans(x, *a, **k):
    if len(captured) < 8:
        captured.append(x.detach().cpu())
    return _orig_kmeans(x, *a, **k)


qvg._kmeans_batched = _patched_kmeans


def analyze(x, tag, device, K=256, ref_iters=30):
    iters_list = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
    x = x.float().contiguous().to(device)
    T = x.shape[0]
    if T < K:
        print(f"[{tag}] T={T} < K={K}, 跳过")
        return
    c_ref, a_ref = _orig_kmeans(x, K, ref_iters, seed=777, use_triton=True)
    a_ref = a_ref.long()
    sse_ref = ((x - _gather_centroids(c_ref, a_ref)) ** 2).mean().item()

    print(f"[{tag}] shape={tuple(x.shape)}  mean|x|={x.abs().mean().item():.4f}")
    print(f"{'iters':>6} | {'SSE':>12} | {'SSE/ref':>9} | {'质心L2差':>12} | {'assign一致率':>12}")
    for it in iters_list:
        c, a = _orig_kmeans(x, K, it, seed=777, use_triton=True)
        a = a.long()
        sse = ((x - _gather_centroids(c, a)) ** 2).mean().item()
        cd = (c - c_ref).pow(2).sum().sqrt().item()
        am = (a == a_ref).float().mean().item()
        print(f"{it:>6} | {sse:>12.4f} | {sse/sse_ref:>9.4f} | {cd:>12.4f} | {am:>12.4f}")


def main():
    args = efm.parse_args()
    set_seed(args.seed)
    game = InteractiveGameInference(args)
    mode = game.config.pop("mode")
    m = game.pipeline.generator.model
    print(f"[info] num_heads={m.num_heads}  head_dim={m.dim // m.num_heads}")

    num_blocks = args.max_num_output_frames // 3
    actions, revisit_pairs, shape = efm.generate_trajectory(
        num_blocks, game.device, seed=args.seed, shape=args.shape)
    imgs = efm.select_images(args.img_dir, 1)
    cond, noise = efm.build_inputs(game, args, mode, img_path=imgs[0], keep_vae=True)

    method = efm.METHOD_BY_KEY["quanevict"]
    efm.run_one(game, args, cond, noise, actions, method, "dump")
    torch.cuda.empty_cache()

    print(f"\ncaptured {len(captured)} 个 kmeans 输入（residual）")
    for i, x in enumerate(captured[:8]):
        print(f"  [{i}] {tuple(x.shape)}  mean|x|={x.float().abs().mean().item():.4f}")

    # 取前几个 T 足够大的 chunk 做分析（K/V 交替，各 chunk 分布不同）
    done = 0
    for i, x in enumerate(captured):
        if x.shape[0] < 256:
            continue
        analyze(x, f"residual[{i}]", game.device)
        torch.cuda.empty_cache()
        done += 1
        if done >= 4:
            break


if __name__ == "__main__":
    main()

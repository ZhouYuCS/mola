"""QuanEvict 分段推理开销 profiling。

Monkey-patch wan.utils.qvg 各阶段函数，实测一次完整推理（1 图 / 51 latent 帧）
中 QVG 压缩/解压各子阶段的累计耗时，配合总延迟拆解 QuanEvict 的慢在哪。

用法：
  python profile_quanevict.py [--qvg_sparse_ratio 0.3] [--qvg_use_triton 1/0]
"""
import os
import sys
import time

os.environ.pop("EVAL_SAVE_VIDEO", None)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# 在 import inference_streaming 前禁用 mp4 编码
import utils.visualize as viz  # noqa: E402
viz.process_video = lambda *a, **k: None  # noqa: E402

import evaluate_four_methods as efm  # noqa: E402
import wan.utils.qvg as qvg  # noqa: E402
import pipeline.causal_inference as ci  # noqa: E402
from inference_streaming import InteractiveGameInference  # noqa: E402
from utils.misc import set_seed  # noqa: E402

# ---- monkey-patch 计时（GPU 同步，确保测到真实 GPU 时间）----
_stats = {}
_first = {}


def _sync():
    torch.cuda.synchronize()


def _record(name, dt):
    t, c = _stats.get(name, (0.0, 0))
    _stats[name] = (t + dt, c + 1)


def _wrap(fn, name):
    def inner(*a, **k):
        _sync()
        s = time.perf_counter()
        r = fn(*a, **k)
        _sync()
        _record(name, time.perf_counter() - s)
        return r
    return inner


_orig_compress = qvg.qvg_compress
_orig_decomp = qvg.qvg_decompress_frames
_orig_kmeans = qvg._kmeans_batched
_orig_pairwise = qvg._pairwise_sqdist
_orig_quant = qvg._symmetric_quantize
_orig_quant_flat = qvg._symmetric_quantize_flat
_orig_keepmask = qvg._residual_keep_mask


def _patched_compress(*a, **k):
    if "x_shape" not in _first:
        _first["x_shape"] = tuple(a[0].shape)
        _first["num_centroids"] = k.get("num_centroids")
        _first["num_stages"] = k.get("num_stages")
        _first["kmeans_iters"] = k.get("kmeans_iters")
        _first["group_size"] = k.get("group_size")
        _first["bits"] = k.get("bits")
        _first["use_triton"] = k.get("use_triton")
    _sync()
    s = time.perf_counter()
    r = _orig_compress(*a, **k)
    _sync()
    _record("qvg_compress", time.perf_counter() - s)
    return r


qvg.qvg_compress = _patched_compress
qvg.qvg_decompress_frames = _wrap(_orig_decomp, "qvg_decompress_frames")
qvg._kmeans_batched = _wrap(_orig_kmeans, "_kmeans_batched")
qvg._pairwise_sqdist = _wrap(_orig_pairwise, "_pairwise_sqdist")
qvg._symmetric_quantize = _wrap(_orig_quant, "_symmetric_quantize")
qvg._symmetric_quantize_flat = _wrap(_orig_quant_flat, "_symmetric_quantize_flat")
qvg._residual_keep_mask = _wrap(_orig_keepmask, "_residual_keep_mask")

# causal_inference.py 通过 from-import 绑定了 qvg_compress / qvg_decompress_frames，
# 实际推理路径走的是这些模块全局名，需单独覆盖。
ci.qvg_compress = _patched_compress
ci.qvg_decompress_frames = _wrap(_orig_decomp, "qvg_decompress_frames")


def main():
    args = efm.parse_args()
    set_seed(args.seed)
    game = InteractiveGameInference(args)
    mode = game.config.pop("mode")
    m = game.pipeline.generator.model

    print(f"[info] num_heads={m.num_heads}  head_dim={m.dim // m.num_heads}  dim={m.dim}")
    print(f"[info] num_transformer_blocks={game.pipeline.num_transformer_blocks}  "
          f"frame_seq_length={game.pipeline.frame_seq_length}  "
          f"num_frame_per_block={game.pipeline.num_frame_per_block}")

    num_blocks = args.max_num_output_frames // 3
    actions, revisit_pairs, shape = efm.generate_trajectory(
        num_blocks, game.device, seed=args.seed, shape=args.shape)

    imgs = efm.select_images(args.img_dir, 1)
    cond, noise = efm.build_inputs(game, args, mode, img_path=imgs[0], keep_vae=True)

    method = efm.METHOD_BY_KEY["quanevict"]
    name = f"prof_quanevict_r{args.qvg_sparse_ratio}"

    _sync()
    t0 = time.perf_counter()
    video, bank, peak, lat = efm.run_one(game, args, cond, noise, actions, method, name)
    _sync()
    total = time.perf_counter() - t0

    print("\n===== 分段开销（GPU 同步计时） =====")
    print(f"总推理耗时          : {total:8.2f}s")
    print(f"bank={bank / 1e6:.1f}MB  peak={peak / 1e9:.2f}GiB")
    print(f"首次 qvg_compress 输入形状: {_first.get('x_shape')}")
    print(f"num_centroids={_first.get('num_centroids')}  num_stages={_first.get('num_stages')}  "
          f"kmeans_iters={_first.get('kmeans_iters')}  group_size={_first.get('group_size')}  "
          f"bits={_first.get('bits')}  use_triton={_first.get('use_triton')}")

    # 顶层阶段（真实计时）：压缩 + 解压
    top = ["qvg_compress", "qvg_decompress_frames"]
    qvg_total = sum(_stats[n][0] for n in top if n in _stats)
    for name in top:
        if name in _stats:
            t, c = _stats[name]
            print(f"  {name:24s}: {t:8.2f}s  x {c:4d} 次")
    print(f"  {'QVG 合计':24s}: {qvg_total:8.2f}s  ({qvg_total / total * 100:.1f}%)")

    # 压缩子阶段（已含在 qvg_compress 内，仅供定位，不重复累加）
    sub = ["_kmeans_batched", "_pairwise_sqdist",
           "_symmetric_quantize", "_symmetric_quantize_flat",
           "_residual_keep_mask"]
    print("  -- 压缩子阶段（已含在 qvg_compress 内，仅供定位）--")
    for name in sub:
        if name in _stats:
            t, c = _stats[name]
            print(f"    {name:22s}: {t:8.2f}s  x {c:4d} 次")

    other = total - qvg_total
    print(f"  其余（DiT/采样/VAE/IO）: {other:8.2f}s  ({other / total * 100:.1f}%)")


if __name__ == "__main__":
    main()

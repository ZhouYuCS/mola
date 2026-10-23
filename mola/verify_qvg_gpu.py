"""QVG Triton 内核 GPU 数值验证（隔离加载，不触发 wan 包顶层初始化）。

验证三项：
1. _pairwise_dot：Triton vs torch.einsum（k-means 的 GEMM 项）
2. scatter_residual：Triton vs index_add_（帧级反量化 scatter-add）
3. 端到端：qvg_compress + qvg_decompress_frames 的 Triton/回退一致性

用法（需 CUDA + triton）：
    python verify_qvg_gpu.py
"""
import sys
import types
import importlib.util
from pathlib import Path

import torch


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_qvg(root: Path):
    qvt = _load_module("qvg_triton", root / "wan" / "utils" / "qvg_triton.py")
    wan = types.ModuleType("wan")
    wan_utils = types.ModuleType("wan.utils")
    wan.utils = wan_utils
    wan_utils.qvg_triton = qvt
    sys.modules["wan"] = wan
    sys.modules["wan.utils"] = wan_utils
    sys.modules["wan.utils.qvg_triton"] = qvt
    qvg = _load_module("wan.utils.qvg", root / "wan" / "utils" / "qvg.py")
    return qvt, qvg


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA 不可用，请在 GPU 环境运行。", flush=True)
        sys.exit(1)

    root = Path(__file__).resolve().parent
    qvt, qvg = _load_qvg(root)
    if not qvt.has_triton():
        print("FAIL: 未检测到 triton。", flush=True)
        sys.exit(1)

    dev = "cuda"
    torch.manual_seed(0)
    results = []

    # 1) _pairwise_dot
    T, H, K, D = 3520, 12, 256, 128
    x = torch.randn(T, H, D, device=dev, dtype=torch.float32)
    c = torch.randn(H, K, D, device=dev, dtype=torch.float32)
    ref = torch.einsum("thd,hkd->thk", x, c)
    got = qvt._pairwise_dot(x, c)
    ad = (ref - got).abs()
    rel = ad / (ref.abs() + 1e-6)
    ok1 = ad.max().item() < 1e-3 and rel.max().item() < 1e-4
    results.append(ok1)
    print(f"[{'PASS' if ok1 else 'FAIL'}] _pairwise_dot      "
          f"max_abs={ad.max().item():.3e} max_rel={rel.max().item():.3e}", flush=True)

    # 2) scatter_residual
    N, G, B = 20000, 2, 64
    D = G * B
    residual_int = torch.randint(-1, 2, (N, G, B), device=dev, dtype=torch.int8)
    scale = torch.rand(N, G, device=dev, dtype=torch.float32) + 0.05
    flat_idx = torch.randperm(N, device=dev, dtype=torch.int32)  # 唯一索引
    c_ref = torch.zeros(N, D, device=dev, dtype=torch.float32)
    resid = (residual_int.float() * scale.unsqueeze(-1)).reshape(N, D)
    c_ref.index_add_(0, flat_idx.long(), resid)
    c_got = torch.zeros(N, D, device=dev, dtype=torch.float32)
    qvt.scatter_residual(residual_int, scale, flat_idx, c_got)
    ad2 = (c_ref - c_got).abs()
    ok2 = ad2.max().item() < 1e-5
    results.append(ok2)
    print(f"[{'PASS' if ok2 else 'FAIL'}] scatter_residual  "
          f"max_abs={ad2.max().item():.3e}", flush=True)

    # 3) 端到端 round-trip
    frame_seqlen, num_frames = 880, 12
    total = frame_seqlen * num_frames
    x = torch.randn(1, total, 12, 128, device=dev, dtype=torch.bfloat16)
    kw = dict(bits=2, num_centroids=64, num_stages=1, group_size=64,
              kmeans_iters=3, seed=0, sparsify=True, sparse_ratio=0.5, num_frames=num_frames)
    frames = [0, 5, 11]
    p_tri = qvg.qvg_compress(x, use_triton=True, **kw)
    p_ref = qvg.qvg_compress(x, use_triton=False, **kw)
    out_tri = qvg.qvg_decompress_frames(p_tri, frames, out_dtype=None, use_triton=True)
    out_ref = qvg.qvg_decompress_frames(p_ref, frames, out_dtype=None, use_triton=False)
    assert out_tri.shape == out_ref.shape == (len(frames) * frame_seqlen, 12, 128)
    assign_mismatch = (p_tri['assign'][0] != p_ref['assign'][0]).float().mean().item()
    ad3 = (out_tri - out_ref).abs().max().item()
    ok3 = assign_mismatch < 0.01 and ad3 < 1e-2
    results.append(ok3)
    print(f"[{'PASS' if ok3 else 'FAIL'}] roundtrip(frames={frames}) "
          f"assign_mismatch={assign_mismatch:.2e} max_abs={ad3:.3e}", flush=True)

    all_ok = all(results)
    print("ALL PASS" if all_ok else "SOME FAILED", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

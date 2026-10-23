"""k-means 融合路径 vs 回退 的微基准（规模对齐真实推理）。"""
import time
import torch
from wan.utils import qvg

torch.manual_seed(0)
dev = "cuda"
T, H, D, K, iters = 10560, 12, 128, 256, 10
x = torch.randn(T, H, D, device=dev)

def bench(use_triton):
    # 预热
    qvg._kmeans_batched(x, K, 2, seed=0, use_triton=use_triton)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        c, a = qvg._kmeans_batched(x, K, iters, seed=0, use_triton=use_triton)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 3

t_triton = bench(True)
t_torch = bench(False)
print(f"kmeans_batched (T={T},K={K},D={D},iters={iters}):")
print(f"  triton fused : {t_triton:.3f}s")
print(f"  torch fallback: {t_torch:.3f}s")
print(f"  加速比        : {t_torch / t_triton:.2f}x")

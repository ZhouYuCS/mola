"""验证 k-means Triton 融合路径与 PyTorch 回退的一致性。"""
import torch
from wan.utils import qvg

torch.manual_seed(0)
dev = "cuda"
T, H, D, K, iters = 2048, 12, 128, 256, 5
x = torch.randn(T, H, D, device=dev)

print("has_triton =", qvg._qvt.has_triton())

c_t, a_t = qvg._kmeans_batched(x, K, iters, seed=0, use_triton=True)
c_p, a_p = qvg._kmeans_batched(x, K, iters, seed=0, use_triton=False)

same = (a_t == a_p).float().mean().item()
cent_diff = (c_t - c_p).norm(dim=-1).mean().item()
cent_scale = c_t.norm(dim=-1).mean().item()
print(f"assign 一致率 = {same:.4f}")
print(f"centroids 平均 L2 差 = {cent_diff:.6f}   (centroids 平均模长 = {cent_scale:.4f})")
print(f"assign dtype triton={a_t.dtype} torch={a_p.dtype}")
print("OK" if same > 0.98 else "DIFF TOO LARGE")

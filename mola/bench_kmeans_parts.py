"""精细拆解 k-means 一次迭代各部分耗时。"""
import time
import torch
import torch.nn.functional as F
from wan.utils import qvg, qvg_triton as qvt

torch.manual_seed(0)
dev = "cuda"
T, H, D, K = 10560, 12, 128, 256
x = torch.randn(T, H, D, device=dev)
c = torch.randn(H, K, D, device=dev)

def t(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms

# 1. pairwise_sqdist (triton dot)
t1 = t(lambda: qvg._pairwise_sqdist(x, c, use_triton=True))
t1e = t(lambda: qvg._pairwise_sqdist(x, c, use_triton=False))
print(f"pairwise_sqdist triton-dot : {t1:.2f} ms")
print(f"pairwise_sqdist einsum-dot : {t1e:.2f} ms")

dist = qvg._pairwise_sqdist(x, c, use_triton=True)

# 2. argmin
t2 = t(lambda: dist.argmin(dim=-1))
print(f"argmin                     : {t2:.2f} ms")

# 3. onehot + einsum + counts
def old_update():
    a = dist.argmin(dim=-1)
    onehot = F.one_hot(a, num_classes=K).float()
    counts = onehot.sum(dim=0).clamp_min(1.0)
    sums = torch.einsum('thk,thd->hkd', onehot, x)
    return sums, counts
t3 = t(old_update)
print(f"onehot+einsum+counts (旧)  : {t3:.2f} ms")

# 4. scatter kernel
assign32 = dist.argmin(dim=-1).to(torch.int32)
sums = torch.zeros(H, K, D, device=dev)
counts = torch.zeros(H, K, device=dev)
def new_update():
    sums.zero_(); counts.zero_()
    qvt.kmeans_scatter_update(x, assign32, sums, counts)
    return sums, counts
t4 = t(new_update)
print(f"scatter kernel (新)        : {t4:.2f} ms")

# 5. 校验 scatter 与 einsum 结果一致
s_old, c_old = old_update()
s_new, c_new = new_update()
print("sums 一致:", torch.allclose(s_old, s_new, atol=1e-3), " counts 一致:", torch.allclose(c_old, c_new))

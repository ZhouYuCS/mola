"""分析 k-means 不同 iters 的聚类结果差异。

用与 QVG 实际相同的维度（T,H,D,K），随机采样多组数据，
固定 init 种子，跑不同 iters 的 _kmeans_batched，比较：
  - SSE      ：平均到质心平方距离（k-means 目标函数，越低越好）
  - 质心 L2 差：相对收敛基准 iters 的 Frobenius 距离
  - assign 一致率：相对收敛基准 iters 的赋值一致比例
"""
import torch
from wan.utils.qvg import _kmeans_batched, _gather_centroids


def run(T=10560, H=12, D=128, K=256, seeds=(0, 1, 2, 3, 4),
        iters_list=(1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30), ref_iters=30):
    dev = 'cuda'
    rows = {i: {'sse': [], 'cd': [], 'am': []} for i in iters_list}

    for seed in seeds:
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(T, H, D, device=dev, generator=g).float()

        c_ref, a_ref = _kmeans_batched(x, K, ref_iters, seed=1000 + seed, use_triton=True)
        a_ref = a_ref.long()
        sse_ref = ((x - _gather_centroids(c_ref, a_ref)) ** 2).mean().item()

        for it in iters_list:
            c, a = _kmeans_batched(x, K, it, seed=1000 + seed, use_triton=True)
            a = a.long()
            sse = ((x - _gather_centroids(c, a)) ** 2).mean().item()
            cd = (c - c_ref).pow(2).sum().sqrt().item()
            am = (a == a_ref).float().mean().item()
            rows[it]['sse'].append(sse)
            rows[it]['cd'].append(cd)
            rows[it]['am'].append(am)

    n = len(seeds)
    print(f"维度 T={T} H={H} D={D} K={K} | {n} 个随机样本平均 | 收敛基准 iters={ref_iters}")
    print(f"{'iters':>6} | {'SSE':>12} | {'SSE/ref':>9} | {'质心L2差':>12} | {'assign一致率':>12}")
    for it in iters_list:
        sse = sum(rows[it]['sse']) / n
        cd = sum(rows[it]['cd']) / n
        am = sum(rows[it]['am']) / n
        print(f"{it:>6} | {sse:>12.4f} | {sse/sse_ref:>9.4f} | {cd:>12.4f} | {am:>12.4f}")


if __name__ == "__main__":
    run()

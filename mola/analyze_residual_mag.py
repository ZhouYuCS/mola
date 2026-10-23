"""分析 QVG 残差幅值，统计不同阈值下的残差驱逐率。

数据格式：``torch.save`` 保存的 ``list[(layer_idx, kind, chunk_id, mag)]``，
其中 ``kind`` 为 ``'k'`` / ``'v'``，``mag`` 为 ``[T, H]`` fp32（残差幅值
``residual.norm(dim=-1)``，即 L2 范数）。

驱逐率定义为 ``P(mag < threshold)``，即残差幅值低于阈值而被驱逐的
(token, head) 占比。这里的"驱逐"对象是残差，不是整个 K/V 向量。

用法：
    python analyze_residual_mag.py --input <path> [--thresholds 0,0.1,0.5,1.0]
"""

import argparse
import numpy as np
import torch


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def quantile_sampled(x, qs, max_elems=10_000_000):
    """对过大张量采样后估算分位数，避免 torch.quantile 排序超大张量。"""
    x = x.reshape(-1)
    if x.numel() > max_elems:
        idx = torch.randint(0, x.numel(), (max_elems,), dtype=torch.long)
        x = x[idx]
    return torch.quantile(x, torch.as_tensor(qs, dtype=x.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='residual_mag_out 保存的文件')
    ap.add_argument(
        '--thresholds', default=None,
        help='逗号分隔的绝对阈值；默认基于 K/V 合并的全局均值自动生成')
    args = ap.parse_args()

    data = load(args.input)
    assert data, 'empty data'

    kinds = ['k', 'v']
    H = data[0][3].shape[1]
    num_layers = max(rec[0] for rec in data) + 1

    # (kind, layer) -> list of mag
    by_layer_kind = {}
    for layer, kind, cid, mag in data:
        by_layer_kind.setdefault((kind, layer), []).append(mag)

    mag_all = {}
    for kind in kinds:
        tensors = [m for (k, _), ms in by_layer_kind.items() if k == kind for m in ms]
        mag_all[kind] = torch.cat(tensors, dim=0).float()  # [T_total, H]

    # 残差幅值量级参考
    print('===== 残差幅值量级（mag = ||residual||_2，L2 范数） =====')
    for kind in kinds:
        m = mag_all[kind].reshape(-1)
        q = quantile_sampled(m, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f'[{kind.upper()}] min={q[0]:.6g} p10={q[1]:.6g} p25={q[2]:.6g} '
              f'p50={q[3]:.6g} p75={q[4]:.6g} p90={q[5]:.6g} max={q[6]:.6g} '
              f'mean={m.mean():.6g}')

    allcat = torch.cat([mag_all[k].reshape(-1) for k in kinds])
    if args.thresholds:
        thr_list = [float(x) for x in args.thresholds.split(',')]
    else:
        mean = allcat.mean().item()
        thr_list = [0.0, mean * 0.25, mean * 0.5, mean * 0.75,
                    mean, mean * 1.5, mean * 2.0, mean * 3.0]

    # 全局驱逐率
    print('\n===== 全局驱逐率 P(mag < thr) =====')
    header = 'threshold'.ljust(14) + ''.join(f'{k.upper():>14}' for k in kinds)
    print(header)
    for thr in thr_list:
        row = f'{thr:<14.6g}'
        for kind in kinds:
            rate = (mag_all[kind] < thr).float().mean().item()
            row += f'{rate:>14.4f}'
        print(row)

    # 选代表性阈值：全局中位数（驱逐率约 50%）
    rep_thr = quantile_sampled(allcat, [0.5]).item()

    # per-layer
    print(f'\n===== 代表性阈值 {rep_thr:.6g} 下的 per-layer 驱逐率 =====')
    for kind in kinds:
        rates = []
        for layer in range(num_layers):
            ms = by_layer_kind.get((kind, layer), [])
            if not ms:
                continue
            m = torch.cat(ms, dim=0)
            rates.append((m < rep_thr).float().mean().item())
        arr = np.array(rates)
        print(f'[{kind.upper()}] layers={len(arr)} mean={arr.mean():.4f} '
              f'min={arr.min():.4f} max={arr.max():.4f} std={arr.std():.4f} '
              f'spread={arr.max()-arr.min():.4f}')
        print('   per-layer: ' + ' '.join(f'{r:.3f}' for r in rates))

    # per-head（聚合所有 layer）
    print(f'\n===== 代表性阈值 {rep_thr:.6g} 下的 per-head 驱逐率（聚合所有 layer） =====')
    for kind in kinds:
        m = mag_all[kind]  # [T, H]
        head_rates = (m < rep_thr).float().mean(dim=0).numpy()
        print(f'[{kind.upper()}] heads={H} mean={head_rates.mean():.4f} '
              f'min={head_rates.min():.4f} max={head_rates.max():.4f} '
              f'std={head_rates.std():.4f} spread={head_rates.max()-head_rates.min():.4f}')
        print('   per-head: ' + ' '.join(f'{r:.3f}' for r in head_rates))

    # layer x head 二维差异
    print(f'\n===== 交叉差异（代表性阈值 {rep_thr:.6g}） =====')
    for kind in kinds:
        mat = torch.zeros(num_layers, H)
        for layer in range(num_layers):
            ms = by_layer_kind.get((kind, layer), [])
            if not ms:
                mat[layer] = float('nan')
                continue
            m = torch.cat(ms, dim=0)
            mat[layer] = (m < rep_thr).float().mean(dim=0)
        valid = ~torch.isnan(mat).any(dim=1)
        mat = mat[valid]
        head_std_across_layers = mat.std(dim=0)
        layer_std_across_heads = mat.std(dim=1)
        print(f'[{kind.upper()}] head 跨 layer 的 std: mean={head_std_across_layers.mean():.4f} '
              f'max={head_std_across_layers.max():.4f}')
        print(f'[{kind.upper()}] layer 跨 head 的 std: mean={layer_std_across_heads.mean():.4f} '
              f'max={layer_std_across_heads.max():.4f}')


if __name__ == '__main__':
    main()

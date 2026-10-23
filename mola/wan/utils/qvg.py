"""Quant VideoGen (QVG): 2-bit KV-cache 量化的推理算法实现。

参考论文:
  Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache
  Quantization (arXiv:2602.02958)

QVG 针对自回归视频扩散模型不断增长的 KV-cache，提出两个核心组件:

1. Semantic-Aware Smoothing (SAS):
   把 KV-cache 按 chunk 划分后，用 k-means 将语义相近的 token 聚成 C 组，
   每组减去其质心 (centroid)，得到幅值更小、更易量化的残差:

       R_i = X_{G_i} - C_i

2. Progressive Residual Quantization (PRQ):
   对残差迭代地重复 SAS，粗到细地逐级压缩:

       R^(0) = X
       R^(t), C^(t), pi^(t) = SA-Smoothing(R^(t-1), C),  t = 1..T

   最后对 R^(T) 做对称逐组整数量化，重建时从第 T 级逐级加回质心。

量化公式 (symmetric per-group integer quantization, 论文 Eq. 1-2):

   S_X = max(|X|) / (2^(b-1) - 1)
   X_INT = round(X / S_X)
   X_hat = S_X * X_INT

本文件提供两个入口:

- ``qvg_roundtrip``: 量化-反量化 round-trip（用于误差仿真）。
- ``qvg_compress`` / ``qvg_decompress``: 真压缩存储 + 按需反量化恢复，供
  12 帧 chunk 的 KV bank 使用。

此外支持「残差比例稀疏化」：按每个 (token, head) 的最终残差幅值（L2 范数）
排序，每个 head 保留幅值最大的 top-K 残差（K 由保留比例决定，该比例可由
QuanEvict 敏感度自适应到不同 head），其余残差不存储，解压时直接用质心近似，
从而降低显存占用与读取时的访存/计算开销。可选 Triton 内核见
``wan.utils.qvg_triton``。
"""

import torch
import torch.nn.functional as F

from wan.utils import qvg_triton as _qvt


def _clamp_centroids(x: torch.Tensor, k: int) -> int:
    """当 token 数小于期望质心数时，退化为每个 token 一个质心。"""
    T = x.shape[0]
    return min(k, T)


def _kmeans_init(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """k-means 初始化。

    采用随机采样 ``k`` 个 token 作为初始质心（每个 head 独立），后续由
    Lloyd 迭代细化。k-means++ 初始化可作为进一步优化。

    Args:
        x: ``[T, H, D]``，T 为 token 数，H 为 head 数，D 为特征维。
        k: 质心数（已 clamp 到 <= T）。

    Returns:
        质心 ``[H, K, D]``。
    """
    T, H, D = x.shape
    if k >= T:
        # 退化为每个 token 一个质心（此时残差近似为 0）
        centroids = torch.zeros(H, k, D, device=x.device, dtype=x.dtype)
        centroids[:, :T, :] = x.transpose(0, 1)
        return centroids

    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(T, generator=g, device=x.device)[:k]
    return x[idx, :, :].transpose(0, 1).contiguous()  # [H, K, D]


def _pairwise_sqdist(x: torch.Tensor, c: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    """计算 ``||x - c||^2``。

    Args:
        x: ``[T, H, D]``。
        c: ``[H, K, D]``。
        use_triton: 是否用 Triton 内核计算 GEMM 项（回退 einsum）。

    Returns:
        距离平方 ``[T, H, K]``。
    """
    x_norm = (x * x).sum(dim=-1, keepdim=True)          # [T, H, 1]
    c_norm = (c * c).sum(dim=-1)                         # [H, K]
    dot = _qvt._pairwise_dot(x, c) if use_triton else torch.einsum(
        'thd,hkd->thk', x, c)
    return x_norm + c_norm.unsqueeze(0) - 2.0 * dot


def _kmeans_batched(x: torch.Tensor, k: int, iters: int, seed: int,
                    use_triton: bool = True):
    """对每个 head 独立执行向量化的 Lloyd k-means。

    Args:
        x: ``[T, H, D]``。
        k: 质心数。
        iters: Lloyd 迭代次数。
        seed: 随机种子。
        use_triton: 是否用 Triton 加速质心累加（scatter-add）。

    Returns:
        (centroids, assign)：
          centroids ``[H, K, D]``；assign ``[T, H]`` (long)。
    """
    k = _clamp_centroids(x, k)
    T, H, D = x.shape
    centroids = _kmeans_init(x, k, seed)

    can_triton = (
        use_triton and _qvt.has_triton() and x.is_cuda
        and D >= 16 and (D & (D - 1)) == 0
    )

    if can_triton:
        # 距离+argmin 用 Triton 融合内核（tf32 dot，不落地 [T,H,K] 距离张量），
        # 质心累加用 Triton scatter（原子加，替代 onehot+einsum）。
        sums = torch.zeros((H, k, D), device=x.device, dtype=torch.float32)
        counts = torch.zeros((H, k), device=x.device, dtype=torch.float32)
        for _ in range(iters):
            assign32 = _qvt.kmeans_assign(x, centroids)  # [T, H] int32
            sums.zero_()
            counts.zero_()
            _qvt.kmeans_scatter_update(x, assign32, sums, counts)
            centroids = sums / counts.clamp_min(1.0).unsqueeze(-1)
        # 最终质心确定后，重新计算一次 assignment
        assign32 = _qvt.kmeans_assign(x, centroids)
        return centroids, assign32

    for _ in range(iters):
        dist = _pairwise_sqdist(x, centroids, use_triton=False)  # [T, H, K]
        assign = dist.argmin(dim=-1)                            # [T, H]
        onehot = F.one_hot(assign, num_classes=k).float()       # [T, H, K]
        counts = onehot.sum(dim=0).clamp_min(1.0)               # [H, K]
        sums = torch.einsum('thk,thd->hkd', onehot, x)          # [H, K, D]
        centroids = sums / counts.unsqueeze(-1)

    # 最终质心确定后，重新计算一次 assignment
    dist = _pairwise_sqdist(x, centroids, use_triton=False)
    assign = dist.argmin(dim=-1)
    return centroids, assign


def _gather_centroids(centroids: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
    """按 assignment 取回每个 token 所属的质心。

    Args:
        centroids: ``[H, K, D]``。
        assign: ``[T, H]`` (long)。

    Returns:
        ``[T, H, D]``。
    """
    H, K, D = centroids.shape
    T = assign.shape[0]
    h_idx = torch.arange(H, device=centroids.device).view(1, H).expand(T, H).reshape(-1)
    idx = h_idx * K + assign.reshape(-1)
    return centroids.reshape(H * K, D)[idx].reshape(T, H, D)


def _symmetric_quantize_dequantize(x: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """对称逐组整数量化的 round-trip（论文 Eq. 1-2）。"""
    T, H, D = x.shape
    if D % group_size != 0:
        group_size = D
    qmax = float((1 << (bits - 1)) - 1)

    xg = x.reshape(T, H, D // group_size, group_size)
    scale = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    x_int = (xg / scale).round().clamp(-qmax, qmax)
    return (x_int * scale).reshape(T, H, D)


def _qvg_roundtrip_batched(
    x: torch.Tensor,
    bits: int,
    num_centroids: int,
    num_stages: int,
    group_size: int,
    kmeans_iters: int,
    seed: int,
) -> torch.Tensor:
    """QVG round-trip 的核心实现（输入已 reshape 为 ``[T, H, D]``）。"""
    residual = x
    centroids_list = []
    assign_list = []

    for t in range(num_stages):
        centroids, assign = _kmeans_batched(
            residual, num_centroids, kmeans_iters, seed + t)
        residual = residual - _gather_centroids(centroids, assign)
        centroids_list.append(centroids)
        assign_list.append(assign)

    reconstructed = _symmetric_quantize_dequantize(residual, bits, group_size)

    for t in range(num_stages - 1, -1, -1):
        reconstructed = reconstructed + _gather_centroids(
            centroids_list[t], assign_list[t])

    return reconstructed


def qvg_roundtrip(
    x: torch.Tensor,
    bits: int = 2,
    num_centroids: int = 256,
    num_stages: int = 1,
    group_size: int = 64,
    kmeans_iters: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """Quant VideoGen 量化-反量化统一入口（round-trip 误差仿真）。"""
    if bits is None or bits <= 0 or x.numel() == 0:
        return x

    orig_shape = x.shape
    H = x.shape[-2]
    D = x.shape[-1]
    xf = x.reshape(-1, H, D).float()

    out = _qvg_roundtrip_batched(
        xf, bits, num_centroids, num_stages, group_size, kmeans_iters, seed)

    return out.reshape(orig_shape).to(x.dtype)


def _symmetric_quantize(
    x: torch.Tensor, bits: int, group_size: int
) -> tuple:
    """对称逐组整数量化，返回 ``(scale, x_int)`` 用于压缩存储。

    Args:
        x: ``[T, H, D]``。
        bits: 位宽 b。
        group_size: 沿通道维的分组大小。

    Returns:
        scale ``[T, H, D//group_size]``（fp32）；x_int ``[T, H, D//group_size, group_size]``（int8）。
    """
    T, H, D = x.shape
    if D % group_size != 0:
        group_size = D
    qmax = float((1 << (bits - 1)) - 1)

    xg = x.reshape(T, H, D // group_size, group_size)
    scale = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax  # [T,H,G,1]
    x_int = (xg / scale).round().clamp(-qmax, qmax).to(torch.int8)      # [T,H,G,B]
    return scale.squeeze(-1), x_int


def _symmetric_dequantize(scale: torch.Tensor, x_int: torch.Tensor) -> torch.Tensor:
    """对称逐组反量化：``x = x_int * scale``（dense 形式）。"""
    T, H, G, B = x_int.shape
    return (x_int.float() * scale.unsqueeze(-1)).reshape(T, H, G * B)


def _symmetric_quantize_flat(
    x: torch.Tensor, bits: int, group_size: int
) -> tuple:
    """对称逐组整数量化（flat 形式，用于稀疏化的 compacted residual）。

    Args:
        x: ``[N, D]``。

    Returns:
        scale ``[N, D//group_size]``；x_int ``[N, D//group_size, group_size]``（int8）。
    """
    N, D = x.shape
    if D % group_size != 0:
        group_size = D
    qmax = float((1 << (bits - 1)) - 1)

    xg = x.reshape(N, D // group_size, group_size)
    scale = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax  # [N,G,1]
    x_int = (xg / scale).round().clamp(-qmax, qmax).to(torch.int8)      # [N,G,B]
    return scale.squeeze(-1), x_int


def _symmetric_dequantize_flat(scale: torch.Tensor, x_int: torch.Tensor) -> torch.Tensor:
    """对称逐组反量化（flat 形式）：``[N,G] x [N,G,B] -> [N,D]``。"""
    N, G, B = x_int.shape
    return (x_int.float() * scale.unsqueeze(-1)).reshape(N, G * B)


def _residual_keep_mask(residual: torch.Tensor, ratio) -> torch.Tensor:
    """计算残差稀疏化的保留掩码（按 L2 幅值 + 每 head 保留比例）。

    Args:
        residual: ``[T, H, D]``（SAS 之后的最终残差）。
        ratio: 保留比例，标量或 ``[H]``（每 head 一个比例，0~1）。

    Returns:
        keep_mask ``[T, H]``（bool，True 表示保留残差）。
    """
    T, H, _ = residual.shape
    mag = residual.norm(dim=-1)  # [T, H]  L2 范数
    ratio = torch.as_tensor(ratio, device=residual.device, dtype=torch.float32)
    if ratio.dim() == 0:
        ratio = ratio.expand(H)
    k = (ratio * T).round().long().clamp(min=1, max=T)  # [H]
    sorted_mag, _ = torch.sort(mag, dim=0, descending=True)  # [T, H]
    thr = sorted_mag[k - 1, torch.arange(H, device=residual.device)]  # [H]
    return mag >= thr.unsqueeze(0)


def quanevict_ratios(w, ratio_base, gamma=1.0, eps=1e-12):
    """QuanEvict：用敏感度给每个 (layer, head) 加权分配 KV bank 预算。

    引入敏感度后总 bank 大小保持不变（sum(ratio) == ratio_base * numel(w)），
    每个 (layer, head) 的保留比例正比于其敏感度：

        ratio_i = ratio_base * w_i / mean(w)

    敏感 head（w_i 大）→ ratio 大 → 保留更多；不敏感 head → ratio 小 → 驱逐更多。

    gamma 参数保留用于向后兼容，不再参与计算（旧版 ratio_base*(w_i/w_bar)^gamma
    会随 gamma 改变总 bank 大小，已废弃）。
    """
    w = torch.as_tensor(w, dtype=torch.float32)
    w = w.clamp_min(eps)
    w_bar = w.mean().clamp_min(eps)
    ratio = ratio_base * w / w_bar
    return ratio.clamp(0.0, 1.0).to(w.dtype)


def qvg_compress(
    x: torch.Tensor,
    bits: int = 2,
    num_centroids: int = 256,
    num_stages: int = 1,
    group_size: int = 64,
    kmeans_iters: int = 4,
    seed: int = 0,
    sparsify: bool = False,
    sparse_ratio: float = 0.5,
    use_triton: bool = True,
    num_frames: int = 1,
    return_residual_mag: bool = False,
) -> dict:
    """把一个完整 chunk 的 K/V 压缩成 QVG 表示（不保留浮点残差）。

    Returns:
        dict:
          - centroids: list of ``[H, K, D]``（每 stage 一个）
          - assign: list of ``[T, H]``（uint8）
          - scale: ``[T, H, D//group_size]``（非稀疏）或 ``[N, D//group_size]``（稀疏）
          - residual_int: ``[T, H, D//group_size, group_size]``（非稀疏）或
            ``[N, D//group_size, group_size]``（稀疏，int8）
          - orig_shape: 输入原始形状
          - sparse: bool
          - keep_mask: ``[T, H]``（uint8，仅稀疏时存在）
    """
    if bits is None or bits <= 0 or x.numel() == 0:
        return None

    orig_shape = x.shape
    H = x.shape[-2]
    D = x.shape[-1]
    xf = x.reshape(-1, H, D).float()

    residual = xf
    centroids_list = []
    assign_list = []
    for t in range(num_stages):
        centroids, assign = _kmeans_batched(
            residual, num_centroids, kmeans_iters, seed + t, use_triton=use_triton)
        residual = residual - _gather_centroids(centroids, assign)
        centroids_list.append(centroids)
        assign_list.append(assign.to(torch.uint8))

    residual_mag = residual.norm(dim=-1) if return_residual_mag else None
    extra = {'residual_mag': residual_mag} if return_residual_mag else {}

    if sparsify:
        keep_mask = _residual_keep_mask(residual, sparse_ratio)  # [T,H] bool
        keep_idx = keep_mask.reshape(-1).nonzero().squeeze(1).to(torch.int32)  # [N] token*H+head
        residual_kept = residual.reshape(-1, D)[keep_idx.long()]    # [N,D]
        scale, x_int = _symmetric_quantize_flat(residual_kept, bits, group_size)
        frame_offsets = _frame_offsets_from_mask(keep_mask, num_frames)
        return {
            'centroids': centroids_list,
            'assign': assign_list,
            'scale': scale,
            'residual_int': x_int,
            'keep_idx': keep_idx,
            'frame_offsets': frame_offsets,
            'num_frames': num_frames,
            'orig_shape': orig_shape,
            'sparse': True,
            **extra,
        }

    scale, x_int = _symmetric_quantize(residual, bits, group_size)
    return {
        'centroids': centroids_list,
        'assign': assign_list,
        'scale': scale,
        'residual_int': x_int,
        'orig_shape': orig_shape,
        'sparse': False,
        **extra,
    }


def _frame_offsets_from_mask(keep_mask: torch.Tensor, num_frames: int) -> torch.Tensor:
    """按帧累计保留的 (token, head) 数，得到每帧在 compact 维度的 ``[start, end)``。

    ``keep_mask`` 展平顺序为 token 在前、head 在后（即 ``token*H + head``），
    因此 ``keep_idx`` 天然按帧分组；这里只需求每帧保留项个数的前缀和。
    """
    T, H = keep_mask.shape
    frame_seqlen = T // num_frames
    per_frame = keep_mask.reshape(num_frames, frame_seqlen, H).sum(dim=(1, 2)).to(torch.int32)
    return torch.cat([torch.zeros(1, dtype=torch.int32, device=keep_mask.device),
                      per_frame.cumsum(0)])  # [num_frames + 1]


def _decompress_sparse_frames(packed: dict, frame_ids, out_dtype=None, use_triton: bool = True, out=None) -> torch.Tensor:
    """稀疏形式按帧反量化，只恢复 ``frame_ids`` 指定的帧。

    支持直接写入预分配的 ``out``（``[len(frame_ids) * frame_seqlen, H, D]``），
    避免逐帧 fp32 中间张量的 cat 与整块 ``.to(out_dtype)`` 拷贝。

    Returns:
        ``[len(frame_ids) * frame_seqlen, H, D]``（按 ``frame_ids`` 顺序拼接）。
    """
    centroids = packed['centroids']
    assign = packed['assign']
    keep_idx = packed['keep_idx']            # [N] int32，token*H+head
    frame_offsets = packed['frame_offsets']  # [num_frames+1] int32
    scale = packed['scale']                  # [N,G]
    residual_int = packed['residual_int']    # [N,G,B]
    num_frames = packed['num_frames']

    T, H = assign[-1].shape
    frame_seqlen = T // num_frames
    G, B = residual_int.shape[1], residual_int.shape[2]
    D = G * B

    frame_ids = [int(f) for f in frame_ids]
    n_frames = len(frame_ids)

    # 多级残差：先对除最后一级外的质心求和（默认 num_stages==1，此循环为空）
    base = None
    for t in range(len(centroids) - 1):
        g = _gather_centroids(centroids[t], assign[t].long())
        base = g if base is None else base + g

    last_c = centroids[-1]
    last_a = assign[-1].long()

    if out is None:
        dtype = out_dtype if out_dtype is not None else last_c.dtype
        out = torch.empty([n_frames * frame_seqlen, H, D], device=last_c.device, dtype=dtype)

    # 融合解压路径：gather 质心 + scatter 残差 + 加 base + cast 一次完成，
    # 需要 Triton + CUDA + D 为 2 的幂（三个内核的 BLOCK_D 均为 D）。
    use_fused = (
        use_triton
        and _qvt.has_triton()
        and last_c.is_cuda
        and D >= 16
        and (D & (D - 1)) == 0
    )
    if use_fused:
        scratch = torch.empty([frame_seqlen, H, D], device=last_c.device, dtype=torch.float32)
        last_a32 = last_a.to(torch.int32)

    for i, f in enumerate(frame_ids):
        lo = int(frame_offsets[f])
        hi = int(frame_offsets[f + 1])
        seg_idx = keep_idx[lo:hi]                                  # [N_f]
        token_local = (seg_idx // H) - f * frame_seqlen            # [N_f] 帧内 token
        head_idx = seg_idx % H                                     # [N_f]
        flat_idx = token_local * H + head_idx

        if use_fused:
            _qvt.decompress_frame_fused(
                last_c,
                last_a32[f * frame_seqlen:(f + 1) * frame_seqlen],
                residual_int[lo:hi],
                scale[lo:hi],
                flat_idx,
                out[i * frame_seqlen:(i + 1) * frame_seqlen],
                scratch,
                base[f * frame_seqlen:(f + 1) * frame_seqlen] if base is not None else None,
            )
            continue

        c = _gather_centroids(last_c, last_a[f * frame_seqlen:(f + 1) * frame_seqlen])  # [frame_seqlen,H,D]
        c_flat = c.reshape(-1, D)
        if use_triton:
            _qvt.scatter_residual(residual_int[lo:hi], scale[lo:hi], flat_idx, c_flat)
        else:
            resid = _symmetric_dequantize_flat(scale[lo:hi], residual_int[lo:hi])
            c_flat.index_add_(0, flat_idx.long(), resid)
        if base is not None:
            c = c + base[f * frame_seqlen:(f + 1) * frame_seqlen]
        out[i * frame_seqlen:(i + 1) * frame_seqlen] = c

    return out


def qvg_decompress(packed: dict, out_dtype=None, use_triton: bool = True) -> torch.Tensor:
    """从 ``qvg_compress`` 的压缩表示恢复浮点 K/V（残差反量化 + 逐级加回质心）。"""
    if packed.get('sparse', False):
        num_frames = packed.get('num_frames', 1)
        x = _decompress_sparse_frames(packed, range(num_frames), out_dtype, use_triton)
        return x.reshape(packed['orig_shape'])

    x = _symmetric_dequantize(packed['scale'], packed['residual_int'])
    for t in range(len(packed['centroids']) - 1, -1, -1):
        x = x + _gather_centroids(packed['centroids'][t], packed['assign'][t].long())
    x = x.reshape(packed['orig_shape'])
    if out_dtype is not None:
        x = x.to(out_dtype)
    return x


def qvg_decompress_frames(packed: dict, frame_ids, out_dtype=None, use_triton: bool = True, out=None) -> torch.Tensor:
    """只反量化 ``frame_ids`` 指定的帧（帧级稀疏检索用）。

    检索到的若干帧可能分布在不同的压缩 chunk 中，本函数只恢复目标帧，避免
    解压整个 chunk 的访存与计算开销。返回 ``[len(frame_ids) * frame_seqlen, H, D]``。
    提供 ``out`` 时直接写入该预分配 buffer（形状同上）。
    """
    if not packed.get('sparse', False):
        raise ValueError('qvg_decompress_frames 仅支持 sparse=True 的 packed dict')
    return _decompress_sparse_frames(packed, frame_ids, out_dtype, use_triton, out=out)

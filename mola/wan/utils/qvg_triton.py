"""Triton 加速内核 for QVG（k-means 距离 + 残差稀疏化解压）。

本模块提供两个可选的 Triton 内核，均带有纯 PyTorch 回退：

1. ``_pairwise_dot``: 计算 ``x @ c^T``（``[T,H,D] x [H,K,D] -> [T,H,K]``），
   用于 k-means 距离计算的 GEMM 项。当 Triton 不可用、输入不在 CUDA 上、
   或 ``D`` 不是 2 的幂时，回退到 ``torch.einsum``。

2. ``scatter_residual``: 把稀疏保留的残差反量化后 scatter-add 回质心
   ``c_flat``，供帧级解压 ``qvg_decompress_frames`` 使用。回退路径用
   ``index_add_`` 完成 scatter-add。

所有内核只做前向推理用到的算子，不引入梯度。
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - 环境未安装 triton
    triton = None
    tl = None
    _HAS_TRITON = False


def has_triton() -> bool:
    return _HAS_TRITON


if _HAS_TRITON:

    @triton.jit
    def _pairwise_dot_kernel(
        x_ptr, c_ptr, out_ptr,
        T, H, K, D,
        BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """计算 ``out[t,h,k] = sum_d x[t,h,d] * c[h,k,d]``。

        布局均为行优先连续：x ``[T,H,D]``、c ``[H,K,D]``、out ``[T,H,K]``。
        """
        pid_t = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_h = tl.program_id(2)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)   # [BLOCK_T]
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)   # [BLOCK_K]
        offs_d = tl.arange(0, BLOCK_D)                      # [BLOCK_D]

        # x tile [BLOCK_T, BLOCK_D]
        x_ptrs = x_ptr + offs_t[:, None] * (H * D) + pid_h * D + offs_d[None, :]
        x_tile = tl.load(x_ptrs, mask=offs_t[:, None] < T, other=0.0)

        # c tile [BLOCK_K, BLOCK_D]，随后转置为 [BLOCK_D, BLOCK_K]
        c_ptrs = c_ptr + pid_h * (K * D) + offs_k[:, None] * D + offs_d[None, :]
        c_tile = tl.load(c_ptrs, mask=offs_k[:, None] < K, other=0.0)

        acc = tl.dot(x_tile, tl.trans(c_tile), input_precision="ieee")  # [BLOCK_T, BLOCK_K]

        out_ptrs = out_ptr + offs_t[:, None] * (H * K) + pid_h * K + offs_k[None, :]
        tl.store(out_ptrs, acc, mask=(offs_t[:, None] < T) & (offs_k[None, :] < K))

    @triton.jit
    def _scatter_residual_kernel(
        residual_ptr,   # [N,G,B] int8
        scale_ptr,      # [N,G] fp32
        idx_ptr,        # [N] int32，目标行（帧内 flatten token*H+head）
        out_ptr,        # [frame_seqlen*H, D] fp32
        N, D,
        G: tl.constexpr, B: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """``out[idx[i]] += dequant(residual[i], scale[i])``（原地 scatter-add）。"""
        pid = tl.program_id(0)

        offs_g = tl.arange(0, G)
        offs_b = tl.arange(0, B)
        r_int = tl.load(
            residual_ptr + pid * (G * B) + offs_g[:, None] * B + offs_b[None, :])
        sc = tl.load(scale_ptr + pid * G + offs_g)
        r = r_int.to(tl.float32) * sc[:, None]          # [G, B]
        r_flat = tl.reshape(r, (BLOCK_D,))              # [D]

        tgt = tl.load(idx_ptr + pid).to(tl.int32)
        offs_d = tl.arange(0, BLOCK_D)
        tl.atomic_add(out_ptr + tgt * D + offs_d, r_flat)


    @triton.jit
    def _gather_centroids_kernel(
        centroids_ptr,   # [H, K, D] fp32
        assign_ptr,      # [frame_seqlen, H] int32
        scratch_ptr,     # [frame_seqlen, H, D] fp32
        H, K, D,
        BLOCK_D: tl.constexpr,
    ):
        """``scratch[t,h,:] = centroids[h, assign[t,h], :]``（纯索引拷贝，无算术）。"""
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        a = tl.load(assign_ptr + pid_t * H + pid_h)
        offs_d = tl.arange(0, BLOCK_D)
        c = tl.load(centroids_ptr + pid_h * K * D + a * D + offs_d)
        tl.store(scratch_ptr + pid_t * H * D + pid_h * D + offs_d, c)

    @triton.jit
    def _add_base_cast_kernel(
        scratch_ptr,     # [frame_seqlen, H, D] fp32
        base_ptr,        # [frame_seqlen, H, D] fp32（HAS_BASE=False 时不访问）
        out_ptr,         # [frame_seqlen, H, D] out_dtype
        H, D,
        HAS_BASE: tl.constexpr,
        OUT_BF16: tl.constexpr,
        OUT_FP16: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """``out[t,h,:] = cast(scratch[t,h,:] + base[t,h,:])``。"""
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)
        v = tl.load(scratch_ptr + pid_t * H * D + pid_h * D + offs_d)
        if HAS_BASE:
            v = v + tl.load(base_ptr + pid_t * H * D + pid_h * D + offs_d)
        if OUT_BF16:
            v = v.to(tl.bfloat16)
        elif OUT_FP16:
            v = v.to(tl.float16)
        tl.store(out_ptr + pid_t * H * D + pid_h * D + offs_d, v)


def _pairwise_dot(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``x @ c^T``：``[T,H,D] x [H,K,D] -> [T,H,K]``（Triton 加速，回退 einsum）。"""
    if not (_HAS_TRITON and x.is_cuda):
        return torch.einsum('thd,hkd->thk', x, c)

    T, H, D = x.shape
    K = c.shape[1]
    if D < 16 or (D & (D - 1)) != 0:
        # D 不是 2 的幂时回退，保持通用性
        return torch.einsum('thd,hkd->thk', x, c)

    BLOCK_T = 32
    BLOCK_K = 32
    BLOCK_D = D
    out = torch.empty((T, H, K), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(K, BLOCK_K), H)
    _pairwise_dot_kernel[grid](
        x, c, out, T, H, K, D, BLOCK_T, BLOCK_K, BLOCK_D)
    return out


def scatter_residual(
    residual_int: torch.Tensor,
    scale: torch.Tensor,
    flat_idx: torch.Tensor,
    c_flat: torch.Tensor,
) -> None:
    """帧级反量化 scatter-add（原地修改 ``c_flat``）。

    语义：``c_flat[flat_idx[i]] += dequant(residual_int[i], scale[i])``，
    即把稀疏保留的 ``(token, head)`` 残差加回其质心行。Triton 不可用、不在
    CUDA 上、或 ``D`` 不是 2 的幂时，回退 ``index_add_``。
    """
    N, G, B = residual_int.shape
    D = G * B

    if not (_HAS_TRITON and residual_int.is_cuda) or D < 16 or (D & (D - 1)) != 0:
        resid = (residual_int.float() * scale.unsqueeze(-1)).reshape(N, D)
        c_flat.index_add_(0, flat_idx.long(), resid)
        return

    _scatter_residual_kernel[(N,)](
        residual_int, scale, flat_idx, c_flat, N, D,
        G=G, B=B, BLOCK_D=D)

def decompress_frame_fused(
    centroids: torch.Tensor,
    assign: torch.Tensor,
    residual_int: torch.Tensor,
    scale: torch.Tensor,
    flat_idx: torch.Tensor,
    out: torch.Tensor,
    scratch: torch.Tensor,
    base: torch.Tensor = None,
) -> None:
    """融合帧级反量化：gather 质心 + 反量化 scatter + 加 base + cast 到 ``out``。

    ``scratch``（fp32，``[frame_seqlen, H, D]``）由调用方预分配并跨帧复用；三步
    都在 scratch 上完成，最后一步把 ``scratch + base`` 直接 cast 写入 ``out``，
    避免逐帧分配 fp32 中间张量以及 ``c + base`` 的额外拷贝。

    注意：调用方需保证已满足 Triton 可用、输入在 CUDA 上、且 ``D`` 为 2 的幂。
    """
    H, K, D = centroids.shape
    frame_seqlen = assign.shape[0]
    assign32 = assign if assign.dtype == torch.int32 else assign.to(torch.int32)
    grid = (frame_seqlen, H)

    # 1) gather 质心 -> scratch（纯索引，无算术）
    _gather_centroids_kernel[grid](
        centroids, assign32, scratch, H, K, D, BLOCK_D=D)

    # 2) 反量化残差 scatter-add -> scratch（复用现有原子加内核）
    scatter_residual(residual_int, scale, flat_idx, scratch.reshape(-1, D))

    # 3) scratch + base -> cast -> out
    base_arg = base if base is not None else scratch  # HAS_BASE=False 时不访问
    _add_base_cast_kernel[grid](
        scratch, base_arg, out, H, D,
        HAS_BASE=base is not None,
        OUT_BF16=out.dtype == torch.bfloat16,
        OUT_FP16=out.dtype == torch.float16,
        BLOCK_D=D)



if _HAS_TRITON:

    @triton.jit
    def _kmeans_scatter_update_kernel(
        x_ptr, assign_ptr, sums_ptr, counts_ptr,
        T, H, K, D,
        BLOCK_D: tl.constexpr,
    ):
        """sums[h,assign[t,h]] += x[t,h]，counts[h,assign[t,h]] += 1（替代 onehot+einsum）。"""
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_d = tl.arange(0, BLOCK_D)
        x_t = tl.load(x_ptr + pid_t * (H * D) + pid_h * D + offs_d)
        k = tl.load(assign_ptr + pid_t * H + pid_h)

        tl.atomic_add(sums_ptr + pid_h * (K * D) + k * D + offs_d, x_t)
        tl.atomic_add(counts_ptr + pid_h * K + k, 1.0)


def kmeans_scatter_update(
    x: torch.Tensor,
    assign: torch.Tensor,
    sums: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    """Triton 版质心累加：sums[h,assign[t,h]] += x[t,h]，counts 同步 +1。

    sums ``[H, K, D]`` 与 counts ``[H, K]`` 由调用方预分配并清零；``D`` 须为 2 的幂。
    """
    T, H, D = x.shape
    K = sums.shape[1]
    _kmeans_scatter_update_kernel[(T, H)](x, assign, sums, counts, T, H, K, D, BLOCK_D=D)


if _HAS_TRITON:

    @triton.jit
    def _kmeans_assign_kernel(
        x_ptr, c_ptr, assign_ptr,
        T, H, K, D,
        BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """融合 k-means 距离 + argmin：只写出 [T,H] 的 assign，不落地 [T,H,K] 距离。

        dot 用 tf32 tensor core（QVG 后续还有 2bit 量化，距离微差可忽略）。
        """
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)

        x_ptrs = x_ptr + offs_t[:, None] * (H * D) + pid_h * D + offs_d[None, :]
        x_tile = tl.load(x_ptrs, mask=offs_t[:, None] < T, other=0.0)
        x_norm = tl.sum(x_tile * x_tile, axis=1)          # [BLOCK_T]

        INF = 1e30
        best_dist = tl.full((BLOCK_T,), INF, dtype=tl.float32)
        best_idx = tl.zeros((BLOCK_T,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            c_ptrs = c_ptr + pid_h * (K * D) + offs_k[:, None] * D + offs_d[None, :]
            c_tile = tl.load(c_ptrs, mask=offs_k[:, None] < K, other=0.0)
            dot = tl.dot(x_tile, tl.trans(c_tile), input_precision="tf32")
            c_norm = tl.sum(c_tile * c_tile, axis=1)       # [BLOCK_K]
            dist = x_norm[:, None] - 2.0 * dot + c_norm[None, :]
            dist = tl.where((offs_k < K)[None, :], dist, INF)

            local_idx = tl.argmin(dist, axis=1)            # [BLOCK_T]
            local_min = tl.min(dist, axis=1)                # [BLOCK_T]
            global_idx = local_idx + k0
            update = local_min < best_dist
            best_dist = tl.where(update, local_min, best_dist)
            best_idx = tl.where(update, global_idx, best_idx)

        tl.store(assign_ptr + offs_t * H + pid_h, best_idx, mask=offs_t < T)


def kmeans_assign(
    x: torch.Tensor,
    c: torch.Tensor,
    BLOCK_T: int = 32,
    BLOCK_K: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Triton 融合版 k-means 距离 + argmin：返回 assign ``[T, H]`` (int32)。

    相比 ``einsum + argmin``，避免落地 ``[T,H,K]`` 距离张量，dot 用 tf32 tensor core。
    """
    T, H, D = x.shape
    K = c.shape[1]
    assign = torch.empty((T, H), device=x.device, dtype=torch.int32)
    grid = (triton.cdiv(T, BLOCK_T), H)
    _kmeans_assign_kernel[grid](
        x, c, assign, T, H, K, D,
        BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K, BLOCK_D=D,
        num_warps=num_warps,
    )
    return assign


if _HAS_TRITON:

    @triton.jit
    def _fastar_top1_kernel(
        kf_ptr, bank_ptr, maxsim_ptr, argmax_ptr,
        S, G,
        NHEADS: tl.constexpr, D: tl.constexpr,
        BLOCK_S: tl.constexpr, BLOCK_G: tl.constexpr,
    ):
        """对每个空间 token s，求 bank 中 head-mean 余弦最大的代表 token 全局索引。

        kf ``[S, NHEADS, D]``、bank ``[G, NHEADS, D]`` 均已归一化（d 维 L2）。
        逐 head 用 ``tl.dot`` 算内积并累加，除以 ``NHEADS`` 得 head-mean 余弦；
        对 G 维分块跑 running max/argmax，全程不落地 ``[S, G, NHEADS]`` 中间张量。
        """
        pid = tl.program_id(0)
        offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)   # [BLOCK_S]
        offs_d = tl.arange(0, D)                          # [D]
        s_mask = offs_s < S

        NEG_INF = -1e30
        best = tl.full([BLOCK_S], NEG_INF, tl.float32)
        best_idx = tl.zeros([BLOCK_S], tl.int64)

        for g0 in range(0, G, BLOCK_G):
            offs_g = g0 + tl.arange(0, BLOCK_G)           # [BLOCK_G]
            g_mask = offs_g < G

            acc = tl.zeros([BLOCK_S, BLOCK_G], tl.float32)
            for h in range(0, NHEADS):
                kf_h = tl.load(
                    kf_ptr + offs_s[:, None] * (NHEADS * D) + h * D + offs_d[None, :],
                    mask=s_mask[:, None], other=0.0)      # [BLOCK_S, D]
                bank_h = tl.load(
                    bank_ptr + offs_g[:, None] * (NHEADS * D) + h * D + offs_d[None, :],
                    mask=g_mask[:, None], other=0.0)      # [BLOCK_G, D]
                acc += tl.dot(kf_h, tl.trans(bank_h), input_precision="ieee")

            sim = acc / NHEADS                            # [BLOCK_S, BLOCK_G]
            sim = tl.where(g_mask[None, :], sim, NEG_INF)

            local_max = tl.max(sim, axis=1)               # [BLOCK_S]
            local_arg = tl.argmax(sim, axis=1)            # [BLOCK_S] int32
            update = local_max > best
            best = tl.where(update, local_max, best)
            best_idx = tl.where(update, (g0 + local_arg).to(tl.int64), best_idx)

        tl.store(maxsim_ptr + offs_s, best, mask=s_mask)
        tl.store(argmax_ptr + offs_s, best_idx, mask=s_mask)


def fastar_top1_merge(
    kf_n: torch.Tensor,
    bank_k_n: torch.Tensor,
    BLOCK_S: int = 16,
    BLOCK_G: int = 64,
    num_warps: int = 4,
):
    """对每个空间 token s，在全部代表 token 中找 head-mean 余弦 top-1。

    kf_n ``[S, n, d]``、bank_k_n ``[G, n, d]``，均已在最后一维 L2 归一化。
    返回 ``(max_sim [S] fp32, argmax [S] int64)``。Triton 不可用或不在 CUDA 上时，
    回退到分块 ``einsum``（与旧实现等价，避免一次性 ``[S, G, n]`` 中间张量 OOM）。
    """
    S, n, d = kf_n.shape
    G = bank_k_n.shape[0]

    if not (_HAS_TRITON and kf_n.is_cuda) or G == 0:
        max_sim = torch.full([S], -float('inf'), device=kf_n.device, dtype=torch.float32)
        argmax = torch.zeros([S], device=kf_n.device, dtype=torch.int64)
        chunk = 1024
        for start in range(0, G, chunk):
            end = min(start + chunk, G)
            blk = bank_k_n[start:end]
            sim_blk = torch.einsum('snd,gnd->sgn', kf_n, blk).mean(dim=-1)
            blk_max, blk_arg = sim_blk.max(dim=1)
            better = blk_max > max_sim
            max_sim[better] = blk_max[better]
            argmax[better] = start + blk_arg[better]
        return max_sim, argmax

    max_sim = torch.empty([S], device=kf_n.device, dtype=torch.float32)
    argmax = torch.empty([S], device=kf_n.device, dtype=torch.int64)
    grid = (triton.cdiv(S, BLOCK_S),)
    _fastar_top1_kernel[grid](
        kf_n, bank_k_n, max_sim, argmax, S, G,
        NHEADS=n, D=d,
        BLOCK_S=BLOCK_S, BLOCK_G=BLOCK_G,
        num_warps=num_warps,
    )
    return max_sim, argmax

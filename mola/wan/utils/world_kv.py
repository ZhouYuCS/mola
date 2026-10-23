"""轨迹稀疏注意力辅助函数（训练无关）。

把离散动作（WASD + 鼠标 yaw/pitch）累积成伪位姿向量，用于判断当前 chunk 与
历史 chunk 在运动轨迹上的接近程度（回访检测），从而实现稀疏注意力。

参考论文 WorldKV 与 lingbot 实现（wan/modules/model_fast.py）。
"""
import torch


def build_action_state(keyboard_cond, mouse_cond=None, num_latent_frames=None,
                       vae_time_compression_ratio=4):
    """把 RGB 帧级的离散动作累积成每个 latent 帧的伪位姿向量。

    keyboard_cond : [B, num_rgb_frames, K]，K 维 one-hot（universal 为 W,S,A,D）
    mouse_cond    : [B, num_rgb_frames, 2] 或 None（templerun 无鼠标）
    返回          : [B, num_latent_frames, 4] = [cum_fwd, cum_side, cum_pitch, cum_yaw]
    """
    B = keyboard_cond.shape[0]
    if num_latent_frames is None:
        num_latent_frames = (keyboard_cond.shape[1] - 1) // vae_time_compression_ratio + 1

    device = keyboard_cond.device
    cum = torch.zeros(B, 4, device=device, dtype=torch.float32)
    states = []
    for i in range(num_latent_frames):
        rgb_idx = min(i * vae_time_compression_ratio, keyboard_cond.shape[1] - 1)
        kb = keyboard_cond[:, rgb_idx].float()
        # universal keyboard: [W, S, A, D]；用 W-S 表示前后、D-A 表示左右
        cum[:, 0] += (kb[:, 0] - kb[:, 1])
        cum[:, 1] += (kb[:, 3] - kb[:, 2])
        if mouse_cond is not None:
            m = mouse_cond[:, rgb_idx].float()  # [pitch, yaw]
            cum[:, 2] += m[:, 0]
            cum[:, 3] += m[:, 1]
        states.append(cum.clone())
    return torch.stack(states, dim=1)  # [B, num_latent_frames, 4]

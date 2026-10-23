import os
import argparse
import torch
import numpy as np
import copy

from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image

from pipeline import CausalInferenceStreamingPipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.visualize import process_video
from utils.misc import set_seed
from utils.conditions import *
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml", help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the checkpoint")
    parser.add_argument("--output_folder", type=str, default="outputs/", help="Output folder")
    parser.add_argument("--img_path", type=str, default="demo_images/universal/0000.png", help="输入图像路径（非交互式）")
    parser.add_argument("--max_num_output_frames", type=int, default=360,
                        help="Max number of output latent frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0", help="Path to the VAE model folder")
    # QVG 量化（Quant VideoGen）
    parser.add_argument("--qvg_bits", type=int, default=2, help="QVG 对称量化比特（0 表示不量化，默认 2-bit）")
    parser.add_argument("--qvg_num_centroids", type=int, default=256, help="QVG PRQ 每级质心数 K")
    parser.add_argument("--qvg_num_stages", type=int, default=1, help="QVG PRQ 残差级数 S")
    parser.add_argument("--qvg_group_size", type=int, default=64, help="QVG 对称逐组量化组大小 B")
    parser.add_argument("--qvg_kmeans_iters", type=int, default=4, help="QVG k-means 迭代次数")
    parser.add_argument("--qvg_seed", type=int, default=0, help="QVG k-means 随机种子")
    parser.add_argument("--qvg_sparsify", action=argparse.BooleanOptionalAction, default=True, help="QVG 残差稀疏化（低幅值残差不存储，读取时用质心近似）")
    parser.add_argument("--qvg_sparse_ratio", type=float, default=0.5, help="每 head 保留比例（0~1；QuanEvict 下为基准比例 ratio_base）")
    parser.add_argument("--qvg_use_triton", action=argparse.BooleanOptionalAction, default=True, help="使用 Triton 加速 k-means 距离与稀疏解压")
    parser.add_argument("--qvg_sensitivity_wk", type=str, default=None, help="QuanEvict：视觉 K 敏感度 npy 路径（[L,H]；None 表示统一比例）")
    parser.add_argument("--qvg_sensitivity_wv", type=str, default=None, help="QuanEvict：视觉 V 敏感度 npy 路径（[L,H]；None 表示统一比例）")
    parser.add_argument("--qvg_gamma", type=float, default=1.0, help="QuanEvict：保留比例随敏感度变化的强度（0=统一比例）")
    parser.add_argument("--batch_size", type=int, default=1, help="并行批大小；>1 时重复当前输入到该 batch size")
    parser.add_argument("--cache_transfer", action=argparse.BooleanOptionalAction, default=False,
                        help="将 KV cache 统一搬迁到 CPU；False 时缓存全程保留在 GPU")
    parser.add_argument("--fastar_lsh_tables", type=int, default=4,
                        help="fastar_full 的 LSH 哈希表数量（SimHash 多表探测；0 表示关闭 LSH，退回精确暴力 top-1）")
    parser.add_argument("--fastar_lsh_bits", type=int, default=8,
                        help="fastar_full 的 LSH 每张表哈希位数（SimHash sign random projection 桶编码）")
    parser.add_argument("--fastar_lsh_seed", type=int, default=0,
                        help="fastar_full 的 LSH 随机投影矩阵种子")
    parser.add_argument("--fastar_lsh_bucket_cap", type=int, default=8,
                        help="FAST-AR LSH 候选桶过滤上限；越小越省显存，默认 8 以降低峰值 OOM 风险")
    parser.add_argument("--residual_mag_out", type=str, default=None, help="保存残差幅值 mag 的路径（阈值驱逐率实验用；None 表示不收集）")
    # 轨迹稀疏注意力（基于动作累积伪位姿的回访检测）
    parser.add_argument("--use_retrieval", action="store_true", help="启用轨迹稀疏注意力（回访检测）")
    parser.add_argument("--retrieval_topk_frames", type=int, default=4, help="帧级位姿检索时最多开放的远处 latent 帧数（默认 4）")
    parser.add_argument("--traj_threshold", type=float, default=None, help="轨迹距离阈值；None 表示退化为 top-k")
    parser.add_argument("--kv_bank_on_gpu", action=argparse.BooleanOptionalAction, default=True, help="KV bank 存 GPU（默认 CPU）")
    parser.add_argument("--quant_chunk_frames", type=int, default=12, help="每个压缩 chunk 的 latent 帧数（需为 num_frame_per_block 整数倍）")
    parser.add_argument("-worldkv", "--worldkv", action="store_true", help="使用原始 WorldKV 做法：帧级轨迹检索 + 原始 KV（不引入 QVG 量化）")
    args = parser.parse_args()
    return args


def build_revisit_trajectory(mode, num_blocks, device):
    """生成带回访的动作轨迹（块级），用于非交互式脚本化推理。

    轨迹让伪位姿在后期回到早期位置附近，从而触发帧级回访检索。
    返回 list[dict]，元素结构与 get_current_action 一致：
      {"keyboard": tensor[int64], "mouse": tensor[float32]}。
    """
    assert mode == 'universal', '回访轨迹当前仅支持 universal 模式'

    KB = {
        'forward': [1, 0, 0, 0],
        'back':    [0, 1, 0, 0],
        'left':    [0, 0, 1, 0],
        'right':   [0, 0, 0, 1],
        'stop':    [0, 0, 0, 0],
    }
    CAM = 0.1
    MOUSE = {
        'stop':  [0.0, 0.0],
        'left':  [0.0, -CAM],
        'right': [0.0, CAM],
        'up':    [CAM, 0.0],
        'down':  [-CAM, 0.0],
    }

    # 阶段：(keyboard, mouse, 占比)；前进 -> 右转前进 -> 后退 -> 左转，形成回环
    phases = [
        ('forward', 'stop',  0.25),
        ('forward', 'right', 0.10),
        ('back',    'stop',  0.25),
        ('back',    'left',  0.10),
        ('forward', 'stop',  0.15),
        ('forward', 'right', 0.15),
    ]
    actions = []
    for kb_name, mouse_name, frac in phases:
        for _ in range(max(1, round(num_blocks * frac))):
            actions.append({
                'keyboard': torch.tensor(KB[kb_name], device=device, dtype=torch.int64),
                'mouse': torch.tensor(MOUSE[mouse_name], device=device, dtype=torch.float32),
            })
    while len(actions) < num_blocks:
        actions.append({
            'keyboard': torch.tensor(KB['stop'], device=device, dtype=torch.int64),
            'mouse': torch.tensor(MOUSE['stop'], device=device, dtype=torch.float32),
        })
    return actions[:num_blocks]


class InteractiveGameInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        self.config = OmegaConf.load(self.args.config_path)

    def _init_models(self):
        # Initialize pipeline
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        pipeline = CausalInferenceStreamingPipeline(self.config, generator=generator, vae_decoder=current_vae_decoder)
        if self.args.checkpoint_path:
            print("Loading Pretrained Model...")
            state_dict = load_file(self.args.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        # QVG 量化配置
        m = self.pipeline.generator.model
        m.qvg_bits = self.args.qvg_bits
        m.qvg_num_centroids = self.args.qvg_num_centroids
        m.qvg_num_stages = self.args.qvg_num_stages
        m.qvg_group_size = self.args.qvg_group_size
        m.qvg_kmeans_iters = self.args.qvg_kmeans_iters
        m.qvg_seed = self.args.qvg_seed
        m.qvg_sparsify = self.args.qvg_sparsify
        m.qvg_sparse_ratio = self.args.qvg_sparse_ratio
        m.qvg_use_triton = self.args.qvg_use_triton
        m.qvg_sensitivity_wk = self.args.qvg_sensitivity_wk
        m.qvg_sensitivity_wv = self.args.qvg_sensitivity_wv
        m.qvg_gamma = self.args.qvg_gamma

        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image
    
    def generate_videos(self, mode='universal'):
        assert mode in ['universal', 'gta_drive', 'templerun']
        
        img_path = self.args.img_path
        image = load_image(img_path)

        image = self._resizecrop(image, 352, 640)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
        batch_size = max(1, self.args.batch_size)
        if batch_size > 1:
            image = image.repeat(batch_size, 1, 1, 1, 1)
        # Encode the input image as the first latent
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.args.max_num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1) 
        visual_context = self.vae.clip.encode_video(image)
        # 编码完成后释放 encoder/clip/decoder 包装，降低显存（后续解码只用 current_vae_decoder）
        del self.vae
        torch.cuda.empty_cache()
        sampled_noise = torch.randn(
            [batch_size, 16, self.args.max_num_output_frames, 44, 80], device=self.device, dtype=self.weight_dtype
        )
        num_frames = (self.args.max_num_output_frames - 1) * 4 + 1
        
        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
        }
        
        if mode == 'universal':
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        elif mode == 'gta_drive':
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)
        keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).expand(batch_size, -1, -1).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict['keyboard_cond'] = keyboard_condition
        
        num_blocks = self.args.max_num_output_frames // self.pipeline.num_frame_per_block
        scripted_actions = build_revisit_trajectory(mode, num_blocks, self.device) if mode == 'universal' else None

        residual_mag_sink = [] if self.args.residual_mag_out else None
        with torch.no_grad():
            videos = self.pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=False,
                output_folder=self.args.output_folder,
                name=os.path.basename(img_path),
                mode=mode,
                scripted_actions=scripted_actions,
                use_retrieval=self.args.use_retrieval,
                retrieval_topk_frames=self.args.retrieval_topk_frames,
                traj_threshold=self.args.traj_threshold,
                kv_bank_on_gpu=self.args.kv_bank_on_gpu,
                quant_chunk_frames=self.args.quant_chunk_frames,
                worldkv=self.args.worldkv,
                residual_mag_sink=residual_mag_sink,
                cache_transfer=self.args.cache_transfer,
                fastar_full=self.args.fastar_full if hasattr(self.args, 'fastar_full') else False,
                fastar_lsh_tables=self.args.fastar_lsh_tables,
                fastar_lsh_bits=self.args.fastar_lsh_bits,
                fastar_lsh_seed=self.args.fastar_lsh_seed,
                fastar_lsh_bucket_cap=self.args.fastar_lsh_bucket_cap,
            )

        if residual_mag_sink is not None:
            os.makedirs(os.path.dirname(os.path.abspath(self.args.residual_mag_out)), exist_ok=True)
            torch.save(residual_mag_sink, self.args.residual_mag_out)
            print(f"[ResidualMag] saved {len(residual_mag_sink)} records to {self.args.residual_mag_out}", flush=True)
        
def main():
    """Main entry point for video generation."""
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)
    pipeline = InteractiveGameInference(args)
    mode = pipeline.config.pop('mode')
    pipeline.generate_videos(mode)

if __name__ == "__main__":
    main()
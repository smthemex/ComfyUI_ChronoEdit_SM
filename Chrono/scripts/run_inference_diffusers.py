# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ChronoEdit Diffusers Inference Script

This script provides a command-line interface for running video editing inference
using the ChronoEdit model with the Diffusers backend.

Example usage:

# Basic usage
PYTHONPATH=$(pwd) python scripts/run_inference_diffusers.py \
    --input assets/images/input_2.png --offload_model --use-prompt-enhancer \
    --prompt "Add a sunglasses to the cat's face" \
    --output output2.mp4 --seed 42  \
    --model-path ./checkpoints/ChronoEdit-14B-Diffusers

# Basic usage with temporal reasoning
PYTHONPATH=$(pwd) python scripts/run_inference_diffusers.py \
    --enable-temporal-reasoning assets/images/input_2.png --offload_model \
    --prompt "Add a sunglasses to the cat's face"  \
    --output output_reasoning_offload.mp4 --seed 42 \
    --model-path ./checkpoints/ChronoEdit-14B-Diffusers

# Advanced usage with lora settings
PYTHONPATH=$(pwd) python scripts/run_inference_diffusers.py \
    --input assets/images/input_2.png \
    --prompt "Add a sunglasses to the cat's face"  \
    --output output_lora.mp4 \
    --num-inference-steps 8 \
    --guidance-scale 1.0 \
    --flow-shift 2.0 \
    --lora-scale 1.0 \
    --seed 42 \
    --lora-path ./checkpoints/ChronoEdit-14B/nvidia/lora/chronoedit_distill_lora_clean.safetensors \
    --model-path ./checkpoints/ChronoEdit-14B-Diffusers
"""


import os
import sys
import numpy as np
import torch
from diffusers.schedulers import UniPCMultistepScheduler
from omegaconf import OmegaConf
from diffusers import  GGUFQuantizationConfig
from diffusers.hooks import apply_group_offloading
#from diffusers import WanTransformer3DModel
from ..chronoedit_diffusers.pipeline_chronoedit import ChronoEditPipeline
from contextlib import contextmanager
from ..chronoedit_diffusers.transformer_chronoedit_ import WanTransformer3DModel
from .prompt_enhancer import load_model as load_prompt_enhancer
from .prompt_enhancer import enhance_prompt
from transformers import Qwen2_5_VLForConditionalGeneration,AutoConfig

from ..util import load_checkpoint_and_dispatch_
try:
    from transformers import Qwen3VLMoeForConditionalGeneration
    transfomer_vrsion_low=False
except:
    transfomer_vrsion_low=True
from accelerate import init_empty_weights

# try:
#     diffusers_module = sys.modules.get('diffusers')
#     if diffusers_module:
#         setattr(diffusers_module, 'WanTransformer3DModel', WanTransformer3DModel)
# except Exception as e:
#     print(f"Warning: Could not register WanTransformer3DModel with diffusers module: {e}")

# Resolution presets
RESOLUTION_PRESETS = {
    "480p": 480 * 832,
    "720p": 720 * 1280,
    "1080p": 1080 * 1920,
}

@contextmanager
def temp_patch_module_attr(module_name: str, attr_name: str, new_obj):
    mod = sys.modules.get(module_name)
    if mod is None:
        yield
        return
    had = hasattr(mod, attr_name)
    orig = getattr(mod, attr_name, None)
    setattr(mod, attr_name, new_obj)
    try:
        yield
    finally:
        if had:
            setattr(mod, attr_name, orig)
        else:
            try:
                delattr(mod, attr_name)
            except Exception:
                pass

def calculate_dimensions(image,  mod_value):
    """
    Calculate output dimensions based on resolution settings.
    
    Args:
        image: PIL Image
        mod_value: Modulo value for dimension alignment # mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]  *2
        
    Returns:
        Tuple of (width, height)
    """
    
    # Get max area from preset or override 
    target_area = 720 * 1280
    
    # Calculate dimensions maintaining aspect ratio
    if isinstance(image, torch.Tensor):
        aspect_ratio=image.shape[1] / image.shape[2]
    else:
        aspect_ratio = image.height / image.width
    calculated_height = round(np.sqrt(target_area * aspect_ratio)) // mod_value * mod_value
    calculated_width = round(np.sqrt(target_area / aspect_ratio)) // mod_value * mod_value
    
    return calculated_width, calculated_height


def load_dit_model(ckpt_path, gguf_path,dir_path):

    if gguf_path is not None:
        with temp_patch_module_attr("diffusers", "WanTransformer3DModel", WanTransformer3DModel):
            transformer = WanTransformer3DModel.from_single_file(
                gguf_path,
                config=os.path.join(dir_path, "Chrono/ChronoEdit-14B-Diffusers/transformer"),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
                )
    elif ckpt_path is not None:
        from safetensors.torch import load_file
        sd=load_file(ckpt_path,device="cpu")
        sd=replace_key(sd)
        with init_empty_weights():
            with temp_patch_module_attr("diffusers", "WanTransformer3DModel", WanTransformer3DModel):
                config=WanTransformer3DModel.load_config(os.path.join(dir_path, "Chrono/ChronoEdit-14B-Diffusers/transformer"))
                transformer=WanTransformer3DModel.from_config(config)
        model_state_dict = transformer.state_dict()
        expected_keys = set(model_state_dict.keys())
        del transformer
        # 过滤掉模型不需要的键值对
        filtered_sd = {k: v for k, v in sd.items() if k in expected_keys}

        # 检查是否有缺失的键
        missing_keys = expected_keys - set(sd.keys())
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        del sd  
        with temp_patch_module_attr("diffusers", "WanTransformer3DModel", WanTransformer3DModel): 
            transformer = WanTransformer3DModel.from_single_file(filtered_sd,config=os.path.join(dir_path, "Chrono/ChronoEdit-14B-Diffusers/transformer"),torch_dtype=torch.bfloat16)
        del filtered_sd
    vae_config = OmegaConf.load(os.path.join(dir_path, "Chrono/ChronoEdit-14B-Diffusers/vae/config.json")) 
    pipe = ChronoEditPipeline.from_pretrained(
            os.path.join(dir_path, "Chrono/ChronoEdit-14B-Diffusers"),
            image_encoder=None,
            text_encoder=None,
            transformer=transformer,
            vae=None,
            torch_dtype=torch.bfloat16,
            vae_config=vae_config,
        )
    return pipe


def load_prompt_model(prompt_enhancer_model):
    model, processor = load_prompt_enhancer(prompt_enhancer_model)
    return model, processor


def prompt_enhance(input_image,prompt,prompt_model,processor,):

    cot_prompt = enhance_prompt(
            input_image,
            prompt,
            prompt_model,
            processor,
        )

    # try:
    #     if isinstance(prompt_model, torch.nn.Module) and hasattr(prompt_model, "_hf_hook"):
    #         accelerate.hooks.remove_hook_from_module(prompt_model, recurse=True)
    #         print("Offloaded prompt model to CPU")
    # except:pass
        
    torch.cuda.empty_cache()
    return cot_prompt

def load_prompt_enhancer_cf(clip_path,repo):
    if "qwen3" in clip_path.lower() and transfomer_vrsion_low:
        raise ValueError("Qwen3 is not supported in low version transfomer")
    print(f"Loading clip text encoder from {clip_path}...")
    config=AutoConfig.from_pretrained(repo)
    with init_empty_weights():
        if "qwen3" in clip_path.lower() and  not transfomer_vrsion_low:  
            model = Qwen3VLMoeForConditionalGeneration._from_config(config)
        elif "qwen2" in clip_path.lower() or "2.5" in clip_path.lower():    
            model = Qwen2_5_VLForConditionalGeneration._from_config(config)
    if clip_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(clip_path)
    else:
        state_dict = torch.load(clip_path,weights_only=False, map_location='cpu')   
    state_dict = remap_state_dict_keys(state_dict)
    model=load_checkpoint_and_dispatch_(model, state_dict, device_map="auto",dtype=torch.bfloat16)
    del state_dict
    model.eval() 
    print("Text encoder loaded successfully")
    return model


def replace_key(t_state_dict):
    new_dict = {}
    unused_keys = {'model_sampling.sigmas'}
    for k, v in t_state_dict.items():
        if k in unused_keys:
            continue
        new_k = k
        if k.startswith("text_embedding.0."):
            new_k = k.replace("text_embedding.0.", "condition_embedder.text_embedder.linear_1.")
        elif k.startswith("text_embedding.2."):
            new_k = k.replace("text_embedding.2.", "condition_embedder.text_embedder.linear_2.")
        elif k.startswith("time_embedding.0."):
            new_k = k.replace("time_embedding.0.", "condition_embedder.time_embedder.linear_1.")
        elif k.startswith("time_embedding.2."):
            new_k = k.replace("time_embedding.2.", "condition_embedder.time_embedder.linear_2.")
        elif k.startswith("time_projection.1."):
            new_k = k.replace("time_projection.1.", "condition_embedder.time_proj.")
        elif k.startswith("img_emb.proj.1."):
            new_k = k.replace("img_emb.proj.1.", "condition_embedder.image_embedder.ff.net.0.proj.")
        elif k.startswith("img_emb.proj.3."):
            new_k = k.replace("img_emb.proj.3.", "condition_embedder.image_embedder.ff.net.2.")
        elif k.startswith("img_emb.proj.0."):
            new_k = k.replace("img_emb.proj.0.", "condition_embedder.image_embedder.norm1.")
        elif k.startswith("img_emb.proj.4."):
            new_k = k.replace("img_emb.proj.4.", "condition_embedder.image_embedder.norm2.")
        elif k.startswith("head.modulation"):
            new_k = k.replace("head.modulation", "scale_shift_table")
        elif k.startswith("head.head."):
            new_k = k.replace("head.head.", "proj_out.")
        elif k.startswith("blocks."):
            if ".ffn.0." in k:
                new_k = k.replace(".ffn.0.", ".ffn.net.0.proj.")
            elif '.ffn.2.' in k:
                new_k = k.replace('.ffn.2.', '.ffn.net.2.')
            elif '.modulation' in k:
                new_k = k.replace('.modulation', '.scale_shift_table')
            elif ".norm3." in k:
                new_k = k.replace('.norm3.', '.norm2.')
            elif "self_attn.o." in k:
                new_k = k.replace("self_attn.o.", "attn1.to_out.0.")
            elif "self_attn.q." in k:
                new_k = k.replace("self_attn.q.", "attn1.to_q.")
            elif "self_attn.k." in k:
                new_k = k.replace("self_attn.k.", "attn1.to_k.")
            elif "self_attn.v." in k:
                new_k = k.replace("self_attn.v.", "attn1.to_v.")
            elif ".self_attn.norm_k." in k:
                new_k = k.replace(".self_attn.norm_k.", ".attn1.norm_k.")
            elif ".self_attn.norm_q." in k:
                new_k = k.replace(".self_attn.norm_q.", ".attn1.norm_q.")
            elif ".cross_attn.k." in k:
                new_k = k.replace(".cross_attn.k.", ".attn2.to_k.")
            elif ".cross_attn.q." in k:
                new_k = k.replace(".cross_attn.q.", ".attn2.to_q.")
            elif ".cross_attn.v." in k:
                new_k = k.replace(".cross_attn.v.", ".attn2.to_v.")
            elif ".cross_attn.o." in k:
                new_k = k.replace(".cross_attn.o.", ".attn2.to_out.0.")
            elif ".cross_attn.norm_k." in k:
                new_k = k.replace(".cross_attn.norm_k.", ".attn2.norm_k.")
            elif ".cross_attn.norm_q." in k:
                new_k = k.replace(".cross_attn.norm_q.", ".attn2.norm_q.")
            elif ".cross_attn.norm_k_img." in k:
                new_k = k.replace(".cross_attn.norm_k_img.", ".attn2.norm_added_k.")
            elif ".cross_attn.k_img." in k:
                new_k = k.replace(".cross_attn.k_img.", ".attn2.add_k_proj.")
            elif ".cross_attn.v_img." in k:
                new_k = k.replace(".cross_attn.v_img.", ".attn2.add_v_proj.")
            else:
                new_k = k   
        else:
            new_k = k
        
        new_dict[new_k] = v
    return new_dict



def remap_state_dict_keys(state_dict):
    """Remap state dict keys to match model architecture"""
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        # 处理视觉编码器前缀
        if key.startswith('visual.') and not key.startswith('model.visual.'):
            new_key = 'model.' + key
        # 处理语言模型前缀
        elif key.startswith('model.layers.') and not key.startswith('model.language_model.'):
            new_key = 'model.language_model.' + key[len('model.'):]
        # 处理嵌入层
        elif key == 'model.embed_tokens.weight':
            new_key = 'model.language_model.embed_tokens.weight'
        elif key == 'model.norm.weight':
            new_key = 'model.language_model.norm.weight'
            
        new_state_dict[new_key] = value
    
    return new_state_dict

def inference_chrono(pipe,image_latent,positive,image_embeds,negative,flow_shift,seed,num_inference_steps,height,width,
              num_frames,num_temporal_reasoning_steps,guidance_scale=5.0,offload_model=True,block_num=1,device="cuda"):
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config,flow_shift=flow_shift)
    apply_group_offloading(pipe.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=block_num)

    generator = torch.Generator(device=device).manual_seed(seed)
    #num_frames = 29 if enable_temporal_reasoning else 5
    enable_temporal_reasoning=True if num_frames == 29 else False
    print(f"Generating {num_frames} frames with {num_inference_steps} steps...")
   
    output = pipe(
        image=None,
        prompt=None,
        negative_prompt=None,
        image_embeds=image_embeds, #torch.Size([1, 257, 1280])
        prompt_embeds=positive,
        negative_prompt_embeds=negative,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        enable_temporal_reasoning=enable_temporal_reasoning,
        num_temporal_reasoning_steps=num_temporal_reasoning_steps,
        generator=generator,
        offload_model=offload_model,
        image_latent=image_latent,
    ).frames

    return output



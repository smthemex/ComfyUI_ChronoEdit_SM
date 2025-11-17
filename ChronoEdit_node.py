 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from .Chrono.scripts.run_inference_diffusers import load_dit_model,load_prompt_enhancer,prompt_enhance,inference_chrono,load_prompt_enhancer_cf,calculate_dimensions
from .model_loader_utils import tensor_upscale,tensor2image,map_0_1_to_neg1_1,map_neg1_1_to_0_1
from .Chrono.chronoedit_diffusers.pipeline_chronoedit import retrieve_latents
import folder_paths

from einops import rearrange
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
from pathlib import PureWindowsPath
from transformers import AutoProcessor
import comfy.model_management as mm
from diffusers.models import AutoencoderKLWan
MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)

folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir


class ChronoEdit_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Model",
            display_name="ChronoEdit_SM_Model",
            category="ChronoEdit_SM",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
            ],
            outputs=[
                io.Custom("ChronoEdit_SM_Model").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf) -> io.NodeOutput:
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        model=load_dit_model(dit_path, gguf_path,node_cr_path)
        return io.NodeOutput(model)
    
class ChronoEdit_SM_Lora(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Lora",
            display_name="ChronoEdit_SM_Lora",
            category="ChronoEdit_SM",
            inputs=[
                io.Custom("ChronoEdit_SM_Model").Input("model"),
                io.Combo.Input("lora_1",options= ["none"] + folder_paths.get_filename_list("loras") ),
                io.Combo.Input("lora_2",options= ["none"] + folder_paths.get_filename_list("loras") ),
                io.Float.Input("lora_scale1", default=1, min=0.1, max=1.0,step=0.1, round=0.01,),
                io.Float.Input("lora_scale2", default=1, min=0.1, max=1.0,step=0.1, round=0.01,),
            ],
            outputs=[
                io.Custom("ChronoEdit_SM_Model").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, model,lora_1,lora_2,lora_scale1,lora_scale2) -> io.NodeOutput:
        lora_path_1=folder_paths.get_full_path("loras", lora_1) if lora_1 != "none" else None
        lora_path_2=folder_paths.get_full_path("loras", lora_2) if lora_2 != "none" else None
        lora_list=[i for i in [lora_path_1,lora_path_2] if i is not None]
    
        lora_scales=[lora_scale1,lora_scale2]
        if lora_list:
            if len(lora_list)!=len(lora_scales): #sacles  
                lora_scales = lora_scales[:1]
            all_adapters = model.get_list_adapters()
            dit_list=[]
            if all_adapters:
                dit_list= all_adapters['transformer']
            adapter_name_list=[]
            for path in lora_list:
                if path is not None:
                    name=os.path.splitext(os.path.basename(path))[0].replace(".", "_")
                    adapter_name_list.append(name)
                    if name in dit_list:
                        continue
                    model.load_lora_weights(path, adapter_name=name)
            print(f"成功加载LoRA权重: {adapter_name_list} (scale: {lora_scales})")        
            model.set_adapters(adapter_name_list, adapter_weights=lora_scales)
            try:
                active_adapters = model.get_active_adapters()
                all_adapters = model.get_list_adapters()
                print(f"当前激活的适配器: {active_adapters}")
                print(f"所有可用适配器: {all_adapters}") 
            except:
                pass
        try:
            dit_list= model.get_list_adapters()['transformer']
            for name in dit_list:
                if lora_list :
                    name_list=[os.path.splitext(os.path.basename(i))[0].replace(".", "_") for i in lora_list ]
                    if name in name_list: #dit_list
                        continue
                    else:
                        model.delete_adapters(name)
                        print(f"去除dit中未加载的lora: {name}")  
                else:
                    model.delete_adapters(name)
        except:pass

        return io.NodeOutput(model)
    
class ChronoEdit_SM_Vae(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Vae",
            display_name="ChronoEdit_SM_Vae",
            category="ChronoEdit_SM",
            inputs=[
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
                io.Combo.Input("vae_decoder",options= ["none"] + folder_paths.get_filename_list("vae") ),
                io.Boolean.Input("tiled", default=False),
                io.Boolean.Input("enable_temporal_reasoning", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                ],
            )
    @classmethod
    def execute(cls, latent,vae,vae_decoder,tiled,enable_temporal_reasoning) -> io.NodeOutput:
        samples=latent["samples"]
        use_lightvae=True 
        if "light" in vae_decoder.lower() or "tae" in vae_decoder.lower():
            vae_decoder=folder_paths.get_full_path("vae", vae_decoder)
            if os.path.basename(vae_decoder).split(".")[0]=="lightvaew2_1":
                from .Chrono.vae import WanVAE
                print("use lightvae decoder")
                vae = WanVAE(vae_path=vae_decoder,dtype=torch.bfloat16,device=device,use_lightvae=True)
            elif os.path.basename(vae_decoder).split(".")[0]=="taew2_1":
                from .Chrono.vae_tiny import WanVAE_tiny
                print("use vae_tiny decoder")
                vae = WanVAE_tiny(vae_path=vae_decoder,dtype=torch.bfloat16,device=device,need_scaled=False)
            elif os.path.basename(vae_decoder).split(".")[0]=="lighttaew2_1":
                from .Chrono.vae_tiny import WanVAE_tiny
                print("use vae_tiny light decoder")
                vae = WanVAE_tiny(vae_path=vae_decoder,dtype=torch.bfloat16,device=device,need_scaled=True)
            else:
                print(f"Unknown vae_name: {vae_decoder},only support lightvae,tae,tae_tiny,lighttae_tiny")
                use_lightvae=False 
        else:
            use_lightvae=False 
            print("use normal decoder")

        if not  use_lightvae: 
            if isinstance(vae, AutoencoderKLWan): 
                vae.to(device)
                samples=samples.to(device)
                if samples.shape[2]>2:
                    video_edit = vae.decode(samples[:, :, [0, -1]], return_dict=False)[0]
                    video_reason = vae.decode(samples[:, :, :-1], return_dict=False)[0]
                    images = torch.cat([video_reason, video_edit[:, :, 1:]], dim=2)
                else:
                    images = vae.decode(samples, return_dict=False)[0]
                #print(f"images shape: {images.shape}") #torch.Size([1, 3, 5, 720, 1280])
                # from diffusers.video_processor import VideoProcessor
                # from PIL import Image
                # video_processor_ = VideoProcessor(vae_scale_factor=8)
                # video = video_processor_.postprocess_video(images, output_type="pil")

                # if isinstance(video, list):
                #     for img in video:
                #         for j,pil_img in enumerate(img):
                #             if isinstance(pil_img, Image.Image):
                #                 pil_img.save(f"{j}.png")
                #             else:
                #                 continue
                images=images.cpu().float()[0]
                images = map_neg1_1_to_0_1(images)
                images = rearrange(images, "C T H W -> T H W C")
            else:
               
                if samples.shape[2]>2: #torch.Size([1, 16, 8, 60, 104])
                    if enable_temporal_reasoning:
                        video_edit = vae.decode(samples[:, :, [0, -1]])
                        video_reason =vae.decode(samples[:, :, :-1])
                        # print(f"Video edit shape: {video_edit.shape}") #Video edit shape: torch.Size([1, 5, 480, 832, 3])
                        # print(f"video_reason edit shape: {video_reason.shape}") #Video edit shape: torch.Size([1, 25, 480, 832, 3])
                        #images = torch.cat([video_reason, video_edit[:, :, 1:]], dim=2)
                        images = torch.cat([video_reason, video_edit[:, 1:, :, :]], dim=1)
                    else:
                        images = vae.decode(samples)
                else:
                    images = vae.decode(samples)
                
                if len(images.shape) == 5: #Combine batches
                    images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
              
        else:
            from .Chrono.vae import WanVAE
            if isinstance(vae, WanVAE):
                if tiled:
                    vae.use_tiling=True
                else:
                    vae.use_tiling=False
                samples=samples.to(device,dtype=torch.bfloat16) 
            else:
                vae.to(device)
            with torch.no_grad():

                images=vae.decode(samples.squeeze(0)).cpu().float()[0] #torch.Size([1, 3, 5, 832, 480])
                images = map_neg1_1_to_0_1(images)
                images = rearrange(images, "C T H W -> T H W C")
        
        print(f"images shape: {images.shape}")
        return io.NodeOutput(images)


class ChronoEdit_SM_Enhance_Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Enhance_Loader",
            display_name="ChronoEdit_SM_Enhance_Loader",
            category="ChronoEdit_SM",
            inputs=[
                io.String.Input("repo", multiline=False, default="Qwen/Qwen2.5-VL-7B-Instruct"),
                io.Combo.Input("clip",options= ["none"] + folder_paths.get_filename_list("clip")),
            ],
            outputs=[
                io.Custom("ChronoEdit_SM_Model_En").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, repo,clip) -> io.NodeOutput:
        clip_path=folder_paths.get_full_path("clip", clip) if clip != "none" else None

        if clip_path is not None:
            if "2.5" in clip_path.lower() or "qwen2" in clip_path.lower(): 
                model_name = os.path.join(node_cr_path,"Qwen/Qwen2.5-VL-7B-Instruct") 
            elif "qwen3" in clip_path.lower(): 
                model_name = os.path.join(node_cr_path,"Qwen/Qwen3-VL-8B-Instruct")
            else:
                raise ValueError(f"Unsupported model: {clip_path}")
            model=load_prompt_enhancer_cf(clip_path,model_name)
            processor = AutoProcessor.from_pretrained(model_name)
        elif repo:
            repo=PureWindowsPath(repo)
            model,processor=load_prompt_enhancer(repo)
        else:
            raise ValueError(f"Unsupported model: {repo} or {clip}")
        pipe={"model":model,"processor":processor,}
        return io.NodeOutput(pipe)

class ChronoEdit_SM_Enhance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Enhance",
            display_name="ChronoEdit_SM_Enhance",
            category="ChronoEdit_SM",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("prompt", multiline=True, default="Transform the image so that inside the floral teacup of steaming tea, a small, cute mouse is sitting and taking a bath; the mouse should look relaxed and cheerful, with a tiny white bath towel draped over its head as if enjoying a spa moment, while the steam rises gently around it, blending seamlessly with the warm and cozy atmosphere."),
                io.Custom("ChronoEdit_SM_Model_En").Input("model"),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                ],
            )
    @classmethod
    def execute(cls, image,prompt,model) -> io.NodeOutput:
        image=tensor2image(image)
        prompt=prompt_enhance(image,prompt,model["model"],model["processor"])

        return io.NodeOutput(prompt)


class ChronoEdit_SM_Latent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Latent",
            display_name="ChronoEdit_SM_Latent",
            category="ChronoEdit_SM",
            inputs=[
                io.Vae.Input("vae"),
                io.ClipVision.Input("clip_vision"),
                io.Image.Input("image"),
                io.Int.Input("width", default=0, min=0, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=0, min=0, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Combo.Input("num_frames",options= [5,29]),
            ],
            outputs=[
                io.Conditioning.Output(display_name="cond"),
                ],
            )
    @classmethod
    def execute(cls, vae,clip_vision,image,width,height,num_frames) -> io.NodeOutput:    


        image_embeds=clip_vision.encode_image(image,crop=True)["penultimate_hidden_states"]
        if width==0 or height==0:
            width, height=calculate_dimensions(image, 16)
        image = tensor_upscale(image, width, height) #BHWC

        if isinstance(vae, AutoencoderKLWan):
            from diffusers.video_processor import VideoProcessor
            video_processor_ = VideoProcessor(vae_scale_factor=8)
            image=tensor2image(image)
           
            image = video_processor_.preprocess(image, height=height, width=width).to(vae.device)
            image = image.unsqueeze(2)
            #image = map_0_1_to_neg1_1(image)
           
            #image=image.permute(0, 3, 1,2).unsqueeze(2) #(B, C, T, H, W)
            image = torch.cat([image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2)
            with torch.no_grad():
                Latent = retrieve_latents(vae.encode(image), sample_mode="argmax")
            torch.cuda.empty_cache()
            Latent = Latent.repeat(1, 1, 1, 1, 1)
            print(f"Latent shape: {Latent.shape}")
        else:
            image = torch.cat([image, image.new_zeros(int(num_frames) - 1, image.shape[1],  image.shape[2], image.shape[3])], dim=0)
            Latent=vae.encode(image)

        cond={"samples":Latent,"num_frames":num_frames,"image_embeds":image_embeds}
        return io.NodeOutput(cond)

class ChronoEdit_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ChronoEdit_SM_KSampler",
            display_name="ChronoEdit_SM_KSampler",
            category="ChronoEdit_SM",
            inputs=[
                io.Custom("ChronoEdit_SM_Model").Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"), 
                io.Conditioning.Input("cond"), #1, 16, 2, 60, 104
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Float.Input("flow_shift", default=2.0, min=0.1, max=10.0, step=0.1, round=0.01,),
                io.Int.Input("steps", default=8, min=1, max=10000),
                io.Float.Input("guidance_scale", default=1.0, min=0.1, max=10.0, step=0.1, round=0.01,),
                io.Int.Input("num_temporal_reasoning_steps", default=50, min=0, max=MAX_SEED,step=1,),
                io.Boolean.Input("offload_model", default=True),
                io.Int.Input("block_num", default=1, min=1, max=48,step=1),
          
            ],
            outputs=[
                io.Latent.Output(display_name="Latent"),
            ],
        )
    @classmethod
    def execute(cls, model,positive,negative,cond,seed,flow_shift, steps, guidance_scale,num_temporal_reasoning_steps,offload_model,block_num) -> io.NodeOutput:
        image_latent=cond["samples"]
        num_frames=cond["num_frames"]
        image_embeds=cond["image_embeds"]
        cf_models=mm.loaded_models()
        try:
            for pipe in cf_models:   
                pipe.unpatch_model(device_to=torch.device("cpu"))
                print(f"Unpatching models.{pipe}")
        except: pass
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        max_gpu_memory = torch.cuda.max_memory_allocated()
        print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")
        width,height=image_latent.shape[3]*8,image_latent.shape[4]*8
        Latent=inference_chrono (model,image_latent,positive[0][0],image_embeds,negative[0][0],flow_shift,seed,steps,
                                 width,height,num_frames,num_temporal_reasoning_steps,guidance_scale,offload_model,block_num,device)
        out={}
        out["samples"]=Latent #torch.Size([1, 16, 2, 60, 104])
        return io.NodeOutput(out)

class ChronoEdit_SM_LoadVAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ChronoEdit_SM_LoadVAE",
            display_name="ChronoEdit_SM_LoadVAE",
            category="ChronoEdit_SM",
            inputs=[
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
            ],
        )
    @classmethod
    def execute(cls, vae,) -> io.NodeOutput:
        vae_path=folder_paths.get_full_path("vae", vae) if vae!="none" else None
        assert vae_path is not None,"ChronoEdit_SM_LoadVAE: VAE file not found"
        if not vae_path.endswith(".safetensors"):
            raise ValueError("ChronoEdit_SM_LoadVAE: VAE file is not safetensors. Please diffuser version")
        vae=AutoencoderKLWan.from_single_file(vae_path,config=os.path.join(node_cr_path,"Chrono/ChronoEdit-14B-Diffusers/vae")) 
        return io.NodeOutput(vae)
    
from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/ChronoEdit_SM_Extension")
async def get_hello(request):
    return web.json_response("ChronoEdit_SM_Extension")

class ChronoEdit_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ChronoEdit_SM_Model,
            ChronoEdit_SM_Lora,
            ChronoEdit_SM_Enhance_Loader,
            ChronoEdit_SM_Enhance,
            ChronoEdit_SM_Vae,
            ChronoEdit_SM_Latent,
            ChronoEdit_SM_KSampler,
            ChronoEdit_SM_LoadVAE,
        ]
async def comfy_entrypoint() -> ChronoEdit_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return ChronoEdit_SM_Extension()




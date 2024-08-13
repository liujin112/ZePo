import os
import torch
import random
import numpy as np
import gradio as gr
from glob import glob
from datetime import datetime

from diffusers import StableDiffusionPipeline,AutoencoderKL
from diffusers import DDIMScheduler, LCMScheduler, EulerDiscreteScheduler

import torch.nn.functional as F
from PIL import Image,ImageDraw
from utils.pipeline import ZePoPipeline
from utils.attn_control import AttentionStyle
from torchvision.utils import save_image
import utils.ptp_utils as ptp_utils

import torchvision.transforms as transforms

try:
    import xformers
    is_xformers = True
except ImportError:
    is_xformers = False

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""
# import sys
# sys.setrecursionlimit(100000)


class GlobalText:
    def __init__(self):
        
        # config dirs
        self.basedir                = os.getcwd()
        self.stable_diffusion_dir   = os.path.join(self.basedir, "models", "StableDiffusion")
        self.personalized_model_dir = './models/Stable-diffusion'
        self.lora_model_dir         = './models/Lora'
        self.savedir            = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample         = os.path.join(self.savedir, "sample")

        # self.savedir_mask         = os.path.join(self.savedir, "mask")

        self.stable_diffusion_list   = ["SimianLuo/LCM_Dreamshaper_v7"
                                        ]
        self.personalized_model_list = []
        self.lora_model_list = []

        self.tokenizer             = None
        self.text_encoder          = None
        self.vae                   = None
        self.unet                  = None
        self.pipeline              = None
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.lora_model_state_dict = {}
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def init_source_image_path(self, source_path):
        self.source_paths = sorted(glob(os.path.join(source_path, '*')))
        self.max_source_index = len(self.source_paths) // 12
        return self.source_paths[0:12]
    def init_style_image_path(self, style_path):
        self.style_paths = sorted(glob(os.path.join(style_path, '*')))
        self.max_style_index = len(self.style_paths) // 12
        return self.style_paths[0:12]
    def init_results_image_path(self):
        results_paths = [os.path.join(self.savedir_sample, file) for file in os.listdir(self.savedir_sample)]
        self.results_paths = sorted(results_paths, key=os.path.getctime, reverse=True)
        self.max_results_index = len(self.results_paths) // 12
        return self.results_paths[0:12]
    
    def load_base_pipeline(self, model_path):
        
        time_start = datetime.now()

        self.scheduler = 'LCM'
        scheduler = LCMScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.pipeline = ZePoPipeline.from_pretrained(model_path,scheduler=scheduler,torch_dtype=torch.float16,).to('cuda')
        if is_xformers:
            self.pipeline.enable_xformers_memory_efficient_attention()
        time_end = datetime.now()
        print(f'Load {model_path} successful in {time_end-time_start}')
        return gr.Dropdown()
    
    def refresh_stable_diffusion(self,model_path):
        
        self.load_base_pipeline(model_path)
        
        return self.stable_diffusion_list[0]

    def update_base_model(self, base_model_dropdown):
        if self.pipeline is None:
            gr.Info(f"Please select a pretrained model path.")
            return None
        else:
            base_model = self.personalized_model_list[base_model_dropdown]
            mid_model = StableDiffusionPipeline.from_single_file(base_model)
            self.pipeline.vae = mid_model.vae
            self.pipeline.unet = mid_model.unet
            self.pipeline.text_encoder = mid_model.text_encoder
            self.pipeline.to(self.device)
            self.personal_model_loaded = base_model_dropdown.split('.')[0]
            print(f'load {base_model_dropdown} model success!')
            return gr.Dropdown()

    
    def generate(self, source, style, 
                       num_steps, co_feat_step,strength,
                       start_ac_layer, end_ac_layer,
                       sty_guidance,cfg_scale, mix_q_scale,
                       Scheduler, save_intermediate, seed, de_bug,
                       target_prompt, negative_prompt_textbox,
                       width_slider,height_slider,
                       tome_sx, tome_sy, tome_ratio,tome,
                       ):


        os.makedirs(self.savedir, exist_ok=True)
        os.makedirs(self.savedir_sample, exist_ok=True)
        
        if self.pipeline == None:
            self.refresh_stable_diffusion(self.stable_diffusion_list[-1])
        model = self.pipeline

        if Scheduler == 'DDIM':
            model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
            print(f"Successful adoption of DDIM scheduler")
        if Scheduler == 'LCM':
            model.scheduler = LCMScheduler.from_config(model.scheduler.config)
            print(f"Successful adoption of LCM scheduler")
        if Scheduler == 'EulerDiscrete':
            model.scheduler = EulerDiscreteScheduler.from_config(model.scheduler.config)

        if seed != '-1' and seed != "": torch.manual_seed(int(seed))
        else: torch.seed()
        
        seed = torch.initial_seed()
        print(f"Seed: {seed}")

        self.sample_count = len(os.listdir(self.savedir_sample))
        

        prompts = [target_prompt] * 3
        source = source.resize((width_slider, height_slider))
        style = style.resize((width_slider, height_slider))


        with torch.no_grad():

            controller = AttentionStyle(num_steps, 
                                         start_ac_layer,
                                         end_ac_layer,
                                         style_guidance=sty_guidance,
                                         mix_q_scale=mix_q_scale,
                                         de_bug=de_bug,
                                         )
                                                        
            ptp_utils.register_attention_control(model, controller, 
                                                 tome, 
                                                 sx=tome_sx,
                                                 sy=tome_sy,
                                                 ratio=tome_ratio,
                                                 de_bug=de_bug,)

            time_begin = datetime.now()
            generate_image = model(prompt=prompts,
                                negative_prompt=negative_prompt_textbox,
                                image=source,
                                style=style,
                                num_inference_steps=num_steps,
                                eta=0.0,
                                guidance_scale=cfg_scale,
                                strength=strength,
                                save_intermediate=save_intermediate,
                                fix_step_index=co_feat_step,
                                de_bug = de_bug,
                                callback = None
                   ).images
            time_end = datetime.now()
            print('generate one image with time {}'.format(time_end-time_begin))

            save_file_name = f"{self.sample_count}_step{num_steps}_sl{start_ac_layer}_el{end_ac_layer}_ST{strength}_CF{co_feat_step}_STG{sty_guidance}_MQ{mix_q_scale}_CFG{cfg_scale}_seed{seed}.jpg"

            
            save_file_path = os.path.join(self.savedir, save_file_name)
                    
            save_image(torch.tensor(generate_image).permute(0, 3, 1, 2), save_file_path, nrow=3, padding=0)
            save_image(torch.tensor(generate_image[2:]).permute(0, 3, 1, 2), os.path.join(self.savedir_sample, save_file_name), nrow=3, padding=0)
            self.init_results_image_path()
        return [
            generate_image[0],
            generate_image[1],
            generate_image[2],
            self.init_results_image_path()
            ]
    

global_text = GlobalText()


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # [ZePo: Zero-Shot Portrait Stylization with Faster Sampling](https://arxiv.org/abs/2408.05492)
            Jin Liu, Huaibo Huang, Jie Cao, Ran He<br>
            [Arxiv](https://arxiv.org/abs/2408.05492) | [Github](https://github.com/liujin112/ZePo)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Select a pretrained model.
                """
            )
            with gr.Row():
                stable_diffusion_dropdown = gr.Dropdown(
                    label="Pretrained Model Path",
                    choices=global_text.stable_diffusion_list,
                    interactive=True,
                    allow_custom_value=True
                )
                stable_diffusion_dropdown.change(fn=global_text.load_base_pipeline, inputs=[stable_diffusion_dropdown], outputs=[stable_diffusion_dropdown])
                
                stable_diffusion_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_stable_diffusion(stable_diffusion_dropdown):
                    global_text.refresh_stable_diffusion(stable_diffusion_dropdown)
                    
                stable_diffusion_refresh_button.click(fn=update_stable_diffusion, inputs=[stable_diffusion_dropdown], outputs=[stable_diffusion_dropdown])


        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for ZePo.
                """
            )
            with gr.Tab("Configs"):
                
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            source_image = gr.Image(label="Source Image",  elem_id="img2maskimg", sources="upload",  type="pil",image_mode="RGB", height=256)
                            style_image = gr.Image(label="Style Image", elem_id="img2maskimg", sources="upload", type="pil", image_mode="RGB", height=256)
                        
                        generate_image = gr.Image(label="Image with PortraitDiff", type="pil", interactive=True, image_mode="RGB", height=512)


                        with gr.Row():
                            recons_content = gr.Image(label="reconstructed content", type="pil", image_mode="RGB", height=256)
                            recons_style = gr.Image(label="reconstructed style", type="pil", image_mode="RGB", height=256)
                        prompt_textbox = gr.Textbox(label="Prompt", value='head', lines=1)
                        negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=1)
                    with gr.Row(equal_height=False):
                        with gr.Column():
                            with gr.Tab("Resolution"):
                                width_slider     = gr.Slider(label="Width", value=512, minimum=256, maximum=1024, step=64)
                                height_slider    = gr.Slider(label="Height", value=512, minimum=256, maximum=1024, step=64)
                                Scheduler = gr.Dropdown(
                                            ["DDIM", "LCM", "EulerDiscrete"],
                                            value="LCM",
                                            label="Scheduler", info="Select a Scheduler")


                            with gr.Tab("Content Gallery"):
                                
                                with gr.Row():
                                    source_path = gr.Textbox(value='./data/content', label="Source Path")
                                    refresh_source_list_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                                source_gallery_index = gr.Slider(label="Index", value=0, minimum=0, maximum=50, step=1)   
                                num_gallery_images = 12   
                                source_image_gallery = gr.Gallery(value=[], columns=4, label="Source Image List")
                                refresh_source_list_button.click(fn=global_text.init_source_image_path, inputs=[source_path], outputs=[source_image_gallery]) 

                                def update_source_list(index):
                                    if int(index) < 0:
                                        index = 0
                                    if int(index) > global_text.max_source_index:
                                        index = global_text.max_source_index
                                    return global_text.source_paths[int(index)*num_gallery_images:(int(index)+1)*num_gallery_images]
                                
                                source_gallery_index.change(fn=update_source_list, inputs=[source_gallery_index], outputs=[source_image_gallery])

                            with gr.Tab("Style Gallery"):
                                
                                with gr.Row():
                                    style_path = gr.Textbox(value='./data/style', label="style Path")
                                    refresh_style_list_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                                style_gallery_index = gr.Slider(label="Index", value=0, minimum=0, maximum=50, step=1)   
                                num_gallery_images = 12   
                                style_image_gallery = gr.Gallery(value=[], columns=4, label="style Image List")
                                refresh_style_list_button.click(fn=global_text.init_style_image_path, inputs=[style_path], outputs=[style_image_gallery]) 
                                

                                def update_style_list(index):
                                    if int(index) < 0:
                                        index = 0
                                    if int(index) > global_text.max_style_index:
                                        index = global_text.max_style_index
                                    return global_text.style_paths[int(index)*num_gallery_images:(int(index)+1)*num_gallery_images]
                                
                                style_gallery_index.change(fn=update_style_list, inputs=[style_gallery_index], outputs=[style_image_gallery])
                            
                            with gr.Tab("Results Gallery"):
                                with gr.Row():
                                    refresh_results_list_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                                    results_gallery_index = gr.Slider(label="Index", value=0, minimum=0, maximum=50, step=1)   
                                num_gallery_images = 12   
                                results_image_gallery = gr.Gallery(value=[], columns=4, label="style Image List")
                                refresh_results_list_button.click(fn=global_text.init_results_image_path, inputs=[], outputs=[results_image_gallery]) 
                                

                                def update_results_list(index):
                                    if int(index) < 0:
                                        index = 0
                                    if int(index) > global_text.max_results_index:
                                        index = global_text.max_results_index
                                    return global_text.results_paths[int(index)*num_gallery_images:(int(index)+1)*num_gallery_images]
                                
                                results_gallery_index.change(fn=update_results_list, inputs=[results_gallery_index], outputs=[style_image_gallery])
                            
                            

                            with gr.Row():    
                                generate_button = gr.Button(value="Generate", variant='primary')

                            with gr.Tab('Base Configs'):
                                num_steps = gr.Slider(label="Total Steps", value=4, minimum=0, maximum=25, step=1)
                                strength = gr.Slider(label="Noisy Ratio", value=0.5, minimum=0, maximum=1, step=0.01,info="How much noise applied to souce image, 50% for better balance.")
                                co_feat_step = gr.Slider(label="Consistency Feature Extract Step", value=99, minimum=0, maximum=999, step=1)
                                

                                with gr.Row():    
                                    start_ac_layer = gr.Slider(label="Start Layer of AC",
                                                            minimum=0,
                                                            maximum=16,
                                                            value=8,
                                                            step=1)
                                    end_ac_layer = gr.Slider(label="End Layer of AC",
                                                            minimum=0,
                                                            maximum=16,
                                                            value=16,
                                                            step=1)
                                    
                                with gr.Row():     
                                    Style_Guidance = gr.Slider(label="Style Guidance Scale",
                                                    minimum=-1,
                                                    maximum=3,
                                                    value=1.2,
                                                    step=0.01,
                                                    )
                                    mix_q_scale = gr.Slider(label='Query Mix Ratio',
                                                            minimum=0,
                                                            maximum=2,
                                                            step=0.05,
                                                            value=1.0,
                                                            )
                                cfg_scale_slider = gr.Slider(label="CFG Scale", value=2.5, minimum=0, maximum=20, info="Classifier-free guidance scale.")
                                
                                with gr.Row():     
                                    save_intermediate = gr.Checkbox(label="save_intermediate", value=False)
                                    de_bug = gr.Checkbox(value=False,label='DeBug')
                            with gr.Tab('ToMe'):
                                with gr.Row():
                                    tome = gr.Checkbox(label="Token Merge", value=True)
                                    
                                    tome_ratio = gr.Slider(label='ratio: ',
                                                            minimum=0,
                                                            maximum=1,
                                                            step=0.1,
                                                            value=0.5)
                                with gr.Row():
                                    tome_sx = gr.Slider(label='sx:',
                                                            minimum=0,
                                                            maximum=64,
                                                            step=2,
                                                            value=2)
                                    tome_sy = gr.Slider(label='sy:',
                                                            minimum=0,
                                                            maximum=64,
                                                            step=2,
                                                            value=2)
                            
                                
                            with gr.Row():
                                seed_textbox = gr.Textbox(label="Seed", value=-1)
                                seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                                seed_button.click(fn=lambda: random.randint(1, 1e16), inputs=[], outputs=[seed_textbox])
        inputs = [
            source_image, style_image,
            num_steps,co_feat_step,strength,
            start_ac_layer, end_ac_layer,
            Style_Guidance,cfg_scale_slider,mix_q_scale,
            Scheduler, save_intermediate, seed_textbox, de_bug,  
            prompt_textbox, negative_prompt_textbox,
            width_slider,height_slider,
            tome_sx, tome_sy, tome_ratio, tome,
        ]

        generate_button.click(
            fn=global_text.generate,
            inputs=inputs,
            outputs=[recons_style,recons_content,generate_image,results_image_gallery]
        )

        ex = gr.Examples(
        [
          ["./data/content/27032.jpg","./data/style/27.jpg",4,0.8,0.5,8427921159605868845],         
          ["./data/content/29812.jpg","./data/style/47.png",4,0.5,0.65,8119359809263726691],
        ],
        [source_image, style_image, num_steps,strength, mix_q_scale, seed_textbox],
        [
            "Example 1",
        ],)


    return demo

if __name__ == "__main__":
    demo = ui()
    demo.launch(show_error=True)
import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from diffusers import DDIMScheduler,LCMScheduler
from torchvision.utils import save_image
from PIL import Image 
from utils.pipeline import ZePoPipeline
from utils.attn_control import AttentionStyle
import utils.ptp_utils as ptp_utils
from datetime import datetime
import torchvision.transforms as transforms


def load_image(image_path, res, device, gray=False):
    image = Image.open(image_path).convert('RGB') if not gray else Image.open(image_path).convert('L')
    image = torch.tensor(np.array(image)).float()
    if gray:
        image = image.unsqueeze(-1).repeat(1,1,3)
    image = image.permute(2, 0, 1)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (res, res))
    image = image.to(device)
    return image


def main():
    args = argparse.ArgumentParser()

    args.add_argument("--start_ac_layer", type=int, default=8)
    args.add_argument("--end_ac_layer", type=int, default=16)
    args.add_argument("--res", type=int, default=512)
    args.add_argument("--cfg_guidance", type=float, default=2)
    args.add_argument("--sty_guidance", type=float, default=0.5)
    args.add_argument("--mix_q_scale", type=float, default=1.0)
    args.add_argument("--prompt", type=str, default='face')
    args.add_argument("--neg_prompt", type=str, default='')
    args.add_argument("--output", type=str, default='./results/')
    args.add_argument("--content", type=str, default=None)
    args.add_argument("--style", type=str, default=None)
    args.add_argument("--model_path", type=str, default='SimianLuo/LCM_Dreamshaper_v7')
    args.add_argument("--num_inference_steps", type=int, default=4)
    args.add_argument("--fix_step_index", type=int, default=99)
    args.add_argument("--tome", action="store_true")
    args.add_argument("--tome_ratio", type=float, default=0.5)

    
    args = args.parse_args()

    out_dir = args.output
    start_ac_layer = args.start_ac_layer
    end_ac_layer = args.end_ac_layer
    num_inference_steps = args.num_inference_steps
    sty_guidance = args.sty_guidance
    fix_step_index = args.fix_step_index
    mix_q_scale = args.mix_q_scale
    de_bug = False
    tome = args.tome
    tome_sx=2
    tome_sy=2
    tome_ratio=args.tome_ratio
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    cfg_scale = args.cfg_guidance
    source_prompt = [args.prompt]
    model_path = args.model_path
    model = ZePoPipeline.from_pretrained(model_path).to(device)
    model.scheduler = LCMScheduler.from_config(model.scheduler.config)
    prompts = source_prompt * 3
    neg_prompt = ''
    

    

    for style_dir in os.listdir(args.style):
        style_name = os.path.splitext(os.path.basename(style_dir))[0]
        os.makedirs(os.path.join(out_dir, style_name), exist_ok=True)
        os.makedirs(os.path.join(out_dir,style_name, 'results_only'), exist_ok=True)
        time_begin = datetime.now()
        style_image = Image.open(os.path.join(args.style, style_dir)).convert('RGB')
        style = style_image.resize((args.res, args.res))
        print(f"Start processing style {style_name} at {time_begin}")
        for content_dir in os.listdir(args.content):
            content_name = os.path.splitext(os.path.basename(content_dir))[0]
            save_name = os.path.join(out_dir,style_name, f"{content_name}.png")
            if os.path.exists(save_name):
                continue

            source_image = Image.open(os.path.join(args.content, content_dir)).convert('RGB')

            source = source_image.resize((args.res, args.res))


            
            controller = AttentionStyle(num_inference_steps, 
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
            with torch.no_grad():


                    
                    generate_image = model(prompt=prompts,
                                negative_prompt=neg_prompt,
                                image=source,
                                style=style,
                                num_inference_steps=num_inference_steps,
                                eta=0.0,
                                guidance_scale=cfg_scale,
                                strength=0.5,
                                save_intermediate=False,
                                fix_step_index=fix_step_index,
                                de_bug = de_bug,
                                callback = None
                   ).images
                    
                    os.makedirs(out_dir, exist_ok=True)

                    generate_image = torch.from_numpy(generate_image).permute(0, 3, 1, 2)

                    save_image(generate_image, save_name, nrow=3, padding=0)
                    save_image(generate_image[-1:], os.path.join(out_dir,style_name, 'results_only',f"{content_name}.png"), nrow=1, padding=0)

                    
        time_end = datetime.now()
        print(f"Finish processing style {style_name} at {time_end} \nTime cost: {time_end-time_begin}, \nPer image cost: {(time_end-time_begin)/len(os.listdir(args.content))}")
        

if __name__ == "__main__":
    main()
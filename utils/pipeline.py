import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import os
import numpy as np
import PIL
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import LCMScheduler
from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

logger = logging.get_logger(__name__)  



class ZePoPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: LCMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.check_inputs
    def check_inputs(
        self, prompt, strength, callback_steps, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_extra_step_kwargs(self, generator, eta):


        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

   
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min( int(num_inference_steps * strength), num_inference_steps)
        init_timestep = max(init_timestep, 1)
        t_start = max(num_inference_steps - init_timestep, 0)
        
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, timestep, device,dtype, denoise_model, generator=None):
        image = image.to(device=device,dtype=dtype)

        batch_size = image.shape[0]

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        

        # add noise to latents using the timestep
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        clean_latents = init_latents
        if denoise_model:
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            latents = init_latents
        else:
            latents = noise

        return latents, clean_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]]=None,
        image: PipelineImageInput = None,
        style: PipelineImageInput = None,
        strength: float = 0.5,
        num_inference_steps: Optional[int] = 50,
        original_inference_steps: Optional[int]  = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        denoise_model: Optional[bool] = True,
        fix_step_index = 0,
        target_start_step = -1,
        save_intermediate = False,
        de_bug=False,

    ):
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)
        num_inference_steps = int(num_inference_steps * (1/strength))
        print(f'num_inference_steps {num_inference_steps} is multiple by {int(1/strength)}.')
        # 2. Define call parameters
        batch_size = len(prompt) 
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        
        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        dtype=self.unet.dtype
        prompt_embeds = self.text_encoder(text_input.input_ids.to(device))[0]
        prompt_embeds=prompt_embeds.to(dtype=dtype, device=device)
        #print("input text embeddings :", prompt_embeds.shape)
        
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if negative_prompt:
                uc_text = negative_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(device))[0]
            unconditional_embeddings=unconditional_embeddings.to(dtype=dtype, device=device)
            prompt_embeds = torch.cat([unconditional_embeddings, prompt_embeds], dim=0)

        #print("prompt embeds shape: ", prompt_embeds.shape)
        
        
        # 4. Preprocess image
        image = self.image_processor.preprocess(image)
        style = self.image_processor.preprocess(style)

        # 5. Prepare timesteps
        if isinstance(self.scheduler, LCMScheduler):
            self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps, 
            device=device, 
            original_inference_steps=original_inference_steps)
        else:
            self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps, 
            device=device,)
        print(f"num_inference_steps is {self.scheduler.timesteps}")
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        
        #print(f"All timesteps is : {timesteps}")
        latent_timestep = torch.tensor([fix_step_index], device=device)
        
        assert timesteps != []

        print("The time-steps are: ", timesteps)
        
        # 6. Prepare latent variables
        src_latents, src_clean_latents = self.prepare_latents(
            image, latent_timestep, device,dtype, denoise_model, generator
        )

        
        sty_latents, sty_clean_latents = self.prepare_latents(
            style, latent_timestep, device,dtype, denoise_model, generator
        )

        
        mutual_latents, _ = self.prepare_latents(
            image, timesteps[:1], device, dtype, denoise_model, generator
        )

        # mutual_latents = src_latents
        #latents = torch.cat([sty_t_latents, src_t_latents], dim=0)
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        generator = extra_step_kwargs.pop("generator", None)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if de_bug:

                    import pdb; pdb.set_trace()
                
                model_input = torch.cat(
                        [
                            sty_latents,
                            src_latents,
                            mutual_latents

                        ],
                        dim=0,
                    )
                # predict the noise residual
                if do_classifier_free_guidance:
                    concat_latent_model_input = torch.cat([model_input] * 2)
                    concat_prompt_embeds = prompt_embeds
                    #raise NotImplementedError("Classifier free guidance is not yet supported")
                else:
                    concat_latent_model_input = model_input
                    concat_prompt_embeds = prompt_embeds
                    assert len(concat_prompt_embeds) == len(concat_latent_model_input)

                timestep = torch.cat([latent_timestep] * (batch_size-1)+[t[None]], dim=0)

                if do_classifier_free_guidance:
                    timestep = torch.cat([timestep] * 2)

                
                concat_noise_pred = self.unet(
                    concat_latent_model_input,
                    timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_hidden_states=concat_prompt_embeds,
                ).sample
                # perform guidance
                if do_classifier_free_guidance:

                    (
                        noise_pred, 
                        noise_pred_uncond, 
                    ) = concat_noise_pred.chunk(2, dim=0)
                    

                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                    
                else:
                    noise_pred = concat_noise_pred
                
                (style_noise_pred, source_noise_pred, mutual_noise_pred) = noise_pred.chunk(3, dim=0)

                noise = torch.randn_like(
                    source_noise_pred
                )


                if isinstance(self.scheduler, LCMScheduler):
                    mutual_latents, pred_x0_mutual = self.scheduler.step(mutual_noise_pred, t, mutual_latents, return_dict=False)
                else:
                    ddim_out = self.scheduler.step(mutual_noise_pred, t, mutual_latents)
                    mutual_latents, pred_x0_mutual = ddim_out.prev_sample, ddim_out.pred_original_sample
                
                
                pred_x0 = torch.cat([sty_clean_latents,src_clean_latents,pred_x0_mutual ], dim=0)
      
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                
        model_input = torch.cat([sty_latents,src_latents,mutual_latents],dim=0,)
        
        # 9. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(pred_x0 / self.vae.config.scaling_factor, return_dict=False)[0]
            has_nsfw_concept = None
        else:
            image = pred_x0
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type='np', do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

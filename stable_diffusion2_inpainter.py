import cv2
import torch
import numpy as np

from PIL import Image
from typing import List, Union
from pathlib import Path
from omegaconf import OmegaConf

from einops import repeat
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


def create_mask(img: Image):
    img_width, img_height  = img.size
    empty_mask = np.zeros((img_height, img_width, 1), dtype=np.uint8)
    
    start_x = 150
    end_x = img_width - 150
    start_y = 150
    end_y = img_height - 150
    
    mask = cv2.rectangle(empty_mask, (start_x, start_y), (end_x, end_y), (1), -1)
    mask = Image.fromarray(np.uint8(mask[:, :, 0] * 255) , 'L')
    
    return mask


class StableDiffusion2Inpainter:
    def __init__(
        self, 
        config_path: Union[str, Path], 
        ckpt_path: Union[str, Path], 
        device: torch.device,
        half_model: bool = False,
        scale: int = 9,
        seed: int = 42
    ):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.device = device
        self.half_model = half_model
        self.scale = scale
        self.seed = seed
        self.sampler = self._initialize_model(self.config_path, self.ckpt_path, self.half_model)

    def _initialize_model(self, config: str, ckpt: str, half_model: bool=False):
        print("Initializing model...")
        
        config = OmegaConf.load(config)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

        if half_model:
            model = model.half()
        model = model.to(self.device)
        sampler = DDIMSampler(model)

        return sampler
    
    def _make_batch_sd(
        self, 
        image: Image.Image,
        mask: Image.Image,
        prompt: str, 
        num_samples: int = 1
    ):
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = image * (mask < 0.5)

        batch = {
            "image": repeat(image.to(device=self.device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [prompt],
            "mask": repeat(mask.to(device=self.device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=self.device), "1 ... -> n ...", n=num_samples),
        }
        return batch

    def _inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        ddim_steps: int,
        num_samples: int = 1,
        w: int = 512,
        h: int = 512
    ) -> np.ndarray:
        """
            returns: np.ndarray of shape (num_samples, h, w, 3)
        """
        
        self.model = self.sampler.model

        prng = np.random.RandomState(self.seed)
        start_code = prng.randn(num_samples, 4, h // 8, w // 8)
        start_code = torch.from_numpy(start_code).to(device=self.device, dtype=torch.float32)

        with torch.no_grad(), torch.autocast(self.device.type):
            batch = self._make_batch_sd(
                image, 
                mask, 
                prompt=prompt,
                num_samples=num_samples
            )

            c = self.model.cond_stage_model.encode(batch["txt"])
            c_cat = list()
            for ck in self.model.concat_keys:
                cc = batch[ck].float()
                if ck != self.model.masked_image_key:
                    bchw = [num_samples, 4, h // 8, w // 8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = self.model.get_first_stage_encoding(
                        self.model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = self.model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [self.model.channels, h // 8, w // 8]
            samples_cfg, _ = self.sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=self.scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,
            )
            x_samples_ddim = self.model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
        return result

    def __call__(
        self,
        input_image: Image.Image,
        input_mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
        num_samples: int = 1,
    ) -> List[Image.Image]:

        assert input_image.size == input_mask.size, "Image and mask should be the same size"
        
        init_image = input_image.convert("RGB")
        init_mask = input_mask.convert("RGB")
        image = pad_image(init_image) # resize to integer multiple of 32
        mask = pad_image(init_mask) # resize to integer multiple of 32
        width, height = image.size
        print("Inpainting...", width, height)

        inpainted = self._inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            ddim_steps=num_inference_steps,
            num_samples=num_samples,
            h=height, 
            w=width
        )
        
        results = [
            Image.fromarray(img.astype(np.uint8)).crop(
                (0, 0, init_image.size[0], init_image.size[1])
            )
            for img in inpainted if img.size != (width, height)
        ]
        return results

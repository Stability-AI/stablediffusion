from custom_inpainting.utils import BBox, open_coco
import torch

from PIL import Image
from pathlib import Path
from typing import Optional

from stable_diffusion2_inpainter import StableDiffusion2Inpainter


class Inpainter:

    def __init__(
        self,
        config_path: str ,
        weights_path: str,
        half_model: bool = False,
        num_inference_steps: int = 60,

        input_width: int = 512,
    ):
        self.num_inference_steps = num_inference_steps

        device = torch.device(f"cuda:0" if torch.cuda.is_available() else 'cpu')

        self.pipe = StableDiffusion2Inpainter(
            config_path=config_path,
            ckpt_path=weights_path,
            device=device,
            half_model=half_model,
        )

        self.input_width = input_width
        
        self.input_stride = 32 # Only used for checking the resolution of the input image

        assert self.input_width % self.input_stride == 0, f"Specified input width {self.input_width} is not a multiple of the stride {self.input_stride}"


    
    def inpaint(
            self,
            image: Image.Image,
            mask: Image.Image,
            prompt: str,
        ) -> Image.Image:
        return self.pipe(
                input_image=image, 
                input_mask=mask,
                prompt=prompt, 
                num_inference_steps=self.num_inference_steps,
            )[0]





import subprocess

import torch
import numpy as np
import typing
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder

from scripts.txt2img import put_watermark
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentUpscaleFinetuneDiffusion
from ldm.util import exists, instantiate_from_config

torch.set_grad_enabled(False)


from cog import BasePredictor, Path, Input

class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "/root/.cache/huggingface"])
        subprocess.run(["mkdir", "/root/.cache/huggingface/hub"])
        subprocess.run(["cp", "-r", "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K", "/root/.cache/huggingface/hub"])
        subprocess.run(["pip3", "install", "-e", "."])
        config = OmegaConf.load('configs/stable-diffusion/x4-upscaling.yaml')
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load('x4-upscaler-ema.ckpt')["state_dict"], strict=False)

        device = torch.device("cuda:0")
        model = model.to(device)
        self.sampler = DDIMSampler(model)

    def predict(
            self,
            input_image: Path = Input(default="Image to upscale (Currently memory is not sufficient for 512x512 inputs)"),
            # scale: float = Input(description="Number of denoising steps", ge=0.1, le=4.0, default=4.0),
            ddim_steps: int = Input(description="Number of denoising steps", ge=2, le=250., default=50),
            ddim_eta: float = Input(description="Upscale factor", ge=0., le=1.0, default=0.),
            seed: int = Input(description="Integer seed", default=0),
    ) -> typing.List[Path]:
        torch.cuda.empty_cache()
        ddim_steps = int(ddim_steps)
        ddim_eta = float(ddim_eta)
        seed = int(seed)
        num_outputs = 1
        scale = 9.0

        image = Image.open(str(input_image))
        w, h = image.size
        width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((width, height))

        noise_level = None
        if isinstance(self.sampler.model, LatentUpscaleDiffusion):
            # TODO: make this work for all models
            noise_level = 20  # , min_value=0, max_value=350, value=20)
            noise_level = torch.Tensor(num_outputs * [noise_level]).to(self.sampler.model.device).long()

        self.sampler.make_schedule(ddim_steps, ddim_eta=ddim_eta, verbose=True)

        scaling_prompt = "a high quality professional photograph"
        result = paint(
            sampler=self.sampler,
            image=image,
            prompt=scaling_prompt,
            seed=seed,
            scale=scale,
            h=height, w=width, steps=ddim_steps,
            num_samples=num_outputs,
            noise_level=noise_level,
            eta=ddim_eta
        )

        outputs = []
        for i, image in enumerate(result):
            path = f"output-{i}.png"
            outputs.append(Path(path))
            image.save(path)
        return outputs


def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    batch = {
        "lr": rearrange(image, 'h w c -> 1 c h w'),
        "txt": num_samples * [txt],
    }
    batch["lr"] = repeat(batch["lr"].to(device=device), "1 ... -> n ...", n=num_samples)
    return batch


def make_noise_augmentation(model, batch, noise_level=None):
    x_low = batch[model.low_scale_key]
    x_low = x_low.to(memory_format=torch.contiguous_format).float()
    x_aug, noise_level = model.low_scale_model(x_low, noise_level)
    return x_aug, noise_level


def paint(sampler, image, prompt, seed, scale, h, w, steps, num_samples=1, callback=None, eta=0., noise_level=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(seed)
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, model.channels, h , w)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    with torch.no_grad(),\
            torch.autocast("cuda"):
        batch = make_batch_sd(image, txt=prompt, device=device, num_samples=num_samples)
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        if isinstance(model, LatentUpscaleFinetuneDiffusion):
            for ck in model.concat_keys:
                cc = batch[ck]
                if exists(model.reshuffle_patch_size):
                    assert isinstance(model.reshuffle_patch_size, int)
                    cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                                   p1=model.reshuffle_patch_size, p2=model.reshuffle_patch_size)
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        elif isinstance(model, LatentUpscaleDiffusion):
            x_augment, noise_level = make_noise_augmentation(model, batch, noise_level)
            cond = {"c_concat": [x_augment], "c_crossattn": [c], "c_adm": noise_level}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [x_augment], "c_crossattn": [uc_cross], "c_adm": noise_level}
        else:
            raise NotImplementedError()

        shape = [model.channels, h, w]
        samples, intermediates = sampler.sample(
            steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
            callback=callback
        )
    with torch.no_grad():
        x_samples_ddim = model.decode_first_stage(samples)
    result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]
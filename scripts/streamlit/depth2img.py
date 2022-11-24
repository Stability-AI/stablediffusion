import sys
import torch
import numpy as np
import streamlit as st
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder

from scripts.txt2img import put_watermark
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.util import AddMiDaS

torch.set_grad_enabled(False)


@st.cache(allow_output_mutation=True)
def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    return sampler


def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
        model_type="dpt_hybrid"
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    # sample['jpg'] is tensor hwc in [-1, 1] at this point
    midas_trafo = AddMiDaS(model_type=model_type)
    batch = {
        "jpg": image,
        "txt": num_samples * [txt],
    }
    batch = midas_trafo(batch)
    batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
    batch["jpg"] = repeat(batch["jpg"].to(device=device), "1 ... -> n ...", n=num_samples)
    batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(device=device), "1 ... -> n ...", n=num_samples)
    return batch


def paint(sampler, image, prompt, t_enc, seed, scale, num_samples=1, callback=None,
          do_full_sample=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(seed)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    with torch.no_grad(),\
            torch.autocast("cuda"):
        batch = make_batch_sd(image, txt=prompt, device=device, num_samples=num_samples)
        z = model.get_first_stage_encoding(model.encode_first_stage(batch[model.first_stage_key]))  # move to latent space
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck]
            cc = model.depth_model(cc)
            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                           keepdim=True)
            display_depth = (cc - depth_min) / (depth_max - depth_min)
            st.image(Image.fromarray((display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8)))
            cc = torch.nn.functional.interpolate(
                cc,
                size=z.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                           keepdim=True)
            cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        if not do_full_sample:
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))
        else:
            z_enc = torch.randn_like(z)
        # decode it
        samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                 unconditional_conditioning=uc_full, callback=callback)
        x_samples_ddim = model.decode_first_stage(samples)
        result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]


def run():
    st.title("Stable Diffusion Depth2Img")
    # run via streamlit run scripts/demo/depth2img.py <path-tp-config> <path-to-ckpt>
    sampler = initialize_model(sys.argv[1], sys.argv[2])

    image = st.file_uploader("Image", ["jpg", "png"])
    if image:
        image = Image.open(image)
        w, h = image.size
        st.text(f"loaded input image of size ({w}, {h})")
        width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((width, height))
        st.text(f"resized input image to size ({width}, {height} (w, h))")
        st.image(image)

        prompt = st.text_input("Prompt")

        seed = st.number_input("Seed", min_value=0, max_value=1000000, value=0)
        num_samples = st.number_input("Number of Samples", min_value=1, max_value=64, value=1)
        scale = st.slider("Scale", min_value=0.1, max_value=30.0, value=9.0, step=0.1)
        steps = st.slider("DDIM Steps", min_value=0, max_value=50, value=50, step=1)
        strength = st.slider("Strength", min_value=0., max_value=1., value=0.9)

        t_progress = st.progress(0)
        def t_callback(t):
            t_progress.progress(min((t + 1) / t_enc, 1.))

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        do_full_sample = strength == 1.
        t_enc = min(int(strength * steps), steps-1)
        sampler.make_schedule(steps, ddim_eta=0., verbose=True)
        if st.button("Sample"):
            result = paint(
                sampler=sampler,
                image=image,
                prompt=prompt,
                t_enc=t_enc,
                seed=seed,
                scale=scale,
                num_samples=num_samples,
                callback=t_callback,
                do_full_sample=do_full_sample,
            )
            st.write("Result")
            for image in result:
                st.image(image, output_format='PNG')


if __name__ == "__main__":
    run()

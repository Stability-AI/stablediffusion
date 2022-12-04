import cv2
import uuid
import torch
import random
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torchvision import transforms as T
from torchvision.transforms.functional import crop

from inpainting import setup_log_file, nearest_multiple, apply_on_src_image, generate_random_prompt, get_regexps_by_group, img_name_match_to_pattern
from stable_diffusion2_inpainter import StableDiffusion2Inpainter
from eg_data_tools.annotation_processing.coco_utils.coco_read_write import open_coco
from eg_data_tools.data.group_regexps import REGEXP_SELECTORS


def get_crop_coords(
    x1: int, y1: int, 
    x2: int, y2: int, 
    crop_size: int, 
    img_height: int, 
    img_width: int
):
    mid_x = int((x1 + x2) / 2)
    mid_y = int((y1 + y2) / 2)
    
    crop_x1 = max(0, mid_x - crop_size // 2)
    if crop_x1 == 0:
        crop_x2 = crop_size
    else:
        crop_x2 = min(mid_x + crop_size // 2, img_width)
    if crop_x2 == img_width:
        crop_x1 = img_width - crop_size
    
    
    crop_y1 = max(0, mid_y - crop_size // 2)
    if crop_y1 == 0:
        crop_y2 = crop_size
    else:
        crop_y2 = min(mid_y + crop_size // 2, img_height)
    if crop_y2 == img_height:
        crop_y1 = img_height - crop_size
    
    return crop_x1, crop_y1, crop_x2, crop_y2

def get_mask_with_bbox(x1, y1, x2, y2, img_height, img_width):
    mask = np.zeros((img_height, img_width, 1), dtype=np.uint8)
    mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (1), -1)
    return mask


def run_sd_inpainting(
    src_images_dir: str,
    coco_ann_path: str,
    inpainted_images_dir: str,
    prompts_file_path: str,
    logs_file_path: str,
    config_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    half_model: bool = False,
    crop_size: int = 640,
    inference_resize: Optional[int] = None,
    generate_prompt: bool = True,
    base_prompt: str = 'person, man',
    num_inference_steps: int = 60,
    device_id: int = 0,
    multiple_coef: int = 32,
    regexpx_group: Optional[str] = None,
    select_random_images: bool = False,
):
    if not any([config_path, weights_path]):
        import stable_diffusion2_inpainter
        config_path = Path(stable_diffusion2_inpainter.__file__).parent / "configs" / "stable-diffusion" / "v2-inpainting-inference.yaml"
        weights_path = Path(stable_diffusion2_inpainter.__file__).parent / "weights" / "512-inpainting-ema.ckpt"

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu')
    pipe = StableDiffusion2Inpainter(
        config_path=config_path,
        ckpt_path=weights_path,
        device=device,
        half_model=half_model,
    )
        
    if inference_resize is not None:
        inference_resize = nearest_multiple(inference_resize, multiple_coef)
    else:
        crop_size = nearest_multiple(crop_size, multiple_coef)
    src_images_dir = Path(src_images_dir)
    prompts_file_path = Path(prompts_file_path)
    logs_file_path = Path(logs_file_path)
    inpainted_images_dir = Path(inpainted_images_dir)
    inpainted_images_dir.mkdir(exist_ok=True)
    
    prompts = [prompt.strip() for prompt in prompts_file_path.read_text().splitlines()]
    # setup_log_file(logs_file_path)
    limages = open_coco(coco_ann_path)
    
    if regexpx_group is not None:
        patterns = get_regexps_by_group(REGEXP_SELECTORS.ALL_REGEXPS, regexpx_group)
        img_names = []
        for img_path in src_images_dir.iterdir():
            if img_name_match_to_pattern(img_path.name, patterns):
                img_names.append(img_path.name)
                continue
    else:
        img_names = [img_path.name for img_path in src_images_dir.iterdir()]
    
    if select_random_images:
        limages = random.choices(limages, k=len(limages) * 2)
    for limage in limages:
        if limage.name not in img_names:
            continue
        
        img_path = src_images_dir / limage.name
        img_uid = str(uuid.uuid4())[:8]
        generated_img_basename = f"{img_path.stem}_{img_uid}"
        if generate_prompt:
            prompt = generate_random_prompt(base_prompt, prompts)
        else:
            prompt = random.choice(prompts)
        
        image = Image.open(img_path)
        for bbox in limage.bbox_list:
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            
            crop_x1, crop_y1, crop_x2, crop_y2 = get_crop_coords(
                x1, 
                y1, 
                x2, 
                y2, 
                crop_size, 
                image.height, 
                image.width
            )
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            
            mask = get_mask_with_bbox(x1, y1, x2, y2, image.height, image.width)
            mask = Image.fromarray(np.uint8(mask[:, :, 0] * 255) , 'L')
            mask = crop(mask, crop_y1, crop_x1, crop_h, crop_w)
            img = crop(image, crop_y1, crop_x1, crop_h, crop_w)

            if inference_resize is not None:
                mask = mask.resize((inference_resize, inference_resize), resample=Image.NEAREST)
                img = img.resize((inference_resize, inference_resize), resample=Image.BILINEAR)
                        
            img = pipe(
                input_image=img, 
                input_mask=mask,
                prompt=prompt, 
                num_inference_steps=num_inference_steps,
            )[0]
            
            if inference_resize is not None:
                mask = mask.resize((crop_size, crop_size), resample=Image.NEAREST)
                img = img.resize((crop_size, crop_size), resample=Image.BILINEAR)
            
            image, mask = apply_on_src_image(image, img, mask, crop_x1, crop_y1, crop_w, crop_h)

        image.save(inpainted_images_dir / f'{generated_img_basename}.jpg')
        
        with logs_file_path.open('a', encoding='utf-8') as f:
            f.write(f'{generated_img_basename}.jpg, "{prompt}"\n')


def parce_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--device_id', type=int, default=0)
    args.add_argument('--src_images_dir', type=str, required=True)
    args.add_argument('--coco_ann_path', type=str, required=True)
    args.add_argument('--generated_images_dir', type=str, required=True)
    args.add_argument('--prompts_file_path', type=str, required=True)
    args.add_argument('--logs_file_path', type=str, required=True)
    args.add_argument('--config_path', type=str, default=None)
    args.add_argument('--weights_path', type=str, default=None)
    args.add_argument('--half_model', action='store_true')
    args.add_argument('--crop_size', type=int, default=640)
    args.add_argument('--inference_resize', type=int, default=None, required=False)
    args.add_argument('--num_infer_steps', type=int, default=60)
    args.add_argument('--generate_prompt', action='store_true')
    args.add_argument('--base_prompt', type=str, default="person, man")
    args.add_argument('--multiple_coef', type=int, default=32)
    args.add_argument('--regexpx_group', type=str, default=None, required=False)
    args.add_argument('--select_random_images', action='store_true')
    return args.parse_args()


if __name__ == "__main__":
    args = parce_args()
    
    run_sd_inpainting(
        src_images_dir=args.src_images_dir,
        coco_ann_path=args.coco_ann_path,
        inpainted_images_dir=args.generated_images_dir,
        prompts_file_path=args.prompts_file_path,
        logs_file_path=args.logs_file_path,
        config_path=args.config_path,
        weights_path=args.weights_path,
        half_model=args.half_model,
        crop_size=args.crop_size,
        inference_resize=args.inference_resize,
        generate_prompt=args.generate_prompt,
        base_prompt=args.base_prompt,
        num_inference_steps=args.num_infer_steps,
        device_id=args.device_id,
        regexpx_group=args.regexpx_group,
        select_random_images=args.select_random_images,
    )

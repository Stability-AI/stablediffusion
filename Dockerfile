#Simple NVIDIA Stability-AI Stable Diffusion Docker file
FROM nvidia/cuda:11.6.2-devel-ubi8

#Install required RPM packages
RUN dnf install git python38 python38-devel python38-setuptools mesa-libGLU python38-pip gcc gcc-c++ -y 

# Install Stability-AI's stable diffusion, and required packages
WORKDIR /WORKDIR/
RUN git clone https://github.com/Stability-AI/stablediffusion.git
WORKDIR /WORKDIR/stablediffusion
RUN python3 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3 -m pip install diffusers 
# Bellow installs packages from requirements.txt manually (to ensure it installs). You can remove this if you have no issues
RUN python3 -m pip install timm albumentations==0.4.3 opencv-python pudb==2019.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.4.2 torchmetrics==0.6 omegaconf==2.1.1 test-tube>=0.7.5 streamlit>=0.73.1 einops==0.3.0 transformers==4.19.2 webdataset==0.2.5 open-clip-torch==2.7.0 gradio==3.13.2 kornia==0.6 invisible-watermark>=0.1.5 streamlit-drawable-canvas==0.8.0 -e .
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install -e .

# Install xformers (this will take some time to compile)
WORKDIR /WORKDIR/
RUN ln -s /usr/bin/gcc-$MAX_GCC_VERSION /usr/local/cuda/bin/gcc 
RUN ln -s /usr/bin/g++-$MAX_GCC_VERSION /usr/local/cuda/bin/g++
ENV CUDA_HOME=/usr/local/cuda-11.6
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.6/include
ENV PATH="/usr/local/cuda-11.6/bin:$PATH"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;8.0;8.6"
RUN git clone https://github.com/facebookresearch/xformers.git
WORKDIR /WORKDIR/xformers
RUN git submodule update --init --recursive
# Bellow installs packages from requirements.txt manually (to ensure it installs). You can remove this if you have no issues
RUN python3 -m pip install triton numpy
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install --verbose -e .
WORKDIR /WORKDIR/stablediffusion


# Pre download OpenCLIP model, 
# Idea from: https://github.com/Stability-AI/stablediffusion/issues/73#issuecomment-1343820268
RUN cat <<'EOT' >> OpenCLIP.py
import open_clip
open_clip.list_pretrained()
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
model.eval()
EOT
RUN chmod +xwr OpenCLIP.py
RUN python3 OpenCLIP.py


#Create run script
RUN cat <<'EOT' >> start.sh
#!/bin/bash
python3 -m xformers.info

if [ -z "$CKPT" ]; then
    echo "missing checkpoint file, ex. -e CKPT=CKPTFILENAME"
fi 
if [ -z "$HIGHT" ]; then
    HIGHT=768
fi 
if [ -z "$WIDTH" ]; then
    WIDTH=768
fi 
if [ -z "$STRENGTH" ]; then
    STRENGTH=0.8
fi 

if [ -n "$SCRIPT" ]; then
  case "$SCRIPT" in

    text-to-image)
        if [ -z "$PROMPT" ]; then
            echo "missing prompt , ex. -e PROMPT=\"image of dog\""
        fi
        python3 scripts/txt2img.py --device cuda --prompt "$PROMPT" --ckpt ./mount/$CKPT --config configs/stable-diffusion/v2-inference-v.yaml --H $HIGHT --W $WIDTH 
    ;;

    depth-to-image)
        if [ -z "$IMAGE" ]; then
            echo "missing image file, ex. -e IMAGE=IMAGEFILENAME"
        fi
        if [ -z "$PROMPT" ]; then
            echo "missing prompt , ex. -e PROMPT=\"image of dog\""
        fi
        python3 scripts/gradio/depth2img.py ./mount/$IMAGE "$PROMPT" configs/stable-diffusion/v2-midas-inference.yaml ./mount/$CKPT
    ;;

    img-to-img)
        if [ -z "$IMAGE" ]; then
            echo "missing image file, ex. -e IMAGE=IMAGEFILENAME"
        fi
        if [ -z "$PROMPT" ]; then
            echo "missing prompt , ex. -e PROMPT=\"image of dog\""
        fi
        python3 scripts/img2img.py --prompt "$PROMPT" --init-img ./mount/$IMAGE --strength $STRENGTH  --H $HIGHT --W $WIDTH --ckpt ./mount/$CKPT
    ;;

    inpainting)
        if [ -z "$IMAGE" ]; then
            echo "missing image file, ex. -e IMAGE=IMAGEFILENAME"
        fi
        if [ -z "$MASK" ]; then
            echo "missing mask file, ex. -e MASK=MASKFILENAME"
        fi
        if [ -z "$PROMPT" ]; then
            echo "missing prompt , ex. -e PROMPT=\"image of dog\""
        fi
        python3 scripts/gradio/inpainting.py ./mount/$IMAGE ./mount/$MASK "$PROMPT" configs/stable-diffusion/v2-inpainting-inference.yaml ./mount/$CKPT
    ;;

    superresolution)
        if [ -z "$IMAGE" ]; then
            echo "missing image file, ex. -e IMAGE=IMAGEFILENAME"
        fi
        if [ -z "$PROMPT" ]; then
            echo "missing prompt , ex. -e PROMPT=\"image of dog\""
        fi
        python3 scripts/gradio/superresolution.py ./mount/$IMAGE $PROMPT configs/stable-diffusion/x4-upscaling.yaml ./mount/$CKPT
    ;;
  esac

  else
    echo "missing script statement, ex. -e SCRIPT=text-to-image"
    echo "options: text-to-image , depth-to-image , img-to-img , inpainting , superresolution"
fi
EOT
# Give script correct permissions
RUN chmod +xwr start.sh
# Remove windows encoding issue
RUN sed -i -e 's/\r$//' start.sh


#Create folder to mount, this stores ckpt/image/mask files, (ex. -v <LocalFoulder>:/mount) 
RUN mkdir mount 


#Run start script on start
CMD ["./start.sh"]



#STEP 1, build Docker image
#Make sure you have Docker, Nvidia and Nvidia Cuda drivers installed on host
#copy Dockerfile to current working directory and run:
    #docker build --tag sd-docker . 

#STEP 2, run Docker Image as container
    #docker run examples:
    #text-to-image: docker run --gpus=all -v <YOUR-OUTPUT-DIRECTORY>:/WORKDIR/stablediffusion/outputs -v <YOUR-MOUNT-DIRECTORY>:/WORKDIR/stablediffusion/mount -e SCRIPT="text-to-image" -e PROMPT="a professional photograph of an astronaut riding a horse" -e CKPT="v2-1_768-ema-pruned.ckpt" -e HIGHT=768 -e WIDTH=768 sd-docker 
        #depth-to-image: docker run --gpus=all  -v <YOUR-OUTPUT-DIRECTORY>:/WORKDIR/stablediffusion/outputs -v <YOUR-MOUNT-DIRECTORY>:/WORKDIR/stablediffusion/mount -e SCRIPT="depth-to-image" -e IMAGE="image.png" -e PROMPT="a professional photograph of an astronaut riding a horse" -e CKPT="512-depth-ema.ckpt" sd-docker 
        #img-to-img: docker run --gpus=all  -v <YOUR-OUTPUT-DIRECTORY>:/WORKDIR/stablediffusion/outputs -v <YOUR-MOUNT-DIRECTORY>:/WORKDIR/stablediffusion/mount -e SCRIPT="img-to-img" -e HIGHT=512 -e WIDTH=512 -e IMAGE="text.png" -e PROMPT="a professional photograph of an astronaut riding a horse" -e STRENGTH=0.8 -e CKPT="512-base-ema.ckpt" sd-docker 
        #inpainting: docker run --gpus=all  -v <YOUR-OUTPUT-DIRECTORY>:/WORKDIR/stablediffusion/outputs -v <YOUR-MOUNT-DIRECTORY>:/WORKDIR/stablediffusion/mount -e SCRIPT="inpainting" -e IMAGE="image.png" -e MASK="mask.png" -e  512-inpainting-ema.ckpt -e HIGHT=512 -e WIDTH=512 sd-docker 
        #superresolution: docker run --gpus=all  -v <YOUR-OUTPUT-DIRECTORY>:/WORKDIR/stablediffusion/outputs -v <YOUR-MOUNT-DIRECTORY>:/WORKDIR/stablediffusion/mount -e SCRIPT="superresolution" -e IMAGE="image.png" -e CKPT="x4-upscaler-ema.ckpt" -e HIGHT=3072 -e WIDTH=3072 sd-docker 


#STEP 2 - alternative
    #You may also like to run multiple/different commands in the sandboxed space. you can bypass the start script and get access to the terminal via:
        # docker run -it --gpus=all -v <YOUR-OUTPUT-DIRECTORY>:/WORKDIR/stablediffusion/outputs -v <YOUR-MOUNT-DIRECTORY>:/WORKDIR/stablediffusion/mount --entrypoint /bin/bash <COMPILED-IMAGE-ID-GOES-HERE>
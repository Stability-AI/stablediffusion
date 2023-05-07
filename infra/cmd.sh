mkdir checkpoints
cd checkpoints
wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt
cd ..
mkdir static
python server.py

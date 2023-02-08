set -e

apt-get update && apt-get install ffmpeg libsm6 libxext6 python3 python3-pip python3-dev -y
pip3 install -r requirements.txt

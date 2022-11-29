FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt-get update && apt-get install -y libgl1-mesa-glx

ARG key
RUN pip install git+https://$key@github.com/everguard-inc/eg_data_tools.git
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install -U numpy 
RUN pip install opencv-python==4.5.5.64
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev


ENV APP_PATH="/app"
RUN mkdir -p ${APP_PATH}
WORKDIR ${APP_PATH}

ENV PYTHONPATH "${PYTHONPATH}:/app/"

# docker build -t $(whoami)/stablediffusion2 .
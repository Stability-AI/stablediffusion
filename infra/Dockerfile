FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

RUN apt update && \
	apt install -y \
		git \
		ffmpeg \
		libsm6 \
		libxext6 \
		wget

# Install dependencies
WORKDIR /app
COPY ./app/requirements.txt /app/requirements.txt
COPY ./app/environment.yaml /app/environment.yaml
COPY ./app/setup.py /app/setup.py
RUN conda env create -f environment.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]

# Install xformers for memory efficient flash attention
RUN conda install xformers -c xformers/label/dev

RUN conda init bash
RUN echo "conda activate ldm" >> $HOME/.bashrc

# Install server dependencies
RUN pip install \
	flask==2.3.2 \
	triton==2.0.0.post1

# Copy files into container
COPY ./app /app
COPY ./server.py /app/server.py
COPY ./cmd.sh /app/cmd.sh

# Start server
EXPOSE 80
CMD ["bash", "cmd.sh"]

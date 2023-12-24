FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update

RUN apt-get install python3 python3-pip -y 
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install faster-whisper transformers
#RUN apt install nvidia-cudnn

ADD ./faster-whisper-small-fa /app/faster-whisper-small-fa

#WORKDIR /app

#VOLUME /app/data

#CMD python3 transcriber.py




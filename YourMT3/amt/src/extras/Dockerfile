FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
LABEL maintainer="https://github.com/mimbres/YourMT3" 

ENV TZ=Europe/London \
    DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget curl \
    nano git ffmpeg sox tmux htop 
RUN pip3 install --upgrade pip
RUN pip3 install mirdata mido git+https://github.com/craffel/mir_eval.git \
    matplotlib lightning>=2.0.2 pytest-timeout pytest deprecated librosa \
    einops transformers wandb

CMD [ "/bin/bash" ]
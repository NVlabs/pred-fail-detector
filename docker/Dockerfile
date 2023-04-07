FROM ubuntu:20.04

# specifying which python version
ARG PYTHON_VERSION=3.9
ARG PASS=p-qad
ENV TZ=America/Los_Angeles

# installing other nice functionalities
RUN apt-get update && \
    apt-get -y install sudo

RUN apt-get install htop -y
RUN apt-get install screen -y
RUN apt-get install software-properties-common -y
RUN apt-get install python3.9 -y
RUN apt-get install pip -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Add sudo user p-qad
RUN adduser --disabled-password --gecos '' p-qad
RUN adduser p-qad sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER p-qad

SHELL ["/bin/bash", "-c"]

WORKDIR /home/p-qad
RUN mkdir p-qad

# Install miniconda
RUN sudo apt-get install wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P /tmp/. && \
    chmod +x /tmp/./Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/./Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH /home/p-qad/miniconda3/bin/:$PATH
RUN conda init
CMD source ~/.bashrc

# install python 3.9 in base conda environment
RUN conda install python=3.9

# install pytorch and other things
RUN conda install pytorch=1.9.0 torchvision=0.10.0 torchaudio=0.9.0 cudatoolkit=10.2 -c pytorch -y
RUN pip install torchmetrics==0.6.0
RUN conda install -c conda-forge firefox geckodriver

# install nuplan-devkit
RUN sudo apt-get update && sudo apt-get install git -y
RUN git clone https://github.com/motional/nuplan-devkit.git
WORKDIR nuplan-devkit
RUN git checkout 3c4f5b8dcd517b27cfd258915ca5fe5c54e3cb0c
RUN sed -i "s|torch==|#torch==|g" requirements.txt && sed -i "s|torchvision==|#torchvision==|g" requirements.txt
RUN pip install -e .

# install rewards deterioration detection
WORKDIR /home/p-qad/p-qad
RUN git clone https://github.com/ido90/Rewards-Deterioration-Detection.git && mv Rewards-Deterioration-Detection drd
WORKDIR drd
RUN sed -i "s|import utils|#import utils|g" StatsCalculator.py

# copy p-qad from local to docker and install requirements
WORKDIR /home/p-qad
COPY --chown=p-qad p-qad/ p-qad/
WORKDIR /home/p-qad/p-qad
RUN pip install -r requirements.txt
RUN pip install pathos==0.2.9

# install hj_reachability
WORKDIR /home/p-qad
RUN git clone https://github.com/StanfordASL/hj_reachability.git
RUN cp -r hj_reachability/hj_reachability p-qad/detection/.

WORKDIR /home/p-qad/p-qad
RUN mkdir detection/target_values && \
    mkdir detection/plots && \
    mkdir nuPlan/dataset
RUN pip install shapely==1.8.2
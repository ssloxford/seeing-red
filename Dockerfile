FROM ubuntu:18.04

# avoid questions when installing stuff in apt-get
ARG DEBIAN_FRONTEND=noninteractive

RUN apt -y update

RUN apt -y install build-essential cmake unzip pkg-config
RUN apt -y install libjpeg-dev libpng-dev libtiff-dev
RUN apt -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt -y install libxvidcore-dev libx264-dev
RUN apt -y install libgtk-3-dev
RUN apt -y install libatlas-base-dev gfortran
RUN apt -y install ffmpeg
RUN apt -y install python3 python3-dev python3-opencv python3-pip python3-tk

RUN python3 -m pip install --upgrade pip

RUN pip3 install numpy==1.15.4 scipy==1.4.0 pandas==1.0.4 matplotlib Pillow scikit-image scikit-video scikit-learn==0.23.2
RUN pip3 install cython
RUN pip3 install pymrmr imutils tslearn==0.3.0 xgboost==1.2.1 pyyaml==5.3.1

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /home

CMD bash


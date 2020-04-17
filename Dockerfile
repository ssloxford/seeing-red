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

RUN pip3 install numpy scipy pandas matplotlib Pillow scikit-image scikit-video scikit-learn
RUN pip3 install cython
RUN pip3 install pymrmr imutils tslearn xgboost pyyaml

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /home

CMD bash


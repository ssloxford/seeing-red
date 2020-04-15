# Seeing Red

This repository contains the code for the paper "Seeing Red: PPG Biometrics Using Smartphone Cameras" published in the [15th IEEE Computer Vision Society Workshop on Biometrics](https://www.vislab.ucr.edu/Biometrics2020/).
This work is a collaboration between [Giulio Lovisotto](https://github.com/giuliolovisotto/), [Henry Turner](http://www.cs.ox.ac.uk/people/henry.turner/) and [Simon Eberz](https://www.cs.ox.ac.uk/people/simon.eberz/) from the System Security Lab at University of Oxford.


## Idea
In this work we investigated the use of photoplethysmography (PPG) for authentication.
An individual's PPG signal can be extracted by taking a video with a smartphone camera as users place their finger on the sensor.
The blood flowing through the finger changes the reflective properties of the skin, which is captured by subtle changes in the video color.

<p align="center"><img src="/images/system-overview.png" width="70%"></p>

We collected PPG signals from 15 participants over several sessions (6-11), in each session the participant places his finger on the camera while a 30 seconds long video is taken.
We extract the raw value of the LUMA component of each video frame to obtain the underlying PPG signal from a video.
The signals are then preprocessed with a set of filters to remove trends and high frequency components, and then each individual heartbeat is separated with a custom algorithm.

<p align="center"><img src="/images/preprocessing.png" width="60%"></p>

We designed a set of features that capture the distinctiveness of each individual's PPG signal and we evaluated the authentication performance with a  set of experiments (see [Reproduce Results](#reproduce-results)).

<p align="center"><img src="/images/features.png" width="70%"></p>

## Dataset

The [dataset]() used for this paper has been published online on ORA and can be freely downloaded.
The dataset contains a set of videos for 14 participants who consented to their data being shared, ethics approval number SSD/CUREC1A CS_C1A_19_032.
Each video is a 30 seconds long recording which was taken as the participant kept his index finger on the smartphone camera, see a preview here.

<p align="center"><img src="/images/video-example.gif" width="50%"></p>

## Reproduce Results

The code runs inside a Docker container and requires `docker` and `docker-compose` to be installed in your system.

You might be able to make this work on a generic python/anaconda environment with some effort. 
  
To reproduce these results, follow these steps:
 1. download the dataset used in the paper at ~~missing link~~, place the downloaded `videos` folder in `seeing-red/data/`
 2. build and start the container by running `./start_container.sh`
 3. attach to the container with `docker attach seeingred_er`
 4. run the entire signal analysis pipeline with `python signal_run_all.py` located in `/home/code/` in the container

Results will be produced in several subfolders in `seeing-red/data/`.



 
 



# Seeing Red: PPG Biometrics Using Smartphone Cameras

This repository contains the code for the paper "[Seeing Red: PPG Biometrics Using Smartphone Cameras](https://arxiv.org/pdf/2004.07088.pdf)" published in the [15th IEEE Computer Vision Society Workshop on Biometrics](https://www.vislab.ucr.edu/Biometrics2020/).
This work is a collaboration between [Giulio Lovisotto](https://github.com/giuliolovisotto/), [Henry Turner](http://www.cs.ox.ac.uk/people/henry.turner/) and [Simon Eberz](https://www.cs.ox.ac.uk/people/simon.eberz/) from the [System Security Lab](http://www.cs.ox.ac.uk/groups/seclab/) at University of Oxford.


## Idea
In this work we investigated the use of photoplethysmography (PPG) for authentication.
An individual's PPG signal can be extracted by taking a video with a smartphone camera as users place their finger on the sensor.
The blood flowing through the finger changes the reflective properties of the skin, which is captured by subtle changes in the video color.

<p align="center"><img src="/images/system-overview.png" width="70%"></p>

We collected PPG signals from 15 participants over several sessions (6-11), in each session the participant places his finger on the camera while a 30 seconds long video is taken.
We extract the raw value of the LUMA component of each video frame to obtain the underlying PPG signal from a video.
The signals are then preprocessed with a set of filters to remove trends and high frequency components, and then each individual heartbeat is separated with a custom algorithm.

<p align="center"><img src="/images/preprocessing.png" width="60%"></p>

We designed a set of features that capture the distinctiveness of each individual's PPG signal and we evaluated the authentication performance with a  set of experiments (see [Reproduce Evaluation](#reproduce-evaluation)).

<p align="center"><img src="/images/features.png" width="70%"></p>

## Dataset

The [dataset](https://ora.ox.ac.uk/objects/uuid:1a04e852-e7e1-4981-aa83-f2e729371484) used for this paper has been published online on ORA and can be freely downloaded.
The dataset contains a set of videos for 14 participants who consented to their data being shared, ethics approval number SSD/CUREC1A CS_C1A_19_032.
Each video is a 30 seconds long recording which was taken as the participant kept his index finger on the smartphone camera, see a preview here.

<p align="center"><img src="/images/video-example.gif" width="50%"></p>

## Reproduce Evaluation

The code runs inside a Docker container and requires `docker` and `docker-compose` to be installed in your system.

You might be able to make this work on a generic python/anaconda environment with some effort. 
  
To reproduce the evaluation, follow these steps:
 1. **read the [paper](https://arxiv.org/pdf/2004.07088.pdf)** - this is the only way you will understand what you are doing
 1. Clone this repository
 1. download the [dataset](https://ora.ox.ac.uk/objects/uuid:1a04e852-e7e1-4981-aa83-f2e729371484) used in the paper, unzip the archive and place the downloaded `videos` folder in `seeing-red/data/`
 1. build and start the container by running `docker-compose up -d`
 1. attach to the container with `docker attach seeingred_er`
 1. in the container, `cd /home/code` and run the entire signal analysis pipeline with `python signal_run_all.py`

Results will be produced in several subfolders in `seeing-red/data/`.

## Citation
If you use this repository please cite the paper as follows:
```
@inproceedings{lovisotto2020seeing,
  title={Seeing Red: PPG Biometrics Using Smartphone Cameras},
  author={Lovisotto, Giulio and Turner, Henry and Eberz, Simon and Martinovic, Ivan},
  booktitle={IEEE 15th Computer Society Workshop on Biometrics},
  year={2020}
}
```

## Contributors
 * [Giulio Lovisotto](https://github.com/giuliolovisotto/)
 * [Henry Turner](http://www.cs.ox.ac.uk/people/henry.turner/)
 * [Simon Eberz](https://www.cs.ox.ac.uk/people/simon.eberz/)

## Acknowledgements

This work was generously supported by a grant from Mastercard and by the Engineering and Physical Sciences Research Council \[grant numbers EP/N509711/1, EP/P00881X/1\].
 
 



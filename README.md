# Seeing Red

This repository contains the code for the paper [Seeing Red: PPG Biometrics Using Smartphone Cameras]() published in the [15th IEEE Computer Vision Society Workshop on Biometrics](https://www.vislab.ucr.edu/Biometrics2020/)

### Reproduce Results

The code runs inside a Docker container and requires `docker` and `docker-compose` to be installed in your system.

You might be able to make this work on a generic python/anaconda environment with some effort. 
  
To reproduce these results, follow these steps:
 1. download the dataset used in the paper at ~~missing link~~, place the downloaded `videos` folder in `seeing-red/data/`
 2. build and start the container by running `./start_container.sh`
 3. attach to the container with `docker attach seeingred_er`
 4. run the entire signal analysis pipeline with `python signal_run_all.py` located in `/home/code/` in the container

Results will be produced in several subfolders in `seeing-red/data/`.



 
 



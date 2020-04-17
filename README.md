# Seeing Red: PPG Biometrics Using Smartphone Cameras

This repository contains the code for the paper "[Seeing Red: PPG Biometrics Using Smartphone Cameras](https://arxiv.org/pdf/2004.07088.pdf)" published in the [15th IEEE Computer Vision Society Workshop on Biometrics](https://www.vislab.ucr.edu/Biometrics2020/).
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
 1. download the [dataset](https://ora.ox.ac.uk/objects/uuid:1a04e852-e7e1-4981-aa83-f2e729371484) used in the paper, unzip the archive and place the downloaded `videos` folder in `seeing-red/data/`
 1. build and start the container by running `docker-compose build`
 1. attach to the container with `docker attach seeingred_er`
 1. in the container, `cd /home/code` and run the entire signal analysis pipeline with `python signal_run_all.py`

Results will be produced in several subfolders in `seeing-red/data/`.

## License

The Clear BSD License + Commons Clause

Copyright (c) 2019-2020,Giulio Lovisotto, Henry Turner, Simon Eberz
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

“Commons Clause” License Condition v1.0

The Software is provided to you by the Licensor under the License, as defined below, subject to the following condition.

Without limiting other conditions in the License, the grant of rights under the License will not include, and the License does not grant to you, the right to Sell the Software.

For purposes of the foregoing, “Sell” means practicing any or all of the rights granted to you under the License to provide to third parties, for a fee or other consideration (including without limitation fees for hosting or consulting/ support services related to the Software), a product or service whose value derives, entirely or substantially, from the functionality of the Software. Any license notice or attribution required by the License must also include this Commons Clause License Condition notice.

## Acknowledgements

This work was generously supported by a grant from Mastercard and by the Engineering and Physical Sciences Research Council \[grant numbers EP/N509711/1, EP/P00881X/1\].
 
 



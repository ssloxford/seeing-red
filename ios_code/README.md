# Heartbeat iOS App Source Code
- This folder contains the source code for the iOS app used in capturing the data.
- The code is not well documented, and none of us are iOS developers, so it should be seen as best effort, and is provided as is with no guarantees as to its performance.
- The code was originally written and used in late 2019 and early 2020, and worked on an iPhone X. 
- As of today (17/6/21) the code has been checked and updated slightly to run on an iPhone 12 Mini running iOS 14.5.1, with Xcode 12.5.
## Provided code
- When we conducted the experiment we uploaded videos directly to a web server. To reduce complexity we have removed this.
- Likewise we have removed any parts for registering demographics etc. of users.
- The code allows you to capture a measurement with the same parameters as used in the paper.
- We compressed videos before uploading them, this compression remains. After completing a measurement two videos will be saved in the camera roll of the device: 1 original and 1 compressed.
## Usage
- If you use any of this code in some work please cite the paper, detailed in the top level readme.
### Derived files
Several files have been derived from other sources, in particular the HRCamMan and VIExportSession files. Both of these contain a link to the original in the comments at the top of the file. Both of these files are therefore covered by the MIT license that their original repos bestow upon them

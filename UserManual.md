
Cellects User Manual
====================

Foreword
--------
<div style="text-align: justify">
Cellects analyzes organisms that grow or move on an immobile surface. 
In the first section of this user manual, we will explain how to set up a first quick analysis with Cellects. 
Any Cellects analysis takes place in 3 steps: data specification, first image analysis and video tracking. 
The second section of this user manual runs thoroughly through every option to run Cellects in particular conditions 
and to finetune an already working setup.

Part one: Setting up a first analysis
--------
## 1/ The first window: data specification
<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure1.png"
  alt="Alt text"
  title="Figure 1: Cellects first window"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
#### **Figure 1**: Cellects first window

On first use (Fig. 1), Cellects needs to identify what type of data to look for.  
Cellects requires a folder containing one stack of images or one video to run an analysis (See Fig. 1). 
When the user selects several folders, Cellects uses the parameters filled for the first one 
to analyze all the remaining folders (See Fig. 10: the several folder window). 
The available variables are shown in Fig. 9: the required output window.

### Image list or video:
The *Image list or video* option indicates whether the data have been stored as an image stack 
(i.e. a set of files, each of them containing a single image) or as a video. 
Images must be named alphanumerically so the program can read them in the right order.

### Image prefix (optional) and extension:
The *Images prefix* and *Images extension* fields allow Cellects to only consider relevant data. 
For instance, setting “exp_” as image prefix and “.jpg” as image extension will cause Cellects to only consider JPG 
files whose name starts with “exp_”. 
The rest of the labeling should be a number indicating the order in which the images were taken.

<br />

Additional note:
- If every .jpg files start with IMG_ but the folder(s) also contains other .jpg files (e.g. named info.jpg), 
the user can exclude all .jpg files that do not start with IMG_ by typing “IMG_” in the *Image prefix* field. 
Cellects accepts all the following formats: bmp, dib, exr, exr, hdr, jp2, jpe, jpeg, jpg, pbm, pfm, pgm, pic, png, 
pnm, ppm, ras, sr, tif, tiff, webp, cr2, cr3, nef, arw, sr2, raf, prf, rw2, pef, dng, 3fr, iiq.

### Folder
The *Folder* field must contain the computer path toward the folder(s) for Cellects to be able to run the analysis. 
The user can copy/paste this path into the field or navigate to the folder using the *Browse* push-button. 
If the user wants to analyze several folders at once, the chosen path must lead to the folder containing all folders 
to analyze.

### Arena number per folder
The *Arena number per folder* tells how many arenas are present in the images. 
Then it will store and analyze the video for each arena separately.

<br />

Additional note:
- If there are several folders to analyze at once, the user can provide a different arena number for each folder
  (see Fig. 10: the several folder window).

### Advanced parameters, Required outputs
These options are detailed in part two: Improving and personalizing the analysis (Fig. 8 and 9).

### Video analysis window and Run all directly
These options appear when the user already did the image analysis for the current folder. 
These shortcuts allow to skip the image analysis and to directly improve the video tracking or 
to run directly the complete analysis.

### Next
Click the *Next* button to go to the image analysis window (Fig. 2), or 
to the window showing the list of folders (Fig. 10) if applicable.

## 2/ The image analysis window: find where the specimens are
<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure2.png"
  alt="Alt text"
  title="Figure 2: Cellects image analysis window"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

Image number (optional)



</div>

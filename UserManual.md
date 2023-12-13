
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

# Table of content
1. [Part one: Setting up a first analysis](#-Part-one:-Setting-up-a-first-analysis)

   2. [2/ The image analysis window: find where the specimens are](#2/-The-image-analysis-window:-find-where-the-specimens-are)

2. [Part two: Improving and personalizing the analysis](#-Part-two:-Improving-and-personalizing-the-analysis)

## Part one: Setting up a first analysis

## 1/ The first window: data specification
<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure1.png"
  alt="Alt text"
  title="Figure 1: Cellects first window"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
#### <u>**Figure 1**: Cellects first window</u>

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

### 2/ The image analysis window: find where the specimens are
<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure2.png"
  alt="Alt text"
  title="Figure 2: Cellects image analysis window"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
#### <u>**Figure 2**: Cellects image analysis window</u>

### Image number (optional)
Which image should be analyzed first? In most cases, it should be the first. 
Changing this number is only useful when cells are invisible on the first image 
(e.g. in the case of appearing colonies of bacteria). 
In that case, select an image showing visible cells in order to enable Cellects to find them.

### Scale with
The *Scale with* option gives a scale to the image. 
Cellects can determine this scale using the width (horizontal size) of the image or the width of the specimens 
on the first image (when they share the same width, 
Cellects can use the average pixel width of all specimens to get the scale). 

Additional notes:
- As using the width of the specimens decreases the first image detection efficiency, we recommend choosing the width of the image. However, if the width of the specimens is known with more accuracy than the width of the image, we recommend choosing the width of the specimens.
- By default, distances and surfaces are in pixels (Cellects store the size of one pixel in a file called software_settings.csv). They can automatically be converted in mm (²) by checking the corresponding checkbox in the advanced parameters window (see Fig. 8).

### Select and draw (optional)
*Select and draw* is a tool allowing the user to inform Cellects that some parts of the image are specimens (Cell) 
and others are background (Back). 
To use that tool, the user must click once on the *Cell* button (to draw a part of the image containing specimens) or 
on the *Back* button (to draw a part of the image containing background). 
The color of the clicked button changes and the user can click and move the cursor on the image to draw the position 
of the specimens or of the background. Each drawing will also appear (with a number) below the corresponding button. 
If the user clicks on one of these numbered drawings, the corresponding selected area disappears, 
enabling the user to correct mistakes.

Additional note:
- If the user wishes to analyze several folders, the *Select and draw* option will only work for the first. 
  If each folder requires using this option, the user has to analyze each folder separately.

### Advanced mode (optional)
The *Advanced mode* allows Cellects to use a previously working set of parameters (see features detailed below).
Except if already familiar with color spaces, the user should not use the *Advanced mode* before using 
one of the *Generate analysis options*. 

Once generated, selecting these options with *Advanced mode* activated allow to:
- See the color space combination corresponding to the displayed image.
- Try a new color space combination by mixing two good options. 
Modify the color space fields and click *Visualize*. 
E.g. If hsv(0, 1, 0) and lab(0, 0, 1) give good results independently, 
the user can try to combine them to get better results. Select the hsv(0, 1, 0) option, 
click the + button and add lab(0, 0, 1) in the new fields. 
- Use a logical operator between the results of two color space combinations. 
Choosing a logical operator will make new color space combination fields appear. 
When the user fills in these fields and clicks *Visualize*, Cellects computes both color space combinations, 
gets a detection result for each, and applies the chosen logical operator between these detections. 
E.g. If the two detections using hsv(0, 1, 0) and lab(0, 0, 1) show some noise 
(many little dots appear on the detection), using the logical operator AND between them will reduce it. 
If detection is insufficient for both, using the logical operator OR can allow one detection to complement the other. 
- Use an algorithm (k-means) that categorizes the image into more than 2 categories. 
The *Heterogeneous background* option appears after the use of the Select and draw option. 
If the detection remains not good enough after having drawn a few specimens and the problematic parts of the background, 
the user should try this algorithm. The user should first check the Heterogeneous background option, 
then select the number of categories to look for in the image and lastly click on *Visualize*. 
As this option reduces the video tracking option number from 5 to 1, it should be used as a last resort. 

### One cell/colony per arena
This option is automatically selected. If there is only one cell (or connected colony) per arena, leave this as it is. 
If there already are (or will be) several cells (or colonies) per arena, unselect this option.

### Generate analysis option
Cellects suggests two algorithms to automatically find the best parameters to detect where the specimens are 
on the first image: clicking on *Quickly* suggests different options after a few minutes at most, 
clicking *Carefully* browses more possibilities and takes more time. If the user runs Cellects for the first time, 
the user should use one of these options. Alternatively, the user can select the advanced mode to view the parameters 
selected by Cellects to analyze the image. By taking note of the parameters that work well, the user will be able to 
use Cellects more quickly via the *Visualize* button, which directly displays the result produced by the parameters 
stored by the software and modifiable by the user.

Additional notes:
- Clicking on the *Quickly* or *Carefully* (or *Visualize*) button will make an orange working message appear. 
Once that message disappears, new option(s) are generated and the image in the center changes accordingly (See Fig. 3).
- If the user already used Cellects, the advanced mode will be faster. Once that option is checked, 
the user can use and adjust a previously working color space combination and click on the *Visualize* button. 
Instead of exploring many color space combinations, *Visualize* will analyze only one. 

<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure3.png"
  alt="Alt text"
  title="Figure 3: Cellects image analysis window after analysis option generation"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

#### <u>**Figure 3**: Cellects image analysis window after analysis option generation</u>

### Select option to read
The drop-down menu on the left of *Select option to read* allows the user to visualize directly the result of 
the analysis corresponding to each option. For instance, in Fig. 3, the central image displays the result of option 1 
and a message below informs the user that this option detected 6 distinct spots in 6 arenas. 
Since these two numbers are equal there is only one specimen per arena, the user should click the *Yes* button.

Otherwise, the user can improve the analysis by setting the arena shape, 
the spot shape(i.e. the shape of each cell to look for in the first image), or 
the spot size (i.e. the horizontal size of  each cell). The user can also draw more specimens or background using 
the *Select and draw tool* or even use the Visualize button to set up manually a color space combination.

Once the magenta/pink contours correspond to the right position of all cells and the right number of detected spots, 
the user can click the *Yes* button (on the left of the question “*Does the color match the cell(s)?*”).
After having clicked on the *Yes* button, an orange working message appears, 
Cellects is automatically looking for the coordinates of the contour of each arena. 

### Video delimitation
After some time, the result of the automatic video delimitation appears in blue in the image at the center of the window
(See the image at the center of Fig. 4). If the video delimitation is correct, click the *Yes* button. 
If it does not, clicking the *No* button will lead the user to another choice: 
“*Click Yes to try a slower but more efficient delineation algorithm, No to do it manually*”. 
If the slower algorithm does not work either, Cellects suggests the manual option again. 
To manually draw each arena, the user has to use a tool similar to c) *Select and draw* and click *Yes* when it is done.

<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure4.png"
  alt="Alt text"
  title="Figure 4: Cellects image analysis window, after arena delineation, differences between first and last image"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

#### <u>**Figure 4**: Cellects image analysis window, after arena delineation, differences between first and last image</u>

### Whether starting area differs from growing area
In our example (Fig. 4), the specimens start upon an oat gel while the rest of the arena 
is only made with transparent agar gel. Therefore, the starting area is not exactly of the same color as 
the growing area (the rest of the arena). That is why this checkbox is checked. 

In all other cases, i.e. when the substrate upon which the specimen grows or moves has the same color everywhere: 
uncheck that option.

### Last image question (optional)
If the user thinks that the parameters used to detect specimens on the first image might not work for all the images, 
the user can fine tune these parameters and see whether they work to detect specimens on the last image.

Clicking *Yes* allows the user to display and analyze the first image and then to go to the video tracking window. 
Clicking *No* will directly lead to the video tracking window.

## 3/ The video tracking window: tune dynamical parameters and analysis
<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure5.png"
  alt="Alt text"
  title="Figure 5: Cellects video tracking window"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

#### <u>**Figure 5**: Cellects video tracking window</u>

### Arena to analyze
The arena to analyze is a number allowing Cellects and the user to identify one arena in particular. 
If there is only one arena, this number should be one. Cellects automatically names the arena according to 
their position in the image, from left to right and from top to bottom.

### Maximal growth factor
Briefly, this factor should be increased (resp. decreases) if the analysis underestimates (resp. overestimates) 
the cell size. The maximal growth factor is a proportion of pixels in the image. 
It indicates Cellects how far the specimen can possibly move or how much it can grow from one image to the next. 

More precisely, it is the upper limit of the proportion of the image that can go from being detected as background 
to being detected as specimens. 

### Repeat video smoothing
Increasing this value will decrease noise when the slope option is chosen. 
Cellects analyzes the intensity dynamics of each pixel in the video. 
When it uses a threshold algorithm based on changes of the derivative (slope evolution), 
it starts by smoothing every pixel intensity curve. The algorithm used to smooth these curves (a moving average) 
can be repeated to make the curve as smooth as necessary. 
This value corresponds to the number of times to apply this algorithm.

### Select analysis option and compute all options
Cellects offers five analysis options of video tracking. The user can choose one option or let the 
“*Compute all options*” checkbox be checked before starting one detection. 
- The frame option applies the algorithm used during the image analysis window, frame by frame, 
without any temporal dynamics.
- The threshold option compares the pixel intensity with the average intensity of the whole image at each time step.
- The slope option compares the slope of the pixel intensity with an automatically defined slope threshold.
- The T and S option is the logical result of the threshold option AND the slope option.
- The T or S option is the logical result of the threshold option OR the slope option.

Additional note:
- When the Heterogeneous background has been checked in the image analysis window, 
only the frame option remains available.

### Load one arena
Clicking this button will simply load the arena corresponding to the number in the field of “*Arena to analyze*”. 
The center of the window displays the video of the corresponding arena. 

### Detection
*Detection* runs one or all options of video tracking for the chosen arena to analyze. 
It allows us to test the effect of any change in the previous parameters of that window. 
Once the detection seems valid, the user can answer “*Done*” to *Step 1: Tune parameters to improve Detection*, 
and try *Post processing*.

<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure6.png"
  alt="Alt text"
  title="Figure 6: Cellects video tracking window during detection visualization"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

#### <u>**Figure 6**: Cellects video tracking window during detection visualization</u>

### Fading detection (optional)
*Fading detection* (Fig. 6) is useful when the specimens not only grow but also move 
(and therefore, a pixel that was covered by the cell may become uncovered). 
When checked, *Fading detection* will monitor cell-covered parts of the video to decide whether and when the area 
has been abandoned by the specimens. This parameter can take a value between -1 and 1 to fine tune fading detection. 
Near - 1, Cellects will almost never detect when the cell leaves an area. Near 1, Cellects may remove cells’ detection 
everywhere in the video (in other words, consider pixels as abandoned by the specimens while they were not). 
A too high fading value can also cause waves-like patterns in the detection videos.

### Post processing
*Post processing* (Fig. 6) will apply the chosen detection algorithm to the video, 
as well as other algorithms allowing to improve detection. These algorithms are:
- *Fading detection*, described above
- *Correct errors around initial shape* (fine tuning in the advanced parameters window, see Fig. 8)
- *Connect distant shapes* (fine tuning in the advanced parameters window, see Fig. 8)
- Operations on binary images such as opening, closing and logical operations.
- 
Once post processing works fine, the user can click “*Done*” to *Step 2: 
Tune fading and advanced parameters to improve post processing*, and analyze all arenas (*Run All*).

<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure7.png"
  alt="Alt text"
  title="Figure 7: Cellects video tracking window, running all arenas"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

#### <u>**Figure 7**: Cellects video tracking window, running all arenas</u>

### Run All
If detection with *Post processin*g leads to a satisfying result for one arena,
the user can try to apply the current parameters to all arenas. 
Clicking the *Run All* button will start by writing the video of each arena in the current folder 
(without compression, this may take a lot of disk space). 
Then, Cellects will analyze videos, one by one (See Fig. 7), 
and save the *Required output* (see Fig. 9) as .csv files, a validation video of each arena, 
and two snapshots of all videos (one after one tenth of the total time and one at the last image).

### Save one result (optional)
This button should be used only after having *Run all* at least one time. For instance, 
if the chosen parameters work fine for 5 arenas over 6, the user can use this Cellects window to change the result of 
the failing arena. To do so, the user must click the *Run All* button one time with the parameters that work most of 
the time (9 over 10), and then, adjust to parameters so that the failed analysis gets better after post processing, 
and finally click on the *Save one result* button (Fig. 7) to replace the saved results of that arena 
(i.e. every corresponding row of the .csv files and the validation video).

## Part two: Improving and personalizing the analysis

## 1/ Advanced parameters

<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure8.png"
  alt="Alt text"
  title="Figure 8: advanced parameters window"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

#### <u>**Figure 8**: advanced parameters window</u>

### Automatically crop images (optional)
This parameter uses the first image detection to crop all images and improve arena and last image detection. 

Additional note:

If the analysis fails or the program crashes while running the image analysis window, 
unselecting that option may solve the problem.

### Subtract background (optional)
This parameter has an impact on almost all steps of the analysis. 
Basically, it takes the first image and subtracts it to every following image before analysis. 
It is mostly useful when the cells are not yet visible in the first image, and according to the setup, 
that may improve or degrade the analysis.

### Correct errors around initial shape (optional)
Only apply this algorithm when there are good reasons to think that detection will be less efficient around the 
initial shape than everywhere else in the arena. This algorithm compensates for this lack of efficiency by 
artificially filling potential gaps around the initial shape during growth.

In addition, we discourage the use of this option if the growth substrate has the same opacity everywhere 
(i.e. if it does not differ between the starting region and the growth region).

### Connect distant shapes (optional)
Only apply this algorithm when there is a large light intensity variation within the specimen and 
no reason for it to be split in two disconnected shapes. 
It allows a disconnected shape to be dynamically connected to the main shape. 
This algorithm is very useful when the color of some parts of the specimen is so close to the background that 
a disconnection appears during detection. This algorithm will fix that apparent disconnection.

### All cells have the same direction
This parameter only affects the slow algorithm of automatic arena detection (i.e. when the fast failed).
Applying this will improve the chances to correctly detect arenas when all cells move in the same direction.

Uncheck that option only when specimens move strongly and in many directions.

### Automatic size threshold for appearance/motion (optional)
Allows the user to set a minimal size threshold (in pixels) to consider that an appearing shape is, for instance, a colony of bacteria.

### Oscillatory parameter (optional)
The user can ask Cellects to proceed an oscillatory analysis of the specimens (See Fig. 9: the required output window). 
The oscillatory period is a parameter allowing Cellects to know what kind of oscillatory process to look for in the variations of luminosity.

### Spatio-temporal scaling (optional)
These parameters are very important as they allow the user to set the spatiotemporal scale. 
The first one sets the time (in minute) between each image of an image stack or frames of a video. 
The second one asks the user if Cellects should convert areas and distances from pixels to mm/mm2 or not.

### Run analysis in parallel (optional)
Letting this parameter be checked will use more processor (CPU) capacity during the detection of several arenas but 
will decrease the time necessary for the analysis.

### Proc max core number (optional)
The maximal number of  logical cores to be invested in the detection of several arenas. 
Generally, if the user allocates more cores to the analyses, they will decrease the time necessary for the analysis .

### Minimal RAM let free
This parameter sets how much RAM will remain available to run other programs while Cellects is running. 
RAM is critical for image and video analysis. If Cellects cannot run because it lacks RAM, 
the user can try to fix this value to 0. However, it will increase the risk that Cellects crashes during the analysis, 
especially if the user opens other applications while Cellects is running.

### Lose accuracy to save RAM
Apply this algorithm when there the computer does not have enough RAM to run the analysis. 

Additional note:

- When applied, this algorithm types the video in uint8 (instead of float64) to save RAM but results in a slight loss of precision.

### Video fps
The number of images per second of the validation videos.

### Keep unaltered videos (optional)
Keeping unaltered videos decreases the time necessary for the analysis. 
However, these files (.npy) take a lot of disk space. 
If the user chooses to keep these unaltered videos while improving the analysis, 
we advise the user to delete these large files when the analysis is fine tuned and completed.

### Saved processed video (optional)
If this option is selected, Cellects saves the validation videos. 
These files are much lighter than the unaltered video and can be read by any video reader to check the efficiency of 
the analysis. We advise the user to keep these videos instead of the unaltered videos.

### Color space combination for video analysis
For advanced users, this color space combination tool allows the user to change the way rgb images are processed 
without coming back to the image analysis window (while not having to redo the whole process of image analysis and 
video delimitation). It allows, from the video tracking window, to directly check the effect of changing from one color
space combination to another on the video detection.

### Heterogeneous background (optional)
For advanced users,  this tool allows to segment (technical term for detect) the image into 2 or 
more categories using a k-means algorithm. It is much slower than the usual threshold detection, 
but allows the specimen detection in a more complex environment (basically, with more different colors).

### Night mode
Change the background of the application, from bright to dark

## 2/ Required outputs

<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure9.png"
  alt="Alt text"
  title="Figure 9: required output window"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

#### <u>**Figure 9**: required output window</u>

Selecting one descriptor will save it in .csv files at the end of the analysis (*Run all*).  
At the time of the creation of this user manual, the available required outputs are: area, perimeter, circularity, 
rectangularity, total hole area, solidity, convexity, eccentricity, euler number, standard deviation over x and y, 
skewness over x and y, kurtosis over x and y, major axes lengths and angles. 
The *Try to detect growth transition* algorithm has been developed to automatically detect the transition from 
an isotropic to a digitated growth (Vogel et al. 2016). The *Proceed oscillation analysis* has been developed to 
study the intracytoplasmic oscillations of P. polycephalum following the ideas developed in 
a literature review by Boussard et al. (2021).

## 3/ Analyze multiple folders at once

<img
  src="https://github.com/Aurele-B/Cellects/blob/main/screenshots/UserManualFigure10.png"
  alt="Alt text"
  title="Figure 10: select folders to analyze window"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

#### <u>**Figure 10**: select folders to analyze window</u>

The window of Figure 2 fig. 10 appears after the first window when the selected folder contains 
several folders containing images corresponding to the *Image prefix* and *Images extension* patterns. 
In this window, the user can adjust the number of arenas for each folder. 
Before clicking *Next*, all folders to analyze must be selected (Ctrl/Cmd + Click to make multiple selections). 
Click the *Next* button to go to the image analysis window (Fig. 2) of the first selected folder.

Additional notes:

- If the *Select and draw* option is necessary to analyze the first image of more than one folder, 
the user must analyze them separately. In that case, click *Previous* and change the path to select only one sub-folder. 
If only one folder requires the use of the *Select and draw option*, 
the user can make sure to analyze this folder first. To do so, deselect all folders, 
select the one to be analyzed first and hold Ctrl/Cmd while selecting the remaining folders.
- If the first selected folder has already been analyzed, Cellects suggests the user to use two shortcuts: 
  - Clicking the *Video analysis window* skips the image analysis window and allows the user to make sure that 
  all video tracking settings are good before analyzing all folders.
  - Clicking *Run all* directly skips both the image analysis window and the video tracking window, 
  and directly applies all saved settings to all selected folders.


</div>

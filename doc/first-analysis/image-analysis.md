# Find where the specimens are in the image analysis window

<figure>
  <img src="/static/UserManualFigure2.png" alt="Cellects image analysis window" width="600">
  <figcaption><strong>Figure 2:</strong> Cellects image analysis window</figcaption>
</figure>

---

<!-- START_Image_number -->
## Image number:
Selects the image number to analyze. This number should only be changed when specimen(s) are
invisible on the first image (e.g. in the case of appearing colonies of bacteria), never otherwise.
When the specimen(s) are invisible, read more advanced images until some material can be detected.
NB:
- When the data is stored as images, this image number comes from the alphanumerical sorting of the
original image labeling.

<!-- END_Image_number -->

---

<!-- START_several_blob_per_arena -->
## One specimen per arena:
Should be selected if there is only one specimen (e.g. a cell or a connected colony) per arena.   If
there already are (or will be) several specimen(s) per arena, unselect this option.
NB:
- This option is selected by default.

<!-- END_several_blob_per_arena -->

---

<!-- START_Scale_with -->
## Scale with:
Set how the true pixel size (in mm) should be computed. to calculate pixel size  Cellects can
determine this scale using the width (horizontal size) of the image or the width of the specimens on
the first image (ideally in cases where they share the same width).
NB:
- Cellects' advanced parameters include the possibility to disable this scaling and get all outputs
in pixels.
- Using the width of the specimens decreases the first image detection efficiency, we recommend
choosing the width of the image.
- However, if the width of the specimens is known with more accuracy than the width of the image,
choose the width of the specimens.
- By default, distances and surfaces are in pixels (Cellects stores the size of one pixel in a file
called `software_settings.csv`).

<!-- END_Scale_with -->

---

<!-- START_Scale_size -->
## Scale size:
The *Scale size* is the length (in mm) of the item(s) used for scaling.

<!-- END_Scale_size -->

---

<!-- START_Select_and_draw -->
## Select and draw:
*Select and draw* is a tool allowing the user to inform Cellects that some parts of the image are
specimens *Cell* and others are background *Back*. To use that tool, the user must click once on the
*Cell* button (to draw a part of the image containing specimens) or  on the *Back* button (to draw a
part of the image containing background). The color of the clicked button changes and the user can
click and move the cursor on the image to draw the position  of the specimens or of the background.
Each drawing will also appear (with a number) below the corresponding button. If the user clicks on
one of these numbered drawings, the corresponding selected area disappears, enabling the user to
correct mistakes.
NB:
- If the user wishes to analyze several folders, the *Select and draw* option will only work for the
first.
- If each folder requires using this option, the user has to analyze each folder separately.

<!-- END_Select_and_draw -->

---

<!-- START_Draw_buttons -->
## Draw buttons:
Click the *Cell* or *Back* button and draw a corresponding area on the image by clicking and holding
down the mouse button.

<!-- END_Draw_buttons -->

---

<!-- START_Advanced_mode -->
## Advanced mode:
The *Advanced mode* allows the user to fine tune the image analysis parameters. This can be useful
to use previously working set of parameters on similar images, or to test the available methods
directly. Even when some analysis option are generated, selecting this option can be useful to
access:
- The color space combination corresponding to the displayed image.
- Various filter to apply on the image before segmentation.
- Other results by adding good channels together or mixing two good options using a logical operator
between them.
- The grid segmentation algorithm.
- The kmeans segmentation algorithm.

<!-- END_Advanced_mode -->

---

<!-- START_Color_combination -->
## Logical operator:
The logical operator to apply between the result of two distinct segmentations. For instance, using
two filters or two color space combinations.

<!-- END_Color_combination -->

---

<!-- START_Filter -->
## Filter:
The filter to apply to the image before segmentation

<!-- END_Filter -->

---

<!-- START_Rolling_window_segmentation -->
## Rolling window segmentation:
Segment small squares of the images to detect local intensity valleys This method segment the image
locally using otsu thresholding on a rolling window

<!-- END_Rolling_window_segmentation -->

---

<!-- START_Kmeans -->
## Kmeans:
The Kmeans algorithm will split the image into categories (a number between 2 and 5) and find the
one corresponding to the specimen(s)

<!-- END_Kmeans -->

---

<!-- START_Generate_analysis_options -->
## Generate analysis options:
Cellects suggests an algorithms to automatically find the best parameters to detect specimens on the
first image:
- **Basic** → suggests options in a few minutes.   Alternatively, the user can select *Advanced
mode* to view or modify the parameters selected by Cellects.
NB:
- Clicking on *Basic* (or *Apply current config*) will provoke the display of a working message (in
orange).

<!-- END_Generate_analysis_options -->

<figure>
  <img src="doc/static/UserManualFigure3.png" alt="Cellects image analysis window after analysis option generation" width="600">
  <figcaption><strong>Figure 3:</strong> Cellects image analysis window after analysis option generation</figcaption>
</figure>

---

<!-- START_Select_option_to_read -->
## Select option to read:
Select the option allowing the best segmentation. This dropdown menu appears after generating
analysis options. Each available option allows the user to directly assess their quality.  For
instance, if option 1 generate a message informing the user that this option detected 6 distinct
spots in 6 arenas and these correspond to the user's expectations they should click the *Yes*
button.  Otherwise, Cellects offers several ways to improve the analysis:
- Setting the *arena shape*, the *spot shape*, or the *spot size*.
- Drawing more specimens or background using *Select and draw*.
- Tuning the advanced mode to set up other methods manually.
-> Each of these options can then be tested using the *Apply current config* button.
NB: Once the magenta/pink contours correspond to the right position of all cells and the right
number of detected spots, the user can click the *Yes* button. After clicking, an orange working
message appears, and Cellects automatically looks for the coordinates of the contour of each arena.

<!-- END_Select_option_to_read -->

---

<!-- START_Arena_shape -->
## Arena shape:
Specifies whether the specimen(s) can move in a circular or rectangular arena.

<!-- END_Arena_shape -->

---

<!-- START_Spot_shape -->
## Set spot shape:
Initial shape of the specimen(s) inside arena(s).

<!-- END_Spot_shape -->

---

<!-- START_Spot_size -->
## Set spot size:
Initial horizontal size of the specimen(s) (in mm). If similar across all specimens, this can also
be used as a scale.

<!-- END_Spot_size -->

---

<!-- START_Video_delimitation -->
## Video delimitation:
After validating the initial detection, the result of the automatic video delimitation appears in
blue in the  center of the window.   If correct, click *Yes*.   If incorrect, click *No*, and
Cellects will suggest:
- A slower, more efficient algorithm
- Or a manual delineation option

<!-- END_Video_delimitation -->

<figure>
  <img src="doc/static/UserManualFigure4.png" alt="Cellects image analysis window, after arena delineation" width="600">
  <figcaption><strong>Figure 4:</strong> Cellects image analysis window, after arena delineation</figcaption>
</figure>

---

<!-- START_Last_image_question -->
## Last image question:
If the user thinks that parameters used on the first image might not work on later images, they can
fine
-tune them using the last image.
- Clicking *Yes* → allows testing on the last image before moving on.
- Clicking *No* → goes directly to video tracking.

<!-- END_Last_image_question -->

---

<!-- START_Start_differs_from_arena -->
## Check if the medium at starting position differs from the rest of the arena:
If the substrate changes between starting and growing areas (e.g. nutritive gel vs transparent
agar), keep this checked. If the substrate is homogeneous everywhere, uncheck this option. This
option is only relevant for experiments in which the medium on which the specimen grow does not have
the same optic properties at the position of the specimen(s) at the first frame and at their
positions later on.

<!-- END_Start_differs_from_arena -->

---

<!-- START_Save_image_analysis -->
## Save image analysis:
Complete the analysis of the current image. Clicking this button is useful to analyze only one
image.  To analyze video(s), click *Next*.
NB:
- When there should be only one specimen per arena, keeps the largest connected component.
- Compute and save (.csv) all descriptors selected in the Required output window on the current
image.
- Save a validation image to assess the efficiency of the segmentation.

<!-- END_Save_image_analysis -->

---

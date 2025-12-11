# Tune parameters of the video tracking window

<figure>
  <img src="doc/static/UserManualFigure5.png" alt="Cellects video tracking window" width="600">
  <figcaption><strong>Figure 5:</strong> Cellects video tracking window</figcaption>
</figure>

---

<!-- START_Arena_to_analyze -->
## Arena to analyze:
This arena number allows the user to load one particular arena in the current folder. Typically, the
user can choose an arena, click on *Detection* to load and analyse one arena and *Read* the
resulting analysis.
NB:
- Cellects automatically names the arena according to their position in the image, from left to
right and from top to bottom.
- If there is only one arena, this number should be one.
- *Post processing* automatically runs *Detection* and *Detection* automatically runs *Load One
arena*
- Loading will be faster if videos are already saved as ind_*.npy.

<!-- END_Arena_to_analyze -->

---

<!-- START_Maximal_growth_factor -->
## Maximal growth factor:
The maximal growth factor is a proportion of pixels in the image and indicates how far the specimen
can possibly move or grow from one image to the next. This factor should be:
- Increased if the analysis underestimates the specimen size.
- Decreased if the analysis overestimates the specimen size.
NB:
- Precisely, this is  the upper limit of the proportion of the image that is allowed to be covered
by the specimen  between two frames.

<!-- END_Maximal_growth_factor -->

---

<!-- START_Temporal_smoothing -->
## Temporal smoothing:
The number of times the video will be smoothed. This is useful to accurately detect variations in
pixel slopes. Temporal smoothing reduces variations from noise  and reveals trends occurring at
larger time scales. Technically, Cellects smoothes pixels curves using a rolling window over time.
NB:
- This algorithm is only useful when segmenting with pixel intensity slopes.
- Repeating this algorithm many times makes all pixels constants and prevent any detection.

<!-- END_Temporal_smoothing -->

---

<!-- START_Segmentation_method -->
## Segmentation method:
Cellects includes five video tracking options:
- **Frame option**: applies the algorithm used during the image analysis window, frame by frame,
without temporal dynamics.
- **Threshold option**: compares pixel intensity with the average intensity of the whole image at
each time step.
- **Slope option**: compares slope of the pixel intensity with an automatically defined slope
threshold.
- **T and S option**: logical AND of threshold and slope options.
- **T or S option**: logical OR of threshold and slope options.
NB:
- Selecting the *Compute all options* before dunning *Detection* allows the comparison of these
methods. Once the analysis completed, select one option and click *Read*.
- Computing only one option is faster and requires less memory.
- When *Heterogeneous background* or *Grid segmentation* has been selected in the image analysis
window, only the *Frame* option remains available.

<!-- END_Segmentation_method -->

---

<!-- START_Load_one_arena -->
## Load one arena:
Clicking this button loads the arena corresponding to the *Arena to analyze*. The center of the
window then displays the first frame of the video of that arena. Click *Read* to check the full
video.

<!-- END_Load_one_arena -->

---

<!-- START_Detection -->
## Detection:
*Detection* runs one (or all) segmentation method on one arena. Once finished, click *Read* to see
the detection result.  If correct, answer *Done* to *Step 1: Tune parameters to improve Detection*,
and check the effect of *Post
-processing*.

<!-- END_Detection -->


<!-- START_Read -->
## Read:
Clicking *Read* starts the video display corresponding to the current state of the analysis.

<!-- END_Read -->

<figure>
  <img src="doc/static/UserManualFigure6.png" alt="Cellects video tracking window during detection visualization" width="600">
  <figcaption><strong>Figure 6:</strong> Cellects video tracking window during detection visualization</figcaption>
</figure>

---

<!-- START_Fading_detection -->
## Fading detection:
*Fading detection* monitors how the specimen(s) leave some areas. This is useful when the specimens
not only grow but also move. Uncheck this option otherwise. When the specimen(s) may leave
previously covered areas, set a value between one and minus one to control the strength of that
detection.
- Near minus one: The algorithm will almost never detect when the specimen(s) leave an area.
- Near one: The algorithm may wrongly remove detection everywhere.

<!-- END_Fading_detection -->

---

<!-- START_Post_processing -->
## Post processing:
*Postprocessing* applies the chosen detection algorithm to the video, on top of additional
algorithms to improve it:
- Standard binary image operations: opening, closing, logical ops.
- *Fading detection*: when specimen(s) may leave areas (optional).
- *Correct errors around initial shape*: when the contour of the initial position of the specimen is
hard to detect (optional).
- *Connect distant shapes*: when the specimen's heterogeneity create wrong disconnections in the
video detection (optional).
- *Prevent fast growth near periphery*: when arena's border (typically petri dishes) may be wrongly
detected as specimen (optional). Once Post
-processing works, the user can click “*Done*” to *Step 2: Tune fading and advanced parameters to
improve Post
-processing*, and then *Run All* arenas.

<!-- END_Post_processing -->

<!-- START_Save_one_result -->
## Save one result:
Complete the analysis of the current video. Clicking this button is useful to analyze only one
video. Click *Run All* to analyze all arenas of the current folder. Both options should be done only
once *Postprocessing* gave a satisfying result. Saving one result includes:
- Computing and saving (.csv) all descriptors selected in the Required output window on all frames
of the current video.
- Save one validation video to assess the efficiency of the segmentation on all frames.
- Save the software settings to remember all current parameters.
NB:
- If most arenas analyzed well, but some failed, the user can: reanalyze every arena by adjust some
parameters to fix  any specific problem and click *Save one result* to replace the saved results for
that arena.
- This option modifies the specific row and validation video corresponding to that modified arena.

<!-- END_Save_one_result -->

<figure>
  <img src="doc/static/UserManualFigure7.png" alt="Cellects video tracking window, running all arenas" width="600">
  <figcaption><strong>Figure 7:</strong> Cellects video tracking window, running all arenas</figcaption>
</figure>

---

<!-- START_Run_All -->
## Run All:
If detection with *Postprocessing* leads to a satisfying result for one arena,   the user can apply
the current parameters to all arenas.
-> Clicking *Run All* will:
- Write the uncompressed video of each arena to the folder (can take a lot of space)
- Analyze videos one by one (Fig. 7)
- Compute and save all descriptors selected in the *Required output* window on all frames of the
current video.
- Save one validation video to assess the efficiency of the segmentation on all frames.
- Save two validation images (after 1/10 of total time and at the last image) to assess the
efficiency of the segmentation.
- Save the software settings to remember all current parameters.

<!-- END_Run_All -->

---

<!-- START_Save_all_choices -->
## Save all choices:
Clicking the *Save all choices* write or updates config files to redo an analysis with the same
parameters later on.

<!-- END_Save_all_choices -->

---

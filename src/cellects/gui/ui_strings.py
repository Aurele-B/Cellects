#!/usr/bin/env python3

########################################
#########     First Window     #########
########################################

FW = dict()
FW["Image_list_or_videos"] = {}
FW["Image_list_or_videos"]["label"] = "Image list or videos"
# START_TIP
FW["Image_list_or_videos"]["tips"] = \
f"""The *Image list or video* option indicates whether the data have been stored as an image stack (i.e.
a set of files, each of them containing a single image) or as a video. Images must be named
alphanumerically so the program can read them in the right order.
"""
# END_TIP

FW["Image_prefix_and_extension"] = {}
FW["Image_prefix_and_extension"]["label"] = "Image prefix and extension"
# START_TIP
FW["Image_prefix_and_extension"]["tips"] = \
f"""The *Images prefix* and *Images extension* fields allow Cellects to only consider relevant data. For
instance, setting 'exp_' as image prefix and '.jpg' as image extension will cause Cellects to only
consider JPG files whose name starts with 'exp_'. The rest of the labeling should be a number
indicating  the order in which the images were taken.
NB:
- Image prefix is optional
- If every .jpg files start with IMG_ but the folder(s) also contains other .jpg files (e.g. named
info.jpg),  the user can exclude all .jpg files that do not start with IMG_ by typing “IMG_” in the
*Image prefix* field.  Cellects accepts all the following formats: bmp, dib, exr, exr, hdr, jp2,
jpe, jpeg, jpg, pbm, pfm, pgm, pic, png,  pnm, ppm, ras, sr, tif, tiff, webp, cr2, cr3, nef, arw,
sr2, raf, prf, rw2, pef, dng, 3fr, iiq.
"""
# END_TIP

FW["Folder"] = {}
FW["Folder"]["label"] = "Folder"
# START_TIP
FW["Folder"]["tips"] = \
f"""The *Folder* field must contain the computer path toward the folder(s) for Cellects to be able to
run the analysis.  The user can copy/paste this path into the field or navigate to the folder using
the *Browse* push button.  If the user wants to analyze several folders at once, the chosen path
must lead to the folder containing all folders  to analyze.
"""
# END_TIP

FW["Arena_number_per_folder"] = {}
FW["Arena_number_per_folder"]["label"] = "Arena number per folder"
# START_TIP
FW["Arena_number_per_folder"]["tips"] = \
f"""The *Arena number per folder* tells how many arenas are present in the images. Then it will store
and analyze the  video for each arena separately.
NB:
- If there are several folders to analyze at once, the user can provide a different arena number for
each folder (see Fig. 10: the several folder window).
"""
# END_TIP

FW["Browse"] = {}
FW["Browse"]["label"] = "Browse"
# START_TIP
FW["Browse"]["tips"] = \
f"""Clicking the *Browse* button helps to find and open a folder to analyze.
"""
# END_TIP

FW["Required_outputs"] = {}
FW["Required_outputs"]["label"] = "Required outputs"
# START_TIP
FW["Required_outputs"]["tips"] = \
f"""Clicking the *Required outputs* button opens the window allowing to choose what descriptors Cellects
will compute on the selected data. Find details about this window in the advanced documentation.
"""
# END_TIP

FW["Advanced_parameters"] = {}
FW["Advanced_parameters"]["label"] = "Advanced parameters"
# START_TIP
FW["Advanced_parameters"]["tips"] = \
f"""Clicking the *Advanced parameters* button opens the window containing all secondary parameters of
the software.  Find details about this window in the advanced documentation.
"""
# END_TIP

FW["Run_all_directly"] = {}
FW["Run_all_directly"]["label"] = "Run all directly"
# START_TIP
FW["Run_all_directly"]["tips"] = \
f"""This option appear when the user already did the image analysis for the current folder.  It is a
shortcut to skip the image analysis and to directly run and fine tune the video tracking.
"""
# END_TIP

FW["Next"] = {}
FW["Next"]["label"] = "Next"
# START_TIP
FW["Next"]["tips"] = \
f"""Click the *Next* button to go to the image analysis window (Fig. 2), or  to the window showing the
list of folders (Fig. 10) if applicable.
"""
# END_TIP

#######################################
######### ImageAnalysisWindow #########
#######################################

IAW = dict()
IAW["Image_number"] = {}
IAW["Image_number"]["label"] = "Image number"
# START_TIP
IAW["Image_number"]["tips"] = \
f"""Selects the image number to analyze. This number should only be changed when specimen(s) are
invisible on the first image (e.g. in the case of appearing colonies of bacteria), never otherwise.
When the specimen(s) are invisible, read more advanced images until some material can be detected.
NB:
- When the data is stored as images, this image number comes from the alphanumerical sorting of the
original image labeling.
"""
# END_TIP

IAW["several_blob_per_arena"] = {}
IAW["several_blob_per_arena"]["label"] = "One specimen per arena"
# START_TIP
IAW["several_blob_per_arena"]["tips"] = \
f"""Should be selected if there is only one specimen (e.g. a cell or a connected colony) per arena.   If
there already are (or will be) several specimen(s) per arena, unselect this option.
NB:
- This option is selected by default.
"""
# END_TIP

IAW["Scale_with"] = {}
IAW["Scale_with"]["label"] = "Scale with"
# START_TIP
IAW["Scale_with"]["tips"] = \
f"""Set how the true pixel size (in mm) should be computed. to calculate pixel size  Cellects can
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
"""
# END_TIP

IAW["Scale_size"] = {}
IAW["Scale_size"]["label"] = "Scale size"
# START_TIP
IAW["Scale_size"]["tips"] = \
f"""The *Scale size* is the length (in mm) of the item(s) used for scaling.
"""
# END_TIP

IAW["Select_and_draw"] = {}
IAW["Select_and_draw"]["label"] = "Select and draw"
# START_TIP
IAW["Select_and_draw"]["tips"] = \
f"""*Select and draw* is a tool allowing the user to inform Cellects that some parts of the image are
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
"""
# END_TIP

IAW["Draw_buttons"] = {}
IAW["Draw_buttons"]["label"] = "Draw buttons"
# START_TIP
IAW["Draw_buttons"]["tips"] = \
f"""Click the *Cell* or *Back* button and draw a corresponding area on the image by clicking and holding
down the mouse button.
"""
# END_TIP

IAW["Advanced_mode"] = {}
IAW["Advanced_mode"]["label"] = "Advanced mode"
# START_TIP
IAW["Advanced_mode"]["tips"] = \
f"""The *Advanced mode* allows the user to fine tune the image analysis parameters. This can be useful
to use previously working set of parameters on similar images, or to test the available methods
directly. Even when some analysis option are generated, selecting this option can be useful to
access:
- The color space combination corresponding to the displayed image.
- Various filter to apply on the image before segmentation.
- Other results by adding good channels together or mixing two good options using a logical operator
between them.
- The grid segmentation algorithm.
- The kmeans segmentation algorithm.
"""
# END_TIP

IAW["Generate_analysis_options"] = {}
IAW["Generate_analysis_options"]["label"] = "Generate analysis options"
# START_TIP
IAW["Generate_analysis_options"]["tips"] = \
f"""Cellects suggests an algorithms to automatically find the best parameters to detect specimens on the
first image:
- **Basic** → suggests options in a few minutes.   Alternatively, the user can select *Advanced
mode* to view or modify the parameters selected by Cellects.
NB:
- Clicking on *Basic* (or *Apply current config*) will provoke the display of a working message (in
orange).
"""
# END_TIP

IAW["Select_option_to_read"] = {}
IAW["Select_option_to_read"]["label"] = "Select option to read"
# START_TIP
IAW["Select_option_to_read"]["tips"] = \
f"""Select the option allowing the best segmentation. This dropdown menu appears after generating
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
"""
# END_TIP

IAW["Arena_shape"] = {}
IAW["Arena_shape"]["label"] = "Arena shape"
# START_TIP
IAW["Arena_shape"]["tips"] = \
f"""Specifies whether the specimen(s) can move in a circular or rectangular arena.
"""
# END_TIP

IAW["Spot_shape"] = {}
IAW["Spot_shape"]["label"] = "Set spot shape"
# START_TIP
IAW["Spot_shape"]["tips"] = \
f"""Initial shape of the specimen(s) inside arena(s).
"""
# END_TIP

IAW["Spot_size"] = {}
IAW["Spot_size"]["label"] = "Set spot size"
# START_TIP
IAW["Spot_size"]["tips"] = \
f"""Initial horizontal size of the specimen(s) (in mm). If similar across all specimens, this can also
be used as a scale.
"""
# END_TIP

IAW["Video_delimitation"] = {}
IAW["Video_delimitation"]["label"] = "Video delimitation"
# START_TIP
IAW["Video_delimitation"]["tips"] = \
f"""After validating the initial detection, the result of the automatic video delimitation appears in
blue in the  center of the window.   If correct, click *Yes*.   If incorrect, click *No*, and
Cellects will suggest:
- A slower, more efficient algorithm
- Or a manual delineation option
"""
# END_TIP

IAW["Last_image_question"] = {}
IAW["Last_image_question"]["label"] = "Last image question"
# START_TIP
IAW["Last_image_question"]["tips"] = \
f"""If the user thinks that parameters used on the first image might not work on later images, they can
fine
-tune them using the last image.
- Clicking *Yes* → allows testing on the last image before moving on.
- Clicking *No* → goes directly to video tracking.
"""
# END_TIP

IAW["Start_differs_from_arena"] = {}
IAW["Start_differs_from_arena"]["label"] = "Check if the medium at starting position differs from the rest of the arena"
# START_TIP
IAW["Start_differs_from_arena"]["tips"] = \
f"""If the substrate changes between starting and growing areas (e.g. nutritive gel vs transparent
agar), keep this checked. If the substrate is homogeneous everywhere, uncheck this option. This
option is only relevant for experiments in which the medium on which the specimen grow does not have
the same optic properties at the position of the specimen(s) at the first frame and at their
positions later on.
"""
# END_TIP

IAW["Save_image_analysis"] = {}
IAW["Save_image_analysis"]["label"] = "Save image analysis"
# START_TIP
IAW["Save_image_analysis"]["tips"] = \
f"""Complete the analysis of the current image. Clicking this button is useful to analyze only one
image.  To analyze video(s), click *Next*.
NB:
- When there should be only one specimen per arena, keeps the largest connected component.
- Compute and save (.csv) all descriptors selected in the Required output window on the current
image.
- Save a validation image to assess the efficiency of the segmentation.
"""
# END_TIP


#################################################
#########     Video analysis Window     #########
#################################################

VAW = {}
VAW["Arena_to_analyze"] = {}
VAW["Arena_to_analyze"]["label"] = "Arena to analyze"
# START_TIP
VAW["Arena_to_analyze"]["tips"] = \
f"""This arena number allows the user to load one particular arena in the current folder. Typically, the
user can choose an arena, click on *Detection* to load and analyse one arena and *Read* the
resulting analysis.
NB:
- Cellects automatically names the arena according to their position in the image, from left to
right and from top to bottom.
- If there is only one arena, this number should be one.
- *Post processing* automatically runs *Detection* and *Detection* automatically runs *Load One
arena*
- Loading will be faster if videos are already saved as ind_*.npy.
"""
# END_TIP

VAW["Maximal_growth_factor"] = {}
VAW["Maximal_growth_factor"]["label"] = "Maximal growth factor"
# START_TIP
VAW["Maximal_growth_factor"]["tips"] = \
f"""The maximal growth factor is a proportion of pixels in the image and indicates how far the specimen
can possibly move or grow from one image to the next. This factor should be:
- Increased if the analysis underestimates the specimen size.
- Decreased if the analysis overestimates the specimen size.
NB:
- Precisely, this is  the upper limit of the proportion of the image that is allowed to be covered
by the specimen  between two frames.
"""
# END_TIP

VAW["Segmentation_method"] = {}
VAW["Segmentation_method"]["label"] = "Segmentation method"
# START_TIP
VAW["Segmentation_method"]["tips"] = \
f"""Cellects includes five video tracking options:
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
"""
# END_TIP

VAW["Load_one_arena"] = {}
VAW["Load_one_arena"]["label"] = "Load one arena"
# START_TIP
VAW["Load_one_arena"]["tips"] = \
f"""Clicking this button loads the arena corresponding to the *Arena to analyze*. The center of the
window then displays the first frame of the video of that arena. Click *Read* to check the full
video.
"""
# END_TIP

VAW["Detection"] = {}
VAW["Detection"]["label"] = "Detection"
# START_TIP
VAW["Detection"]["tips"] = \
f"""*Detection* runs one (or all) segmentation method on one arena. Once finished, click *Read* to see
the detection result.  If correct, answer *Done* to *Step 1: Tune parameters to improve Detection*,
and check the effect of *Post
-processing*.
"""
# END_TIP

VAW["Read"] = {}
VAW["Read"]["label"] = "Read"
# START_TIP
VAW["Read"]["tips"] = \
f"""Clicking *Read* starts the video display corresponding to the current state of the analysis.
"""
# END_TIP

VAW["Fading_detection"] = {}
VAW["Fading_detection"]["label"] = "Fading detection"
# START_TIP
VAW["Fading_detection"]["tips"] = \
f"""*Fading detection* monitors how the specimen(s) leave some areas. This is useful when the specimens
not only grow but also move. Uncheck this option otherwise. When the specimen(s) may leave
previously covered areas, set a value between one and minus one to control the strength of that
detection.
- Near minus one: The algorithm will almost never detect when the specimen(s) leave an area.
- Near one: The algorithm may wrongly remove detection everywhere.
"""
# END_TIP

VAW["Post_processing"] = {}
VAW["Post_processing"]["label"] = "Post processing"
# START_TIP
VAW["Post_processing"]["tips"] = \
f"""*Postprocessing* applies the chosen detection algorithm to the video, on top of additional
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
"""
# END_TIP

VAW["Save_one_result"] = {}
VAW["Save_one_result"]["label"] = "Save one result"
# START_TIP
VAW["Save_one_result"]["tips"] = \
f"""Complete the analysis of the current video. Clicking this button is useful to analyze only one
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
"""
# END_TIP

VAW["Run_All"] = {}
VAW["Run_All"]["label"] = "Run All"
# START_TIP
VAW["Run_All"]["tips"] = \
f"""If detection with *Postprocessing* leads to a satisfying result for one arena,   the user can apply
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
"""
# END_TIP

VAW["Save_all_choices"] = {}
VAW["Save_all_choices"]["label"] = "Save all choices"
# START_TIP
VAW["Save_all_choices"]["tips"] = \
f"""Clicking the *Save all choices* write or updates config files to redo an analysis with the same
parameters later on.
"""
# END_TIP

#################################################
########     Multiple folders Window     ########
#################################################

MF = {}
MF["Check_to_select_all_folders"] = {}
MF["Check_to_select_all_folders"]["label"] = "Check to select all folders"
# START_TIP
MF["Check_to_select_all_folders"]["tips"] = \
f"""Select this option to run the analysis on all folders containing images matching the *Image prefix*
and *Images extension* patterns. Otherwise, use Ctrl/Cmd to select any folder to analyze.
NB:
- This selection only has consequences when one use the *Run All* functionality.
- To keep and use some masks (e.g. representing the background or the starting pixels of the
specimen(s)) on every selected folder, use the *Keep Cell and Back drawing for all folders* option
in *Advanced parameters*.
"""
# END_TIP

#################################################
########     Required Output Window     ########
#################################################

RO = {}
RO["coord_specimen"] = {}
RO["coord_specimen"]["label"] = "Pixels covered by the specimen(s)"
# START_TIP
RO["coord_specimen"]["tips"] = \
f"""Save a .npy file containing the coordinates (t, y, x) of presence of the specimen(s), as detected
with the current parameters.
NB:
- Depending on the number of frames, these files may take a lot a memory.
"""
# END_TIP

RO["Graph"] = {}
RO["Graph"]["label"] = "Graph of the specimen(s) (or network)"
# START_TIP
RO["Graph"]["tips"] = \
f"""Compute the geometrical graph describing the specimen as it is detected with the current parameters.
Cellects compute the graph of each frame on the skeleton of the largest connected component.  If the
user also ask Cellects to detect a network inside the specimen(s), the graph will be computed on
this detected network. The graph is saved as two .csv files:
- One for the vertices containing their coordinates (t, y, x), their id, whether they are tips,
whether they are part of the specimen area at the beginning of the video, and whether they are part
of a small cluster of vertices.
- One for the edges containing their id, the id of their two vertices, their lengths, their average
width and intensity.
NB:
- Depending on the number of frames, these files may take a lot a memory.
- Selecting both network and graph detection is only useful for organisms having a distinguishable
network within their main body (e.g. Physarum polycephalum)
"""
# END_TIP

RO["coord_oscillating"] = {}
RO["coord_oscillating"]["label"] = "Oscillating areas in the specimen(s)"
# START_TIP
RO["coord_oscillating"]["tips"] = \
f"""Compute and save the coordinates (t, y, x) of areas oscillating synchronously in the specimen(s).
This algorithm save two .npy file: one for thickening areas and one for slimming areas.
"""
# END_TIP

RO["coord_network"] = {}
RO["coord_network"]["label"] = "Network in the specimen(s)"
# START_TIP
RO["coord_network"]["tips"] = \
f"""Detect and save (as .npy file) the coordinate (t, y, x) of a distinguishable network in the
specimen(s).
"""
# END_TIP


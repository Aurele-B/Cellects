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
a set of files where each file contains a single image) or as a video. Images must be named
alphanumerically so the program can read them in the right order.
"""
# END_TIP

FW["Image_prefix_and_extension"] = {}
FW["Image_prefix_and_extension"]["label"] = "Image prefix and extension"
# START_TIP
FW["Image_prefix_and_extension"]["tips"] = \
f"""The *Images prefix* and *Images extension* fields allow Cellects to consider relevant data. For
example, setting 'exp_' as image prefix and '.jpg' as image extension will cause Cellects to only
consider JPG files whose name starts with 'exp_'. Remaining labels should indicate the order in
which images were taken.
NB:
- Image prefix is optional
- If every .jpg files start with IMG_ but other .jpg files exist, use the prefix to exclude
irrelevant files.
- Supported formats: bmp, dib, exr, exr, hdr, jp2, jpe, jpeg, jpg, pbm, pfm, pgm, pic, png,  pnm,
ppm, ras, sr, tif, tiff, webp, cr2, cr3, nef, arw, sr2, raf, prf, rw2, pef, dng, 3fr, iiq.
"""
# END_TIP

FW["Folder"] = {}
FW["Folder"]["label"] = "Folder"
# START_TIP
FW["Folder"]["tips"] = \
f"""The *Folder* field must specify the directory path to the folder(s) for Cellects to be able to run
the analysis. The user can copy/paste this path into the field or navigate to the folder using the
*Browse* push button. For batch analysis, provide a path leading directly to the parent folder
containing all subfolders.
"""
# END_TIP

FW["Arena_number_per_folder"] = {}
FW["Arena_number_per_folder"]["label"] = "Arena number per folder"
# START_TIP
FW["Arena_number_per_folder"]["tips"] = \
f"""The *Arena number per folder* specifies how many arenas are present in the images. Cellects will
process and analyze each arena separately.
NB:
- For batch processing, assign different arena counts for each subfolder (see Fig. 11: the several
folder window).
"""
# END_TIP

FW["Browse"] = {}
FW["Browse"]["label"] = "Browse"
# START_TIP
FW["Browse"]["tips"] = \
f"""Clicking the *Browse* button opens a dialog to select a folder for analysis.
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
f"""This option appears when image analysis has already been performed for the current folder. It is a
shortcut to bypass the image analysis step and proceed directly to video tracking refinement.
"""
# END_TIP

FW["Next"] = {}
FW["Next"]["label"] = "Next"
# START_TIP
FW["Next"]["tips"] = \
f"""Click the *Next* button to go to the image analysis window (Fig. 2), or  to the window showing the
list of folders (Fig. 11) if applicable.
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
invisible on the first image (e.g., in the case of appearing colonies of bacteria), never otherwise.
When the specimen(s) are invisible, read more advanced images until some material can be detected.
NB:
- When the data is stored as images, this image number comes from alphanumerical sorting of original
image labels.
"""
# END_TIP

IAW["several_blob_per_arena"] = {}
IAW["several_blob_per_arena"]["label"] = "One specimen per arena"
# START_TIP
IAW["several_blob_per_arena"]["tips"] = \
f"""Select this option if there is only one specimen (e.g., a cell or connected colony) per arena. If
multiple specimens exist (or will be present) in an arena, unselect this option.
NB:
- This option is selected by default.
"""
# END_TIP

IAW["Scale_with"] = {}
IAW["Scale_with"]["label"] = "Scale with"
# START_TIP
IAW["Scale_with"]["tips"] = \
f"""Specify how to compute true pixel size (in mm). Cellects can determine this scale using:
- Image width (horizontal dimension)
- Specimen width on first image (usable when specimens share consistent width)
NB:
- Advanced parameters allow disabling scaling and outputting in pixels.
- Using specimen width reduces initial detection efficiency. We recommend using image width unless
specimen dimensions are known with higher accuracy than imaging equipment.
- Pixel size is stored in a file named `software_settings.csv`.
"""
# END_TIP

IAW["Scale_size"] = {}
IAW["Scale_size"]["label"] = "Scale size"
# START_TIP
IAW["Scale_size"]["tips"] = \
f"""The *Scale size* is the actual length (in mm) corresponding to scaling reference.
NB:
- This value enables conversion from pixel coordinates to physical dimensions.
"""
# END_TIP

IAW["Select_and_draw"] = {}
IAW["Select_and_draw"]["label"] = "Select and draw"
# START_TIP
IAW["Select_and_draw"]["tips"] = \
f"""*Select and draw* allows the user to inform Cellects about specimen (*Cell*) and background (*Back*)
areas in images. To use, click *Cell* or *Back* button (button color changes), then:
- Click and drag mouse on image to mark corresponding area
- Numbered drawings appear below buttons for reference
- (if needed) Click numbered drawing to remove selection.
NB:
- By default, this tool only works for the first folder when analyzing multiple folders. Advanced
parameters include an option to use these same masks in multiple folders.
- To apply saved masks (e.g., background or specimen initiation regions) across selected folders,
enable *Keep Cell and Back drawing for all folders* in *Advanced parameters*.
"""
# END_TIP

IAW["Draw_buttons"] = {}
IAW["Draw_buttons"]["label"] = "Draw buttons"
# START_TIP
IAW["Draw_buttons"]["tips"] = \
f"""Click the *Cell* or *Back* button and draw a corresponding area on the image by clicking and holding
mouse on the image.
"""
# END_TIP

IAW["Advanced_mode"] = {}
IAW["Advanced_mode"]["label"] = "Advanced mode"
# START_TIP
IAW["Advanced_mode"]["tips"] = \
f"""The *Advanced mode* enables fine tuning of image analysis parameters for:
- Custom color space combinations (e.g., HSV, HLS)
- Applying filters before segmentation
- Combining segmentations using logical operators
- Accessing rolling window and Kmeans algorithms
NB:
- Useful for reusing validated parameter sets or testing alternative methods.
"""
# END_TIP

IAW["Color_combination"] = {}
IAW["Color_combination"]["label"] = "Color combination"
# START_TIP
IAW["Color_combination"]["tips"] = \
f"""Color spaces are transformations of the original BGR (Blue Green Red) image Instead of defining an
image by 3 colors,  they transform it into 3 different visual properties
- hsv: hue (color), saturation, value (lightness)
- hls: hue (color), lightness, saturation
- lab: Lightness, Red/Green, Blue/Yellow
- luv and yuv: l and y are Lightness, u and v are related to colors
"""
# END_TIP

IAW["Logical_operator"] = {}
IAW["Logical_operator"]["label"] = "Logical operator"
# START_TIP
IAW["Logical_operator"]["tips"] = \
f"""The *Logical operator* defines how to combine results from distinct segmentations (e.g., merging the
segmentation resulting from a specific color space transformation and filtering with a different
one). Supported operators: AND, OR, XOR.
"""
# END_TIP

IAW["Filter"] = {}
IAW["Filter"]["label"] = "Filter"
# START_TIP
IAW["Filter"]["tips"] = \
f"""Apply a filter to preprocess images before segmentation.
NB:
- Filtering can improve segmentation accuracy by emphasizing relevant image features.
"""
# END_TIP

IAW["Rolling_window_segmentation"] = {}
IAW["Rolling_window_segmentation"]["label"] = "Rolling window segmentation"
# START_TIP
IAW["Rolling_window_segmentation"]["tips"] = \
f"""Segments image regions using a rolling window approach to detect local intensity valleys. The method
applies Otsu's thresholding locally on each window.
"""
# END_TIP

IAW["Kmeans"] = {}
IAW["Kmeans"]["label"] = "Kmeans"
# START_TIP
IAW["Kmeans"]["tips"] = \
f"""The *Kmeans* algorithm clusters pixels into a specified number of categories (2
-5) to identify specimen regions within the image.
"""
# END_TIP

IAW["Generate_analysis_options"] = {}
IAW["Generate_analysis_options"]["label"] = "Generate analysis options"
# START_TIP
IAW["Generate_analysis_options"]["tips"] = \
f"""Cellects proposes algorithms to automatically determine optimal specimen detection parameters on the
first or last image:
- **Basic** → provides suggestions in minutes. Alternatively, the user can switch to *Advanced mode*
to review or modify more specific settings.
NB:
- Selecting *Basic* (or *Apply current config*) will trigger an orange working message during
processing.
"""
# END_TIP

IAW["Select_option_to_read"] = {}
IAW["Select_option_to_read"]["label"] = "Select option to read"
# START_TIP
IAW["Select_option_to_read"]["tips"] = \
f"""Choose the option producing optimal segmentation results. This menu appears after generating
analysis options, allowing direct quality assessment. For example, if Option 1 shows correct
detection (e.g., 6 spots in 6 arenas), click *Yes*. Otherwise, improve analysis via:
- Adjusting arena/spot shapes or sizes
- Using *Select and draw* to annotate specimens/background
- Manual configuration in advanced mode → Test changes with *Apply current config*
NB:
- Confirm when magenta/pink contours match expected positions and counts.
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
f"""Defines the expected shape of specimens within arenas.
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
f"""After confirming initial detection, automatic video delimitation results appear in blue.  Click
*Yes* if accurate or *No* for:
- A slower, higher precision algorithm.
- Manual delineation option.
"""
# END_TIP

IAW["Last_image_question"] = {}
IAW["Last_image_question"]["label"] = "Last image question"
# START_TIP
IAW["Last_image_question"]["tips"] = \
f"""If parameters might fail on later images, test them first on the final frame:
- *Yes* → validates with last image before tracking.
- *No* → proceeds directly to video analysis.
"""
# END_TIP

IAW["Start_differs_from_arena"] = {}
IAW["Start_differs_from_arena"]["label"] = "Check if the medium at starting position differs from the rest of the arena"
# START_TIP
IAW["Start_differs_from_arena"]["tips"] = \
f"""Enable if the substrate differs between initial position and arena growth area (e.g., nutritive gel
vs. agar). Disable for homogeneous substrates.
"""
# END_TIP

IAW["Save_image_analysis"] = {}
IAW["Save_image_analysis"]["label"] = "Save image analysis"
# START_TIP
IAW["Save_image_analysis"]["tips"] = \
f"""Complete the analysis of the current image. Clicking this button analyzes only one image. To analyze
video(s), click *Next*.
NB:
- When analyzing a single specimen per arena, keeps the largest connected component.
- Saves all selected descriptors in .csv format for the current image and generates a validation
image to assess segmentation accuracy.
"""
# END_TIP


#################################################
#########     Video analysis Window     #########
#################################################

VAW = dict()
VAW["Arena_to_analyze"] = {}
VAW["Arena_to_analyze"]["label"] = "Arena to analyze"
# START_TIP
VAW["Arena_to_analyze"]["tips"] = \
f"""This arena number selects a specific arena in the current folder. The user can choose an arena,
click *Detection* to load and analyze it, then *Read* results.
NB:
- Cellects automatically names the arena by their position (left to right, top to bottom).
- For single arena setups, use 1.
- *Post processing* triggers *Detection*, which in turn triggers *Load One arena*.
- Videos can be saved (as .npy files) for later analysis using the Advanced parameter *Keep
unaltered videos*.
"""
# END_TIP

VAW["Maximal_growth_factor"] = {}
VAW["Maximal_growth_factor"]["label"] = "Maximal growth factor"
# START_TIP
VAW["Maximal_growth_factor"]["tips"] = \
f"""This is the maximum allowable proportion of image area that may be covered by specimen movement
between frames. Adjust accordingly:
- Increase if specimen size is underestimated.
- Decrease if specimen size is overestimated.
NB:
- Precisely, this defines an upper bound on relative coverage changes between sequential images.
"""
# END_TIP

VAW["Temporal_smoothing"] = {}
VAW["Temporal_smoothing"]["label"] = "Temporal smoothing"
# START_TIP
VAW["Temporal_smoothing"]["tips"] = \
f"""Applies temporal smoothing to reduce noise and highlight long
-term trends by averaging pixel intensity changes. Use when analyzing slope
-based segmentation results.
NB:
- This uses a moving window algorithm on pixel intensity curves over time.
- Excessive iterations produce constant values, preventing accurate detection.
"""
# END_TIP

VAW["Segmentation_method"] = {}
VAW["Segmentation_method"]["label"] = "Segmentation method"
# START_TIP
VAW["Segmentation_method"]["tips"] = \
f"""Cellects includes five video tracking options:
- **Frame option**: Applies the image analysis algorithm frame by frame, without temporal dynamics.
- **Threshold option**: Compares pixel intensity with the average intensity of the whole image at
each time step.
- **Slope option**: Compares pixel intensity slopes with an automatically defined threshold.
- **T and S option**: logical AND of threshold and slope options.
- **T or S option**: logical OR of threshold and slope options.
NB:
- Selecting the *Compute all options* before dunning *Detection* allows method comparison.  Once
analysis completes. Once the analysis completed, select one option and click *Read*.
- Computing only one option is faster and requires less memory.
- When *Heterogeneous background* or *Grid segmentation* has been selected in the image analysis
window, only the *Frame* option remains available.
"""
# END_TIP

VAW["Load_one_arena"] = {}
VAW["Load_one_arena"]["label"] = "Load one arena"
# START_TIP
VAW["Load_one_arena"]["tips"] = \
f"""Clicking this button loads the arena associated with *Arena to analyze*. The center of the window
displays the first frame of that arena's video. Click *Read* to review the full video.
"""
# END_TIP

VAW["Detection"] = {}
VAW["Detection"]["label"] = "Detection"
# START_TIP
VAW["Detection"]["tips"] = \
f"""*Detection* applies a (or all) segmentation methods to one arena. Once finished, click *Read*  to
view the detection result. If correct, answer *Done* to proceed with tuning parameters for post
processing.
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
f"""*Fading detection* monitors when specimens leave previously occupied areas, useful for  moving
organisms rather than static growth. Uncheck this option if not needed. Set a value  between minus
one and one to control sensitivity:
- Near minus one: Minimal false removal of specimen traces.
- Near one: High risk of over
-removal from all areas.
"""
# END_TIP

VAW["Post_processing"] = {}
VAW["Post_processing"]["label"] = "Post processing"
# START_TIP
VAW["Post_processing"]["tips"] = \
f"""*Post processing* applies detection algorithms with additional enhancements:
- Binary operations: opening, closing, logical ops.
- Fading detection* tracking: when specimen(s) may leave areas (optional).
- *Correct errors around initial shape*: when the contour of the initial position of the specimen is
hard to detect (optional).
- *Connect distant shapes*: when the specimen's heterogeneity create wrong disconnections in the
video detection (optional).
- *Prevent fast growth near periphery*: when arena's border (typically petri dishes) may be wrongly
detected as specimen (optional).
NB:
- Once Post processing works, the user can click “*Done*” to *Step 2: Tune fading and advanced
parameters to improve Post processing*, and then *Run All* arenas.
"""
# END_TIP

VAW["Save_one_result"] = {}
VAW["Save_one_result"]["label"] = "Save one result"
# START_TIP
VAW["Save_one_result"]["tips"] = \
f"""Complete the current video analysis by clicking this button for single arena processing. Saving
includes:
- Calculating all selected descriptors (.csv) per frame.
- Generating validation videos for detection verification.
- Storing configuration parameters for reproducibility.
NB:
- This action will overwrite results and validation data for the current arena.
"""
# END_TIP

VAW["Run_All"] = {}
VAW["Run_All"]["label"] = "Run All"
# START_TIP
VAW["Run_All"]["tips"] = \
f"""Apply validated parameters to all arenas by clicking *Run All*. This action:
- Generates full
-resolution video outputs (storage
-intensive)
- Processes videos sequentially with real time visualization
- Calculates selected descriptors for each frame
- Produces validation content at multiple intervals
- Preserves current configuration settings
"""
# END_TIP

VAW["Save_all_choices"] = {}
VAW["Save_all_choices"]["label"] = "Save all choices"
# START_TIP
VAW["Save_all_choices"]["tips"] = \
f"""Clicking *Save all choices* writes/updates configuration files to preserve analysis parameters for
future replication.
"""
# END_TIP

#################################################
########     Multiple folders Window     ########
#################################################

MF = dict()
MF["Check_to_select_all_folders"] = {}
MF["Check_to_select_all_folders"]["label"] = "Check to select all folders"
# START_TIP
MF["Check_to_select_all_folders"]["tips"] = \
f"""Select this option to run the analysis on all folders containing images matching the *Image prefix*
and *Images extension*. Otherwise, use Ctrl/Cmd to select specific folders for analysis.
NB:
- This setting affects only the *Run All* functionality.
- To apply saved masks (e.g., background or specimen initiation regions) across selected folders,
enable    *Keep Cell and Back drawing for all folders* in *Advanced parameters*.
"""
# END_TIP

#################################################
########     Required Output Window     ########
#################################################

RO = dict()
RO["coord_specimen"] = {}
RO["coord_specimen"]["label"] = "Pixels covered by the specimen(s)"
# START_TIP
RO["coord_specimen"]["tips"] = \
f"""Save a .npy file containing coordinates (t, y, x) of specimen pixel presence as detected by current
parameters.
NB:
- These files may consume significant memory depending on the total frame count.
"""
# END_TIP

RO["Graph"] = {}
RO["Graph"]["label"] = "Graph of the specimen(s) (or network)"
# START_TIP
RO["Graph"]["tips"] = \
f"""Compute a geometrical graph describing the specimen based on current detection parameters.  Cellects
generates this graph using the skeleton of the largest connected component per frame.  If network
detection is enabled, it will be computed on the detected network instead. The output includes:
- A .csv file for vertices with coordinates (t, y, x), IDs, tip status, part of the specimen's
initial position, connection status with other vertices.
- A .csv file for edges with IDs, vertex pairs, lengths, average width, and intensity.
NB:
- These files may consume significant memory depending on the total frame count.
- Network and graph detection together are relevant only for organisms with a distinct internal
network (e.g., *Physarum polycephalum*).
"""
# END_TIP

RO["coord_oscillating"] = {}
RO["coord_oscillating"]["label"] = "Oscillating areas in the specimen(s)"
# START_TIP
RO["coord_oscillating"]["tips"] = \
f"""Compute and save (as .npy files) coordinates (t, y, x) of oscillating areas in the specimen(s).  Two
files are generated: one for thickening regions and one for slimming regions.
"""
# END_TIP

RO["coord_network"] = {}
RO["coord_network"]["label"] = "Network in the specimen(s)"
# START_TIP
RO["coord_network"]["tips"] = \
f"""Detect and save (as .npy file) coordinates (t, y, x) of a distinct network within the specimen(s).
specimen(s).
"""
# END_TIP

####################################################
########     Advanced Parameters Window     ########
####################################################

AP = dict()
AP["Crop_images"] = {}
AP["Crop_images"]["label"] = "Automatically crop images"
# START_TIP
AP["Crop_images"]["tips"] = \
f"""Uses initial image detection to crop all images and improve arena/last image detection.
NB:
- Unselect this option if analysis fails or crashes during image analysis.
"""
# END_TIP

AP["Subtract_background"] = {}
AP["Subtract_background"]["label"] = "Subtract background"
# START_TIP
AP["Subtract_background"]["tips"] = \
f"""Takes the first image and subtracts it from subsequent images. This can improve or degrade detection
depending on dataset characteristics.
"""
# END_TIP

AP["Keep_drawings"] = {}
AP["Keep_drawings"]["label"] = "Keep Cell and Back drawings for all folders"
# START_TIP
AP["Keep_drawings"]["tips"] = \
f"""During initial image analysis, if the user drew cell/back regions to assist detection, this option
saves and uses these annotations across all folders. In summary:
- **Checked** → retain annotations for all folders
- **Unchecked** → apply only to current folder
"""
# END_TIP

AP["Correct_errors_around_initial"] = {}
AP["Correct_errors_around_initial"]["label"] = "Correct errors around initial specimen's position"
# START_TIP
AP["Correct_errors_around_initial"]["tips"] = \
f"""Applies an algorithm to correct detection errors near the initial specimen position due to color
variations (e.g., from nutrient patches). Technical workflow:
- Identifies potential gaps around initial position
- Monitors local growth velocity
- Fills gaps using growth patterns from adjacent pixels
NB:
- ⚠️ Not recommended if the substrate has the same transparency everywhere (i.e. no difference
between starting and growth regions).
"""
# END_TIP

AP["Prevent_fast_growth_near_periphery"] = {}
AP["Prevent_fast_growth_near_periphery"]["label"] = "Prevent fast growth near periphery"
# START_TIP
AP["Prevent_fast_growth_near_periphery"]["tips"] = \
f"""During video analysis, prevents false specimen detection at arena borders by filtering rapid
periphery growth.
- **Checked** → Exclude fast
-moving detections near boundaries
- **Unchecked** → Use standard detection criteria
"""
# END_TIP

AP["Connect_distant_shapes"] = {}
AP["Connect_distant_shapes"]["label"] = "Connect distant shapes"
# START_TIP
AP["Connect_distant_shapes"]["tips"] = \
f"""Algorithm for connecting disjoint specimen regions in cases where there should be only one connected
specimen per arena.  This is useful when the specimen's heterogeneity create wrong disconnections
and the detection is smaller than the true specimen. Technical implementation:
- Identifies disconnected subregions
- Analyzes local growth dynamics
- Recreates connections using spatially consistent growth patterns
NB:
- Increases analysis time substantially.
"""
# END_TIP

AP["Specimens_have_same_direction"] = {}
AP["Specimens_have_same_direction"]["label"] = "All specimens have the same direction"
# START_TIP
AP["Specimens_have_same_direction"]["tips"] = \
f"""Select to optimize arena detection for specimens moving move in the same direction.
- **Checked** → Uses motion pattern analysis for arena localization.
- **Unchecked** → Employs standard centroid based algorithm.
NB:
- Both options work equally when growth is roughly isotropic.
"""
# END_TIP

AP["Appearance_size_threshold"] = {}
AP["Appearance_size_threshold"]["label"] = "Appearance size threshold (automatic if checked)"
# START_TIP
AP["Appearance_size_threshold"]["tips"] = \
f"""Minimum pixel count threshold for identifying specimen emergence (e.g., bacterial colony formation).
- **Checked** → Automatic threshold calculation.
- **Unchecked** → Manual user
-defined threshold.
"""
# END_TIP

AP["Appearance_detection_method"] = {}
AP["Appearance_detection_method"]["label"] = "Appearance detection method"
# START_TIP
AP["Appearance_detection_method"]["tips"] = \
f"""Selection criteria for initial specimen detection:
- Largest: Based on component size metric.
- Most central: Based on arena center proximity.
NB:
- Applicable only to progressively emerging specimens.
"""
# END_TIP

AP["Mesh_side_length"] = {}
AP["Mesh_side_length"]["label"] = "Mesh side length"
# START_TIP
AP["Mesh_side_length"]["tips"] = \
f"""Pixel dimension for analysis window size.
NB:
- Must not exceed minimum image dimension
"""
# END_TIP

AP["Mesh_step_length"] = {}
AP["Mesh_step_length"]["label"] = "Mesh step"
# START_TIP
AP["Mesh_step_length"]["tips"] = \
f"""The size of the step (in pixels) between consecutive rolling window positions.
NB:
- Must not exceed the mesh side length to ensure full coverage of the image.
"""
# END_TIP

AP["Mesh_minimal_intensity_variation"] = {}
AP["Mesh_minimal_intensity_variation"]["label"] = "Mesh minimal intensity variation"
# START_TIP
AP["Mesh_minimal_intensity_variation"]["tips"] = \
f"""The minimal variation in intensity to consider that a given window does contain the specimen(s).
NB:
- This threshold is an intensity value ranging from 0 to 255 (generally small).
- Correspond to the level of noise in the background.
"""
# END_TIP

AP["Expected_oscillation_period"] = {}
AP["Expected_oscillation_period"]["label"] = "Expected oscillation period"
# START_TIP
AP["Expected_oscillation_period"]["tips"] = \
f"""The period (in minutes) of biological oscillations to detect within the specimen(s). Computation is
based on luminosity variations.
"""
# END_TIP

AP["Minimal_oscillating_cluster_size"] = {}
AP["Minimal_oscillating_cluster_size"]["label"] = "Minimal oscillating cluster size"
# START_TIP
AP["Minimal_oscillating_cluster_size"]["tips"] = \
f"""When looking for oscillatory patterns, Cellects detects connected components that are thickening or
slimming synchronously in the specimen(s). This parameter thresholds the minimal size of these
groups of connected pixels. This threshold is useful to filter out small noisy oscillations.
"""
# END_TIP

AP["Spatio_temporal_scaling"] = {}
AP["Spatio_temporal_scaling"]["label"] = "Spatio-temporal scaling"
# START_TIP
AP["Spatio_temporal_scaling"]["tips"] = \
f"""Defines the spatiotemporal scale of the dataset:
- Time between images or frames (minutes).
- An option to convert areas/distances from pixels to mm/mm².
"""
# END_TIP

AP["Parallel_analysis"] = {}
AP["Parallel_analysis"]["label"] = "Run analysis in parallel"
# START_TIP
AP["Parallel_analysis"]["tips"] = \
f"""Allow the use of more than one core of the computer processor.
- **Checked** → Uses multiple CPU cores to analyze arenas in parallel (faster).
- **Unchecked** → Single core analysis.
"""
# END_TIP

AP["Proc_max_core_nb"] = {}
AP["Proc_max_core_nb"]["label"] = "Proc max core number"
# START_TIP
AP["Proc_max_core_nb"]["tips"] = \
f"""Maximum number of logical CPU cores to use during analysis. The default value is set to the total
number of available CPU cores minus one.
"""
# END_TIP

AP["Minimal_RAM_let_free"] = {}
AP["Minimal_RAM_let_free"]["label"] = "Minimal RAM let free"
# START_TIP
AP["Minimal_RAM_let_free"]["tips"] = \
f"""Amount of RAM that should be left available for other programs. Setting to `0` gives Cellects all
memory, but increases crash risk if other apps are open.
"""
# END_TIP

AP["Lose_accuracy_to_save_RAM"] = {}
AP["Lose_accuracy_to_save_RAM"]["label"] = "Lose accuracy to save RAM"
# START_TIP
AP["Lose_accuracy_to_save_RAM"]["tips"] = \
f"""For low memory systems:
- Converts video from `np.float64` to `uint8`
- Saves RAM at the cost of a slight precision loss
"""
# END_TIP

AP["Video_fps"] = {}
AP["Video_fps"]["label"] = "Video fps"
# START_TIP
AP["Video_fps"]["tips"] = \
f"""Frames per second of validation videos.
"""
# END_TIP

AP["Keep_unaltered_videos"] = {}
AP["Keep_unaltered_videos"]["label"] = 'Keep unaltered videos'
# START_TIP
AP["Keep_unaltered_videos"]["tips"] = \
f"""Keeps unaltered `.npy` videos in hard drive.
- **Checked** → Rerunning the same analysis will be faster.
- **Unchecked** → These videos will be written and removed each run of the same analysis.
NB:
- Large files: it is recommended to remove them once analysis is entirely finalized.
"""
# END_TIP

AP["Save_processed_videos"] = {}
AP["Save_processed_videos"]["label"] = 'Save processed videos'
# START_TIP
AP["Save_processed_videos"]["tips"] = \
f"""Saves lightweight processed validation videos (recommended over unaltered videos). These videos
assess analysis accuracy and can be read in standard video players.
"""
# END_TIP

AP["Csc_for_video_analysis"] = {}
AP["Csc_for_video_analysis"]["label"] = 'Color space combination for video analysis'
# START_TIP
AP["Csc_for_video_analysis"]["tips"] = \
f"""Advanced option: Changes the way RGB processing directly in video tracking. Useful for testing new
color spaces without (re)running image analysis.
"""
# END_TIP

AP["Night_mode"] = {}
AP["Night_mode"]["label"] = 'Night mode'
# START_TIP
AP["Night_mode"]["tips"] = \
f"""Switches the application background between light and dark themes.
"""
# END_TIP

AP["Reset_all_settings"] = {}
AP["Reset_all_settings"]["label"] = 'Reset all settings'
# START_TIP
AP["Reset_all_settings"]["tips"] = \
f"""Useful when the software freezes with no apparent reason. To reset all settings, it removes the
config file in the  current folder as well as the config file in the software folder. Then, it
retrieves and saves the default parameters.
"""
# END_TIP

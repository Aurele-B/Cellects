# Advanced parameters

<figure>
  <img src="doc/static/UserManualFigure8.png" alt="Advanced parameters window" width="600">
  <figcaption><strong>Figure 8:</strong> Advanced parameters window</figcaption>
</figure>

---

<!-- START_Crop_images -->
## Automatically crop images:
Uses the first image detection to crop all images and improve arena and last image detection.
NB:
- If the analysis fails or the program crashes while running the image analysis window, unselecting
this option may help.

<!-- END_Crop_images -->

---

<!-- START_Subtract_background -->
## Subtract background:
Takes the first image and subtracts it from every following images. This an either improve or
degrade detection depending on the dataset.

<!-- END_Subtract_background -->

---

<!-- START_Keep_drawings -->
## Keep Cell and Back drawings for all folders:
During the first image analysis, if the user drew cell and back to help detection, this option save
and use this information for all folders. In summary:
- **Checked** → keep this information for all folders
- **Unchecked** → only use it for the current folder

<!-- END_Keep_drawings -->

---

<!-- START_Correct_errors_around_initial -->
## Correct errors around initial specimen's position:
Apply an algorithm allowing to correct missing detection around the initial position of the
specimen.  This option is useful when there are important color variations around that position.
This occurs, for instance,  when the width or diffusion of a nutritive patch blurs a normally
transparent medium.  Select this algorithm only if you realized that detection is less efficient
there. Technically, this algorithm works as follows:
- It detects potential gaps around the initial position of the specimen.
- It monitors the growth speed nearby.
- It fills these gaps in the same way growth occurs in nearby pixels.
NB:
- ⚠️ Do not use if the substrate has the same opacity everywhere (i.e. no difference between
starting and growth regions).

<!-- END_Correct_errors_around_initial -->

---

<!-- START_Prevent_fast_growth_near_periphery -->
## Prevent fast growth near periphery:
During video analysis, the borders of the arena may be wrongly detected as part of the specimen(s),
this option helps to avoid this issue.
- **Checked** → Remove the detection of the specimen(s) that move too fast near periphery.
- **Unchecked** → Do not change the detection.

<!-- END_Prevent_fast_growth_near_periphery -->

---

<!-- START_Connect_distant_shapes -->
## Connect distant shapes:
This error correcting algorithm makes dynamic connections between parts of the specimen.  It can
only be used when there can only be one connected specimen per arena. This is useful when the
specimen's heterogeneity create wrong disconnections and the detection is smaller than the true
specimen. Technically, this algorithm works as follows:
- It detects areas that are disconnected from the main detected area.
- It monitors the growth speed close to this disconnected area.
- It connects this area with the main area in the same way growth occurs in nearby pixels.
NB:
- This option can drastically increase the duration of the analysis.
- It is useful when the specimen color is close to the background and causes disconnections.

<!-- END_Connect_distant_shapes -->

---

<!-- START_Specimens_have_same_direction -->
## All specimens have the same direction:
Selecting this algorithm improves automatic arena detection when all specimens move in the same
direction.
- **Checked** → Improve the chances to correctly detect arenas when specimen(s) move strongly and in
the same direction.
- **Unchecked** → Use the fastest automatic arena detection algorithm, based on the distances
between the centroids of the specimen(s) at the beginning of the video.

<!-- END_Specimens_have_same_direction -->

---

<!-- START_Appearance_size_threshold -->
## Appearance size threshold (automatic if checked):
Defines the minimal size threshold (in pixels) for considering an appearing shape as a specimen
(e.g. bacterial colony).
- **Checked** → Automatically determine the threshold.
- **Unchecked** → Allow the user to set the threshold.

<!-- END_Appearance_size_threshold -->

---

<!-- START_Appearance_detection_method -->
## Appearance detection method:
Two methods available:
- **Largest** → According to the size of the detected components.
- **Most central** → According to the distance to the center of the arena.
NB:
- Only useful when specimen(s) are invisible at the beginning of the experiment and appear
progressively.

<!-- END_Appearance_detection_method -->

---

<!-- START_Mesh_side_length -->
## Mesh side length:
The length of one side (in pixels) of the rolling window.
NB:
- Cannot be larger than the smaller side of the image.

<!-- END_Mesh_side_length -->

---

<!-- START_Mesh_step_length -->
## Mesh step:
The size of the step (in pixels) between two positioning of the rolling window.
NB:
- Should not be larger than the mesh side length to cover all pixels.

<!-- END_Mesh_step_length -->

---

<!-- START_Mesh_minimal_intensity_variation -->
## Mesh minimal intensity variation:
The minimal variation in intensity to consider that a given window do contain the specimen(s).
NB:
- This threshold is an intensity value ranging from 0 to 255 (generally small).
- Correspond to the level of noise in the background.

<!-- END_Mesh_minimal_intensity_variation -->

---

<!-- START_Expected_oscillation_period -->
## Expected oscillation period:
The period (in minutes) of the biological oscillations to detect within the specimen(s). Computation
is based on luminosity variations.

<!-- END_Expected_oscillation_period -->

---

<!-- START_Minimal_oscillating_cluster_size -->
## Minimal oscillating cluster size:
When looking for oscillatory patterns, Cellects detects connected components that are thickening or
slimming synchronously in the specimen(s). This parameter thresholds the minimal size of these
connected group of pixels. This threshold is useful to filter out small noisy oscillations.

<!-- END_Minimal_oscillating_cluster_size -->

---

<!-- START_Spatio_temporal_scaling -->
## Spatio-temporal scaling:
Defines the spatiotemporal scale of the dataset:
- Time between images or frames (minutes)
- Option to convert areas/distances from pixels to mm/mm²

<!-- END_Spatio_temporal_scaling -->

---

<!-- START_Parallel_analysis -->
## Run analysis in parallel:
Allow the use of more than one core of the computer processor.
- **Checked** → A use multiple CPU cores to analyze arenas in parallel (faster).
- **Unchecked** → Single core analysis.

<!-- END_Parallel_analysis -->

---

<!-- START_Proc_max_core_nb -->
## Proc max core number:
Maximum number of logical CPU cores to use during analysis. Default is the available number of cores
minus one.

<!-- END_Proc_max_core_nb -->

---

<!-- START_Minimal_RAM_let_free -->
## Minimal RAM let free:
Amount of RAM to leave available for other programs.   Setting to `0` gives Cellects all memory, but
increases crash risk if other apps are open.

<!-- END_Minimal_RAM_let_free -->

---

<!-- START_Lose_accuracy_to_save_RAM -->
## Lose accuracy to save RAM:
For low
-memory systems:
- Converts video from `np.float64` to `uint8`
- Saves RAM at the cost of a slight precision loss

<!-- END_Lose_accuracy_to_save_RAM -->

---

<!-- START_Video_fps -->
## Video fps:
Frames per second of validation videos.

<!-- END_Video_fps -->

---

<!-- START_Keep_unaltered_videos -->
## Keep unaltered videos:
Keeps unaltered `.npy` videos in hard drive.
- **Checked** → Running the same analysis will be faster.
- **Unchecked** → These videos will be written and removed each run of the same analysis.
NB:
- Larges files: it is recommended to remove them once analysis is entirely finalized.

<!-- END_Keep_unaltered_videos -->

---

<!-- START_Save_processed_videos -->
## Save processed videos:
Saves lightweight processed validation videos (recommended over unaltered videos).   These videos
assess analysis accuracy and can be read in standard video players.

<!-- END_Save_processed_videos -->

---

<!-- START_Csc_for_video_analysis -->
## Color space combination for video analysis:
Advanced option to change RGB processing directly in the video tracking window.   Useful to test new
color spaces without redoing image analysis.

<!-- END_Csc_for_video_analysis -->

---

<!-- START_Night_mode -->
## Night mode:
Switches the application background between light and dark themes.

<!-- END_Night_mode -->

---

<!-- START_Reset_all_settings -->
## Reset all settings:
Useful when the software freeze with no apparent reason. To reset all settings, it removes the
config file in the  current folder as well as the config file in the software folder. Then, it
retrieves and saves the default parameters.

<!-- END_Reset_all_settings -->

---
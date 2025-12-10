# Advanced parameters

<figure>
  <img src="/static/UserManualFigure8.png" alt="Advanced parameters window" width="600">
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
Defines the minimal size threshold (in pixels) for considering an appearing shape as a specimen (e.g. bacterial colony).  
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
- Only useful when specimen(s) are invisible at the beginning of the experiment and appear progressively.

<!-- END_Appearance_detection_method -->

---

<!-- START_ -->
<!-- END_ -->
## Oscillatory parameters
- **Oscillatory period** → sets expected oscillation period for luminosity variations  
- **Minimal oscillating cluster size** → threshold to filter out small noisy oscillations  

These parameters are linked to *Oscillation Analysis* (see Required outputs).

---

<!-- START_ -->
<!-- END_ -->
## Fractal parameters
Currently under development.

---

<!-- START_ -->
<!-- END_ -->
## Network parameters
- **Network detection threshold** → minimum intensity difference to detect cytoplasmic tubular network  
- **Mesh side length** → rolling window size (pixels) for network detection  
- **Mesh step length** → step length (pixels) for rolling window

---

<!-- START_ -->
<!-- END_ -->
## Spatio-temporal scaling
Defines the spatiotemporal scale of the dataset:  
- Time between images or frames (minutes)  
- Option to convert areas/distances from pixels to mm/mm²  

---

<!-- START_ -->
<!-- END_ -->
## Run analysis in parallel
Checked → use multiple CPU cores to analyze arenas in parallel (faster).  
Unchecked → single core analysis.

---

<!-- START_ -->
<!-- END_ -->
## Proc max core number
Maximum number of logical CPU cores to use during analysis.  

---

<!-- START_ -->
<!-- END_ -->
## Minimal RAM let free
Amount of RAM to leave available for other programs.  
Setting to `0` gives Cellects all memory, but increases crash risk if other apps are open.

---

<!-- START_ -->
<!-- END_ -->
## Lose accuracy to save RAM
For low-memory systems:  
- Converts video from `np.float64` to `uint8`  
- Saves RAM at the cost of a slight precision loss

---

<!-- START_ -->
<!-- END_ -->
## Video fps
Frames per second of validation videos.

---

<!-- START_ -->
<!-- END_ -->
## Keep unaltered videos
Keeps unaltered `.npy` videos (faster re-runs, but large files).  
Recommended to delete them once analysis is finalized.

---

<!-- START_ -->
<!-- END_ -->
## Saved processed video
Saves lightweight processed validation videos (recommended over unaltered videos).  
Can be read in standard video players.

---

<!-- START_ -->
<!-- END_ -->
## Color space combination for video analysis
Advanced option to change RGB processing directly in the video tracking window.  
Useful to test new color spaces without redoing image analysis.

---

<!-- START_ -->
<!-- END_ -->
## Heterogeneous background
Segments the image into 2+ categories using a k-means algorithm.  
Much slower than thresholding but works in more complex environments.  
Should be used as a **last resort**.

---

<!-- START_ -->
<!-- END_ -->
## Night mode
Switches the application background between light and dark themes.
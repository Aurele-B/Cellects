# Advanced parameters

<figure>
  <img src="/static/UserManualFigure8.png" alt="Advanced parameters window" width="600">
  <figcaption><strong>Figure 8:</strong> Advanced parameters window</figcaption>
</figure>

---

## Automatically crop images
Uses the first image detection to crop all images and improve arena and last image detection.  

> 💡 If the analysis fails or the program crashes while running the image analysis window, unselecting this option may help.

---

## Subtract background
Takes the first image and subtracts it from every following image before analysis.  
Useful when cells are not visible in the first image.  
Can either improve or degrade detection depending on the dataset.

---

## Keep Cell and Back drawings for all folders
If the user drew cell and background regions during first image analysis:  
- **Checked** → keep this information for all folders  
- **Unchecked** → only use it for the current folder

---

## Correct errors around initial shape
Applied only if detection is less efficient around the initial shape than in the rest of the arena.  
Fills potential gaps around the initial shape during growth.  

⚠️ Do not use if the substrate has the same opacity everywhere (i.e. no difference between starting and growth regions).

---

## Prevent fast growth near periphery
Removes detection of specimens moving too fast near arena borders.  
Unchecked → no correction applied.

---

## Connect distant shapes
Useful when the specimen color is close to the background and causes disconnections.  
Reconnects disconnected parts of a specimen to the main shape dynamically.  

---

## All specimens have the same direction
Improves automatic arena detection when all specimens move in the same direction.  
Uncheck only if specimens move strongly and in many directions.

---

## Appearing cell/colony parameters
Defines the minimal size threshold (in pixels) for considering an appearing shape as a colony.  
Two methods available: detection by **size** or by **position** in the arena.

---

## Oscillatory parameters
- **Oscillatory period** → sets expected oscillation period for luminosity variations  
- **Minimal oscillating cluster size** → threshold to filter out small noisy oscillations  

These parameters are linked to *Oscillation Analysis* (see Required outputs).

---

## Fractal parameters
Currently under development.

---

## Network parameters
- **Network detection threshold** → minimum intensity difference to detect cytoplasmic tubular network  
- **Mesh side length** → rolling window size (pixels) for network detection  
- **Mesh step length** → step length (pixels) for rolling window

---

## Spatio-temporal scaling
Defines the spatiotemporal scale of the dataset:  
- Time between images or frames (minutes)  
- Option to convert areas/distances from pixels to mm/mm²  

---

## Run analysis in parallel
Checked → use multiple CPU cores to analyze arenas in parallel (faster).  
Unchecked → single core analysis.

---

## Proc max core number
Maximum number of logical CPU cores to use during analysis.  

---

## Minimal RAM let free
Amount of RAM to leave available for other programs.  
Setting to `0` gives Cellects all memory, but increases crash risk if other apps are open.

---

## Lose accuracy to save RAM
For low-memory systems:  
- Converts video from `np.float64` to `uint8`  
- Saves RAM at the cost of a slight precision loss

---

## Video fps
Frames per second of validation videos.

---

## Keep unaltered videos
Keeps unaltered `.npy` videos (faster re-runs, but large files).  
Recommended to delete them once analysis is finalized.

---

## Saved processed video
Saves lightweight processed validation videos (recommended over unaltered videos).  
Can be read in standard video players.

---

## Color space combination for video analysis
Advanced option to change RGB processing directly in the video tracking window.  
Useful to test new color spaces without redoing image analysis.

---

## Heterogeneous background
Segments the image into 2+ categories using a k-means algorithm.  
Much slower than thresholding but works in more complex environments.  
Should be used as a **last resort**.

---

## Night mode
Switches the application background between light and dark themes.
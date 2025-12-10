# Tune parameters of the video tracking window

<figure>
  <img src="/static/UserManualFigure5.png" alt="Cellects video tracking window" width="600">
  <figcaption><strong>Figure 5:</strong> Cellects video tracking window</figcaption>
</figure>

---

## Arena to analyze
The arena to analyze is a number allowing Cellects and the user to identify one arena in particular.  
If there is only one arena, this number should be one.  
Cellects automatically names the arena according to their position in the image, from left to right and from top to bottom.

---

## Maximal growth factor
This factor should be increased if the analysis underestimates the cell size, and decreased if it overestimates.  

It is a proportion of pixels in the image and indicates how far the specimen can possibly move or grow from one image to the next.  
More precisely, it is the upper limit of the proportion of the image that can go from background to specimen between frames.

---

## Repeat video smoothing
Increasing this value will decrease noise when the slope option is chosen.  

Cellects analyzes the intensity dynamics of each pixel in the video.  
When it uses a threshold algorithm based on derivative (slope evolution), it starts by smoothing every pixel intensity curve.  
This algorithm (a moving average) can be repeated several times to make the curve smoother.  
This value corresponds to the number of times to apply smoothing.

---

## Segmentation method and compute all options
Cellects offers five analysis options for video tracking:  

- **Frame option** → applies the algorithm used during the image analysis window, frame by frame, without temporal dynamics.  
- **Threshold option** → compares pixel intensity with the average intensity of the whole image at each time step.  
- **Slope option** → compares slope of the pixel intensity with an automatically defined slope threshold.  
- **T and S option** → logical AND of threshold and slope options.  
- **T or S option** → logical OR of threshold and slope options.  

NB:  
- When *Heterogeneous background* has been checked in the image analysis window, only the *Frame* option remains available.

---

## Load one arena
Clicking this button loads the arena corresponding to the number entered in *Arena to analyze*.  
The center of the window then displays the video of the corresponding arena. 

---

## Detection
*Detection* runs one or all options of video tracking for the chosen arena.  
It allows testing the effect of parameter changes in this window.  

Once detection seems valid, the user can answer “*Done*” to *Step 1: Tune parameters to improve Detection*, and proceed to *Post-processing*.

<figure>
  <img src="/static/UserManualFigure6.png" alt="Cellects video tracking window during detection visualization" width="600">
  <figcaption><strong>Figure 6:</strong> Cellects video tracking window during detection visualization</figcaption>
</figure>

---

## Fading detection
*Fading detection* (Fig. 6) is useful when the specimens not only grow but also move.  
This means a pixel that was covered may later be uncovered.  

When checked, *Fading detection* monitors cell-covered areas to decide when they are abandoned.  
It can take a value between -1 and 1:

- Near -1 → Cellects almost never detects abandonment  
- Near +1 → Cellects may wrongly remove detection everywhere  
- Too high values can also cause wave-like artifacts  

---

## Post processing
*Post-processing* (Fig. 6) applies the chosen detection algorithm to the video, plus additional algorithms to improve detection:  

- *Fading detection* (as described above)  
- *Correct errors around initial shape* (fine-tuning in advanced parameters, see Fig. 8)  
- *Connect distant shapes* (fine-tuning in advanced parameters, see Fig. 8)  
- Standard binary image operations: opening, closing, logical ops  

Once Post-processing works, the user can click “*Done*” to *Step 2: Tune fading and advanced parameters to improve Post-processing*, and then analyze all arenas (*Run All*).

<figure>
  <img src="/static/UserManualFigure7.png" alt="Cellects video tracking window, running all arenas" width="600">
  <figcaption><strong>Figure 7:</strong> Cellects video tracking window, running all arenas</figcaption>
</figure>

---

## Run All
If detection with *Post-processing* leads to a satisfying result for one arena,  
the user can apply the current parameters to all arenas.  

Clicking *Run All* will:  
1. Write the uncompressed video of each arena to the folder (can take a lot of space)  
2. Analyze videos one by one (Fig. 7)  
3. Save the *Required output* (.csv files), validation videos, and two snapshots per arena (after 1/10 of total time and at the last image)

---

## Save one result
This button is used only after running *Run All* at least once.  

If most arenas analyzed well, but one failed, the user can:  
1. Run all arenas once with the best parameters (even if imperfect for one arena)  
2. Adjust parameters to fix the failing arena  
3. Click *Save one result* to replace the saved results for that arena (CSV rows + validation video)
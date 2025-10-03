# Find where the specimens are in the image analysis window

<figure>
  <img src="/static/UserManualFigure2.png" alt="Cellects image analysis window" width="600">
  <figcaption><strong>Figure 2:</strong> Cellects image analysis window</figcaption>
</figure>

---

## Image number
Which image should be analyzed first? In most cases, it should be the first.  
Changing this number is only useful when cells are invisible on the first image 
(e.g. in the case of appearing colonies of bacteria).  
In that case, select an image showing visible cells in order to enable Cellects to find them.

---

## One cell or colony per arena
This option is automatically selected. If there is only one cell (or connected colony) per arena, leave this as it is.  
If there already are (or will be) several cells (or colonies) per arena, unselect this option.

---

## Scale with
The *Scale with* option gives a scale to the image.  
Cellects can determine this scale using the width (horizontal size) of the image or the width of the specimens 
on the first image (when they share the same width, Cellects can use the average pixel width of all specimens to get the scale).  

Additional notes:
- Using the width of the specimens decreases the first image detection efficiency, we recommend choosing the width of the image.  
- However, if the width of the specimens is known with more accuracy than the width of the image, choose the width of the specimens.  
- By default, distances and surfaces are in pixels (Cellects stores the size of one pixel in a file called `software_settings.csv`).  
  They can automatically be converted in mm² by checking the corresponding checkbox in the advanced parameters window (see Fig. 8).

---

## Scale size
The *Scale size* is the length (in mm) of the item(s) used for scaling. 

---

## Select and draw
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
- If each folder requires using this option, the user has to analyze each folder separately.

---

## Advanced mode
The *Advanced mode* allows Cellects to use a previously working set of parameters.  

Once generated, selecting these options with *Advanced mode* activated allow to:
- See the color space combination corresponding to the displayed image.  
- Try a new color space combination by mixing two good options.  
- Use a logical operator (AND/OR) between results of two color space combinations.  
- Apply grid segmentation using Otsu thresholding.  
- Use more than two colors with a k-means categorization.  

This option is powerful but should only be used by advanced users.

---

## Generate analysis option
Cellects suggests two algorithms to automatically find the best parameters to detect specimens on the first image:  

- **Quickly** → suggests options in a few minutes.  
- **Carefully** → browses more possibilities, takes longer.  

Alternatively, the user can select *Advanced mode* to view or modify the parameters selected by Cellects.  

Additional notes:
- Clicking on *Quickly* or *Carefully* (or *Visualize*) will make an orange working message appear.  
- If the user already used Cellects, the advanced mode will be faster.  

<figure>
  <img src="/static/UserManualFigure3.png" alt="Cellects image analysis window after analysis option generation" width="600">
  <figcaption><strong>Figure 3:</strong> Cellects image analysis window after analysis option generation</figcaption>
</figure>

---

## Select option to read
The drop-down menu on the left of *Select option to read* allows the user to visualize directly the result of 
the analysis corresponding to each option.  

For instance, in Fig. 3, the central image displays the result of option 1 and a message below informs the user that this option detected 6 distinct spots in 6 arenas.  
Since these two numbers are equal, there is only one specimen per arena, so the user should click the *Yes* button.  

Otherwise, the user can improve the analysis by setting the *arena shape*, the *spot shape*, or the *spot size*.  
The user can also draw more specimens or background using *Select and draw*, or use *Visualize* to set up manually a color space combination.

Once the magenta/pink contours correspond to the right position of all cells and the right number of detected spots, 
the user can click the *Yes* button.  

After clicking, an orange working message appears, and Cellects automatically looks for the coordinates of the contour of each arena.  

---

## Arena shape
Tells whether the specimen(s) can move in a circular or rectangular arena.

---

## Spot shape
Initial shape of the specimen(s) inside arena(s).

---

## Spot size
Initial horizontal size of the specimen(s).  
If similar across all specimens, this can also be used as a scale.

---

## Video delimitation
After validating the initial detection, the result of the automatic video delimitation appears in blue in the center of the window (see Fig. 4).  

If correct, click *Yes*.  
If incorrect, click *No*, and Cellects will suggest:
- A slower, more efficient algorithm  
- Or a manual delineation option  

<figure>
  <img src="/static/UserManualFigure4.png" alt="Cellects image analysis window, after arena delineation" width="600">
  <figcaption><strong>Figure 4:</strong> Cellects image analysis window, after arena delineation</figcaption>
</figure>

---

## Last image question
If the user thinks that parameters used on the first image might not work on later images, they can fine-tune them using the last image.  

- Clicking *Yes* → allows testing on the last image before moving on.  
- Clicking *No* → goes directly to video tracking.

---

## Check if the starting area differs from growing area
If the substrate changes between starting and growing areas (e.g. oat gel vs transparent agar), keep this checked.  
If the substrate is homogeneous everywhere, uncheck this option.
# Real-World Use Cases

## Case 1: Automated Physarum polycephalum tracking using GUI
**Problem**: Track the surface area over time of one individual.  
**Steps**:  
1. Launch the GUI  
Run:  
```bash
Cellects
```
2. Load images via the Data Localisation tab.
Browse the Cellects/data/experiment folder. 
Make sure that:
- Images prefix is 'im'
- Images extension 'is .tif'
- Arena number per folder is '1'
2. Click *Next* to switch to the Image analysis tab.
- Click the *Apply current config* to segment the first image
-> A magenta contour appears around the specimen
- If there*s 1 distinct specimen(s) in 1 arena(s), click *Yes*
-> A blue contour appears around the arena
- Arena delineation is correct, click *Yes*
- In this setting, the medium at starting position is opaque, check the bottom-left checkbox
- There*s no need to improve the segmentation using the last image here, click *No*
3. *Close* the Final checks to get to the Video analysis tab.
4. Click *Done* and run *Post processing*
Once done, click *Read* to see the result of the video tracking.
5. Click *Done* again and run *Save one result*
This will save the .csv files containing the time series describing the individual's growth.
NB:
Other descriptors are available using *Required output*.

## Case 2: Automated Physarum polycephalum tracking using API
**Steps**:  
1. Load the data
```python
import os
from cellects.core.script_based_run import load_data, run_image_analysis, write_videos, run_all_arenas
po = load_data(pathway=os.getcwd() + "/data/experiment", sample_number=1, extension='tif')
po = run_image_analysis(po)
po = write_videos(po)
run_all_arenas(po)
```
2. Find the ind_1.mp4 file in the folder: os.getcwd() + "/data/experiment"
This video summarizes the video segmentation

## Case 3: Colony growth tracking
**Problem**: Get the surface area over time of several appearing colonies
**Steps**:

```python
# 1. Generate the data
import os
from matplotlib import pyplot as plt
from cellects.utils.load_display_save import movie, show
from cellects.core.script_based_run import generate_colony_like_video, load_data, run_image_analysis, run_one_video_analysis

rgb_video = generate_colony_like_video()

# 2. Display the data (optional)
movie(rgb_video)

# 3. Segment the images
po = load_data(rgb_video)
po.vars['several_blob_per_arena'] = True
po = run_image_analysis(po, last_im=rgb_video[-1, ...])

# 4. Visualization of the first image segmentation
show(po.first_image.binary_image)

# 5. Video tracking
po.vars['maximal_growth_factor'] = 0.5
MA = run_one_video_analysis(po, with_video_in_ram=True)

# 4. Visualization of the video segmentation
movie(MA.binary)
plt.plot(MA.one_row_per_frame['time'], MA.one_row_per_frame['area_total'])
plt.show()
```

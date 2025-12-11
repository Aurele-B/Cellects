# Real-World Use Cases

## Case 1: Automated Physarum polycephalum tracking
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
6. 
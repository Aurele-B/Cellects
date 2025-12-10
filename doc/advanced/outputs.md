# Required outputs

<figure>
  <img src="/static/UserManualFigure9.png" alt="Required output window" width="600">
  <figcaption><strong>Figure 9:</strong> Required output window</figcaption>
</figure>

---

<!-- START_coord_specimen -->
## Pixels covered by the specimen(s):
Save a .npy file containing the coordinates (t, y, x) of presence of the specimen(s), as detected
with the current parameters.
NB:
- Depending on the number of frames, these files may take a lot a memory.

<!-- END_coord_specimen -->

---


<!-- START_Graph -->
## Graph of the specimen(s) (or network):
Compute the geometrical graph describing the specimen as it is detected with the current parameters.
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

<!-- END_Graph -->

---


<!-- START_coord_oscillating -->
## Oscillating areas in the specimen(s):
Compute and save the coordinates (t, y, x) of areas oscillating synchronously in the specimen(s).
This algorithm save two .npy file: one for thickening areas and one for slimming areas.

<!-- END_coord_oscillating -->

---


<!-- START_coord_network -->
## Network in the specimen(s):
Detect and save (as .npy file) the coordinate (t, y, x) of a distinguishable network in the
specimen(s).

<!-- END_coord_network -->

---

## Save descriptors
Saves selected descriptors in `.csv` files at the end of the analysis (*Run all*).  

Available descriptors include:  
- Area  
- Perimeter  
- Circularity  
- Rectangularity  
- Total hole area  
- Solidity  
- Convexity  
- Eccentricity  
- Euler number  
- Standard deviation (x, y)  
- Skewness (x, y)  
- Kurtosis (x, y)  
- Major axes lengths and angles  
- Growth transitions
- Oscillations
- Minkowski dimension

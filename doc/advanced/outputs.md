# Required outputs

<figure>
  <img src="/static/UserManualFigure9.png" alt="Required output window" width="600">
  <figcaption><strong>Figure 9:</strong> Required output window</figcaption>
</figure>

---

## Save presence coordinates
Saves very large `.npy` files containing all presence coordinates of specimens:  
- Time  
- Y position  
- X position  

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

Additional algorithms:  
- **Detect growth transition** → automatically detects isotropic to digitated growth transition *(Vogel et al. 2016)*  
- **Proceed oscillation analysis** → analyzes intracytoplasmic oscillations of *P. polycephalum*  
  (based on *Boussard et al., 2021*)  
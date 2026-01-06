---
title: 'Cellects: a software to quantify cell expansion and motion'
tags:
  - Growth and motion quantification
  - 2D biological growth
  - Automated tracking
  - Cell expansion
  - Time-lapse analysis
  - High throughput quantification
  
authors:
  - name: Aurèle Boussard
    orcid: 0000-0002-6083-4272
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Manuel Petit
    orcid: 0000-0002-3211-0066
    equal-contrib: false
    affiliation: 2
  - name: Patrick Arrufat
    orcid: 0000-0002-2073-3725
    equal-contrib: false
    affiliation: 1
  - name: Audrey Dussutour
    orcid: 0000-0002-1377-3550
    equal-contrib: false 
    affiliation: 1
  - name: Alfonso Pérez-Escudero
    orcid: 0000-0002-4782-6139
    equal-contrib: false 
    affiliation: 1
    
affiliations:
 - name: Research Centre on Animal Cognition (CRCA), Centre for Integrative Biology (CBI), Toulouse University, CNRS
   index: 1
   ror: 02v6kpv12
 - name: Morphogenesis Simulation and Analysis In silico (MOSAIC), Plant development and reproduction (RDP), École normale supérieure (ENS) de Lyon
   index: 2
   ror: 04zmssz18
 - name: Jacques Louis Lions Laboratory (LJLL), Sorbonne Université, Paris
   index: 3
   ror: 04xmteb38
date: 3 January 2026
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
Cellects is a versatile open-source software designed for automated quantification of biological growth dynamics across diverse systems. It is particularly useful for researchers studying biological systems ranging from fungi colonies (\autoref{fig:cell_dyn_morph}a-d) to developing embryos  (\autoref{fig:cell_dyn_morph}e-g) and slime molds (\autoref{fig:cell_dyn_morph}a-d), offering accurate measurements in diverse experimental conditions. Its robust algorithms address challenges such as variable contrast, complex morphologies, and multi-arena analysis, enabling accurate tracking and geometric characterization of organisms ranging from fungi to developing embryos. The tool provides detailed insights into growth patterns, motion dynamics, and morphological changes through user-friendly tools that eliminate the need for manual analysis. Its adaptability makes it suitable for a wide range of applications—from tracking microbial colonies to analyzing complex cellular networks—ensuring high accuracy even under challenging imaging conditions.
For example, fungal initial and final growth stages are captured with segmentation contours in Figure \autoref{fig:cell_dyn_morph}a–b, while area and perimeter curves derived from the whole growth process illustrate temporal dynamics in \autoref{fig:cell_dyn_morph}c,d. The software can also be used to track marked cellular tissues (\autoref{fig:cell_dyn_morph}e,f), and produce the data to represent their directional movement (\autoref{fig:cell_dyn_morph}g). Cellects also quantifies morphological descriptors from the segmentation result of every frame (panel 1h). For instance, it can compute convex hulls and bounding rectangles (\autoref{fig:cell_dyn_morph}i) to assess shape complexity through descriptors such as solidity, convexity or eccentricity. Cellects is equipped with a segmentation pipeline for tubular network structures (\autoref{fig:cell_dyn_morph}j) and can reconstruct a graph (\autoref{fig:cell_dyn_morph}k) from it, or from a connected component of any shape. Then, Cellects save the graph components and related features, including vertex connectivities (\autoref{fig:cell_dyn_morph}l) and edge lengths (\autoref{fig:cell_dyn_morph}m). Cellects’ user manual includes all available descriptors.

The software’s flexibility is demonstrated by its application to fungal growth, embryonic cell tracking, and slime mold network analysis, supported by a user-friendly interface for parameter tuning and validation. By integrating multiple arenas, multi-individual tracking, and drift-correction capabilities, Cellects provides standardized outputs saved in .csv files for further analysis.

![Figure 1. /label{#fig:cell_dyn_morph}](paper/figures/CellectsFigure1.jpg){ width=100% }
**Figure 1**: Cellular dynamics and morphologies across systems. **a,b)** Fungal growth from initial (a) to final (b) stages with green segmentation contours, from [@Penil2018]; **c,d)** Corresponding area and perimeter curves over time (c: area, d: perimeter). **e,f)** Tracking of cellular progenitors marked with GFP during the development of a quail embryo (C. japonica, caudal part of stage HH11, from [@Romanos2021]): initial (e) and final (f) images with segmentation lines. **g)** Spider plot of quail cell population movement directions. **h,i)** *Physarum polycephalum* morphology after 16:40 hours of exploration, from our lab. (h: cell segmentation, i: convex hull (orange) and bounding rectangle (green)). **j,k)** Network segmentation (j: blue pseudopods; turquoise network) and graph reconstruction (k: edges colored by width, green vertices, magenta food sources). **l,m)** *P. polycephalum* connectivity metrics: vertex degrees (l) and edge lengths (m). Panels a–d (fungus), e–g (quail), h–m (*P. polycephalum*).

# Statement of need

The proliferation of imaging technologies has enabled high-resolution, time-resolved studies of biological growth across scales—from molecular aggregation to organismal development—yet automated analysis of such datasets remains a critical bottleneck. Existing tools for quantifying growth often suffer from significant limitations that hinder their utility in diverse experimental contexts. First, most software is specialized for single organisms (e.g., bacteria [@Ernebjerg2012], yeast [@Falconnet2011]), failing to generalize across taxa with distinct morphological and optical properties. Second, many lack user-friendly graphical interfaces, requiring advanced programming skills or manual scripting [@Pandey2021]. Third, commercial solutions (e.g., ScanLag [@levin2014scanlag], ColTapp [@Bar2020]) are closed-source, restricting customization and community-driven improvements. Fourth, they often provide limited output variables, necessitating post-analysis manual processing with tools like ImageJ/Fiji [@Schneider2012]. Fifth, current programs struggle to adapt to varying lighting, coloration, or contrast conditions, which are common in heterogeneous biological systems such as *Physarum polycephalum* plasmodia.

To address these gaps, we developed Cellects, an open-source software with a flexible, user-friendly interface designed for multi-organism growth quantification. Cellects integrates dynamic segmentation algorithms to handle poor contrast and variable lighting while outputting extensive geometric descriptors (area, perimeter, solidity, etc.) in real-time time series. Its modular design accommodates diverse taxa—including fungi, plants, and slime molds—and supports validation tools for result refinement. By overcoming the limitations of existing platforms, Cellects enables robust, accessible analysis of growth dynamics across biological systems, fostering reproducibility and cross-disciplinary research.

The software's robustness is validated through \autoref{fig:validation}a-b: Segmentation accuracy across five experimental conditions (high contrast + optimal setup, heterogeneous colors, low contrast + desiccation, low contrast + optimal, low resolution) shows >97% accuracy in challenging scenarios. This validation demonstrates Cellects' ability to adapt to variable lighting, color schemes, and background complexity while maintaining high precision.
Cellects advance high-throughput studies by reducing observer bias while enabling reproducible quantification across diverse biological models. The software's capabilities are demonstrated through diverse applications including fungi growth monitoring (Figures 1a-d, [@Penil2018]), developmental biology studies (Figures 1e-g, quail embryos), and morphological analysis of complex organisms (Figures 1h-m; Figures 2ab, *Physarum polycephalum*).

![Figure 2.
/label{#fig:validation}](paper/figures/CellectsFigure2.jpg){ width=100% }
**Figure 2**: Validation of Cellects across five experimental conditions. **a)** Image of a slime mold, showing the two types of errors detected in our validation: Pixels that belong to the cell but are segmented as background (orange), and pixels that belong to the background but were segmented as cell (green). **b)** Accuracy of the segmentation in 5 different experimental conditions (shown below the bars). The five conditions are (in order) high contrast and optimal setup, high contrast with heterogeneous colors, low contrast and setup prone to desiccation, low contrast and optimal setup, low resolution. Orange: Proportion of cell pixels correctly identified as cell. Green: Proportion of background pixels correctly identified as background. Error bars show the 95% confidence interval. Percentages on top show the average of both bars. 

# Acknowledgements

We thank Audrey Bizet for her work on the first experiment of Figure 3b, Charlotte Dupont and 
Paul-Antoine Badon for the second experiment of Figure 3b, Nirosha Murugan for her help with 
the fourth experiment of Figure 3b, Ana Lucía Morán Hernández for her help with the fifth 
experiment of Figure 3b, and Florent Le Moël and Remi Giorno dit Journo for their help during 
software development.

# References

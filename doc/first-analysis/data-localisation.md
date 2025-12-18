# Data localisation in the first window

Before diving into detailed analysis workflows, Cellects requires initial setup (Fig. 1) to define the scope and format of the data being processed. This first window serves as the gateway to configuring foundational parameters that underpin all subsequent steps: from basic image/video tracking to advanced video analytics. 
Users will specify whether their input consists of an image stack or a single video file, establish naming conventions for targeted files (e.g., prefixes like "exp_" and extensions like ".jpg"), and define the root folder(s) containing experimental data. Additionally, this interface allows users to declare how many independent arenas are present in each dataset—a critical step for ensuring accurate analysis. 
These configurations directly inform later stages (e.g., image analysis, video tracking) and enable automation across multiple folders if required (See Fig. 11).

# Detailed description

<figure>
  <img src="../../static/UserManualFigure1.png" alt="Cellects first window" class="center"
       style="display:block;float:none;margin-left:auto;margin-right:auto;width:100%">
  <figcaption><strong>Figure 1:</strong> Cellects first window</figcaption>
</figure>


---

<!-- START_Image_list_or_videos -->
## Image list or videos:
The *Image list or video* option indicates whether the data have been stored as an image stack (i.e.
a set of files where each file contains a single image) or as a video. Images must be named
alphanumerically so the program can read them in the right order.

<!-- END_Image_list_or_videos -->

---

<!-- START_Image_prefix_and_extension -->
## Image prefix and extension:
The *Images prefix* and *Images extension* fields allow Cellects to consider relevant data. For
example, setting 'exp_' as image prefix and '.jpg' as image extension will cause Cellects to only
consider JPG files whose name starts with 'exp_'. Remaining labels should indicate the order in
which images were taken.
!!! note

	 - Image prefix is optional
	 - If every .jpg files start with IMG_ but other .jpg files exist, use the prefix to excludeirrelevant files.
	 - Supported formats: bmp, dib, exr, exr, hdr, jp2, jpe, jpeg, jpg, pbm, pfm, pgm, pic, png,  pnm,ppm, ras, sr, tif, tiff, webp, cr2, cr3, nef, arw, sr2, raf, prf, rw2, pef, dng, 3fr, iiq.
<!-- END_Image_prefix_and_extension -->

---
<!-- START_Folder -->
## Folder:
The *Folder* field must specify the directory path to the folder(s) for Cellects to be able to run
the analysis. The user can copy/paste this path into the field or navigate to the folder using the
*Browse* push button. For batch analysis, provide a path leading directly to the parent folder
containing all subfolders.

<!-- END_Folder -->

---

<!-- START_Arena_number_per_folder -->
## Arena number per folder:
The *Arena number per folder* specifies how many arenas are present in the images. Cellects will
process and analyze each arena separately.
!!! note

	 - For batch processing, assign different arena counts for each subfolder (see Fig. 11: the severalfolder window).
<!-- END_Arena_number_per_folder -->

---

<!-- START_Browse -->
## Browse:
Clicking the *Browse* button opens a dialog to select a folder for analysis.

<!-- END_Browse -->

---

<!-- START_Advanced_parameters -->
## Advanced parameters:
Clicking the *Advanced parameters* button opens the window containing all secondary parameters of
the software.  Find details about this window in the advanced documentation.

<!-- END_Advanced_parameters -->

---

<!-- START_Required_outputs -->
## Required outputs:
Clicking the *Required outputs* button opens the window allowing to choose what descriptors Cellects
will compute on the selected data. Find details about this window in the advanced documentation.

<!-- END_Required_outputs -->

---

<!-- START_Run_all_directly -->
## Run all directly:
This option appears when image analysis has already been performed for the current folder. It is a
shortcut to bypass the image analysis step and proceed directly to video tracking refinement.

<!-- END_Run_all_directly -->

---

<!-- START_Next -->
## Next:
Click the *Next* button to go to the image analysis window (Fig. 2), or  to the window showing the
list of folders (Fig. 11) if applicable.

<!-- END_Next -->

---
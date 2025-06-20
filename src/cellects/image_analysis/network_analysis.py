
import os
from pathlib import Path
from cellects.utils.utilitarian import *
import matplotlib
matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
from skimage import morphology
from numba.typed import Dict as TDict
from cellects.image_analysis.network_functions import *
from cellects.image_analysis.fractal_functions import *
from cellects.image_analysis.image_segmentation import *


square_33 = np.ones((3, 3), np.uint8)
"""
Prepare data
"""
video_nb = 2
os.chdir("/Users/Directory/Data/dossier1")
binary_coord = np.load(f"coord_specimen{video_nb}_t720_y1475_x1477.npy")
binary_video = np.zeros((720, 1475, 1477), np.uint8)
binary_video[binary_coord[0, :],binary_coord[1, :], binary_coord[2, :]] = 1
origin = binary_video[0, ...]
net_coord = np.load(f"coord_tubular_network{video_nb}_t720_y1475_x1477.npy")
visu = np.load(f"ind_{video_nb}.npy")
first_dict = TDict()
first_dict["lab"] = np.array((0, 0, 1))
first_dict["luv"] = np.array((0, 0, 1))
im_nb = -1
binary_im = binary_video[im_nb, ...]
greyscale_img, _ = generate_color_space_combination(visu[im_nb, ...], list(first_dict.keys()), first_dict, convert_to_uint8=True)
# a = np.array((net_coord))
# a = a[1:,a[0,:]==im_nb]
# import pandas as pd
# pd.DataFrame(a).to_csv(f"/Users/Directory/Scripts/mathematica/training/network_coord_{binary_video.shape[1:]}.csv")


net_vid = np.zeros((720, 1475, 1477), dtype=np.uint8)
net_vid[net_coord[0], net_coord[1], net_coord[2]] = 1

# See(net_vid[im_nb, ...], keep_display=1)

# MAYBE REMOVE : Make sure that the network is only one connected component
full_network, stats, centers = cc(net_vid[im_nb, ...])
full_network[full_network > 1] = 0
network_contour = get_contours(full_network)

# Compute medial axis
skeleton, distances = morphology.medial_axis(full_network, return_distance=True)
distances[np.nonzero(origin)] = 0
# width = 10
# skel_size  = skeleton.sum()
# while width > 0 and skel_size > skeleton.sum() * 0.75:
#     width -= 1
#     skeleton = skeleton.copy()
#     skeleton[distances > width] = 0
#     # Only keep the largest connected component
#     skeleton, stats, _ = cc(skeleton)
#     skeleton[skeleton > 1] = 0
#     skel_size = skeleton.sum()
width = 10
skeleton = skeleton.copy()
skeleton[distances > width] = 0
# Remove the origin
skeleton = skeleton * (1 - origin)
# Only keep the largest connected component
skeleton, stats, _ = cc(skeleton)
skeleton[skeleton > 1] = 0

nb, shape = cv2.connectedComponents(skeleton)
# skeleton = skeleton[450:525, 775:855]

# Find vertices and edges
skeleton, vertices, edges = get_vertices_and_edges_from_skeleton(skeleton)

numbered_vertices, numbered_edges, vertices_table, edges_labels = get_graph_from_vertices_and_edges(vertices, edges)

numbered_vertices, numbered_edges, vertices_table, edges_labels = add_central_vertex(numbered_vertices, numbered_edges, vertices_table, edges_labels, origin, network_contour)

edges_table = get_edges_table(numbered_edges, distances, greyscale_img)

pathway = Path(f"/Users/Directory/Scripts/mathematica/training/")
save_network_as_csv(full_network, skeleton, vertices_table, edges_table, edges_labels, pathway)

save_graph_image(binary_im, full_network, numbered_edges, distances, origin, vertices_table, pathway)











edges = pd.read_csv(f"/Users/Directory/Scripts/mathematica/training/edges_labels_imshape={skeleton.shape}.csv")
vertices_coord = pd.read_csv(f"/Users/Directory/Scripts/mathematica/training/vertices_coord_imshape={skeleton.shape}.csv")
net_coord = pd.read_csv(f"/Users/Directory/Scripts/mathematica/training/full_net_coord_imshape={skeleton.shape}.csv")
skeletonbis = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
skeletonbis[net_coord.iloc[:, 0], net_coord.iloc[:, 1], :] = 150, 50, 40
skeletonbis[vertices_coord.iloc[:, 1], vertices_coord.iloc[:, 2], :] = 20, 255, 20
terminations_coord = vertices_coord.loc[vertices_coord.iloc[:, 3] == 1, :]
skeletonbis[terminations_coord.iloc[:, 1], terminations_coord.iloc[:, 2], :] = 255, 255, 255

cv2.imwrite(f"/Users/Directory/Scripts/mathematica/training/full_description.jpg", skeletonbis)

im = edges.copy()
im[np.nonzero(vertices)[0],np.nonzero(vertices)[1]] = 2
plt.imshow(im)
plt.show()
plt.savefig(f"/Users/aurele/Documents/graphics/meshes/edges_and_vertices.png")
plt.close()
plt.imshow(full_network[700:750, 600:700])
plt.show()
plt.imshow(skeleton[450:525, 775:855])
plt.show()
plt.imshow(skeletonbis)
plt.show()
plt.imshow(numbered_vertices)
plt.show()
plt.imshow(skeleton[650:(origin_centroid[0]+1), 650:(origin_centroid[1]+1)])
plt.show()
plt.imsave(f"/Users/Directory/Scripts/mathematica/training/full_description.jpg", skeletonbis, dpi=500)
plt.close()
np.save(f"/Users/aurele/Documents/graphics/meshes/vertices_1475_1477.npy", np.nonzero(vertices))
np.save(f"/Users/aurele/Documents/graphics/meshes/dilated_vertices_1475_1477.npy", np.nonzero(dilated_vertices))

# fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
# ax = axes.ravel()
#
# ax[0].imshow(net_vid[im_nb, ...], cmap=plt.cm.gray)
# ax[0].set_title('original')
# ax[0].axis('off')
#
# ax[1].imshow(skel1, cmap=plt.cm.gray)
# ax[1].set_title('skeletonize')
# ax[1].axis('off')
#
# ax[2].imshow(skel2, cmap=plt.cm.gray)
# ax[2].set_title('skeletonize (Lee 94)')
# ax[2].axis('off')
#
# ax[3].imshow(dist_on_skel, cmap='magma')
# ax[3].contour(net_vid[im_nb, ...], [0.5], colors='w')
# ax[3].set_title("medial axis")
# ax[3].axis('off')
#
# fig.tight_layout()
# plt.show()
"""
Display network
"""


"""
Compute fractal dimension
"""
zoomed_binary, side_lengths = prepare_box_counting(net_vid[im_nb, ...], min_im_side=128, min_mesh_side=8,zoom_step=0, contours=False)
binary_image=zoomed_binary
for box_diameter in side_lengths:
    display_boxes(zoomed_binary, box_diameter)
    plt.savefig(f"/Users/aurele/Documents/graphics/meshes/{box_diameter}.png", dpi=300)
    plt.close()
box_counting_dimension, r_value, box_nb = box_counting(zoomed_binary, side_lengths, True)
plt.savefig(f"/Users/aurele/Documents/graphics/meshes/full network.png", dpi=300)
plt.close()

zoomed_binary, side_lengths = prepare_box_counting(net_vid[im_nb, ...], min_im_side=128, min_mesh_side=8, zoom_step=0, contours=True)
box_counting_dimension, r_value, box_nb = box_counting(zoomed_binary, side_lengths, True)
plt.savefig(f"/Users/aurele/Documents/graphics/meshes/contour network.png", dpi=300)
plt.close()

zoomed_binary, side_lengths = prepare_box_counting(skeleton.astype(np.uint8), min_im_side=128, min_mesh_side=8, zoom_step=0, contours=False)
box_counting_dimension, r_value, box_nb = box_counting(zoomed_binary, side_lengths, True)
plt.savefig(f"/Users/aurele/Documents/graphics/meshes/medial axis network.png", dpi=300)
plt.close()

"""
Create edges_table
"""
visu = np.load(f"ind_{video_nb}.npy")
dims = visu.shape[:3]
first_dict = TDict()
first_dict["lab"] = np.array((0, 0, 1))
first_dict["luv"] = np.array((0, 0, 1))

converted_video = np.zeros((dims[0], dims[1], dims[2]), dtype=np.uint8)
for f_i in np.arange(dims[0]):
    bgr_image = visu[f_i, ...]
    converted_video[f_i, ...], _ = generate_color_space_combination(bgr_image, list(first_dict.keys()), first_dict,
                                                                    convert_to_uint8=True)

column_names = ["edge_name", "node1", "node2", "length", "avg_width", "avg_int", "betweenness_centrality", "period"]


edges_table = np.zeros((nb - 1, len(column_names)), dtype=np.float64)
edges_table[:, 0] = np.arange(1, nb)
edges_table[:, 3] = stats[1:, 4]
for i in range(1, nb):
    seg = numbered_edges == i
    seg_coord = np.nonzero(seg)
    edges_table[i - 1, 4] = np.mean(dist_on_skel[seg_coord])
    edges_table[i - 1, 5] = np.mean(converted_video[i - 1, seg_coord[0], seg_coord[1]])
    seg_with_vertices = dilated_vertices + seg

    res = a_star_path(vertices, seg)
seg2 = seg.copy()
seg2[res[0], res[1]] = 2
sub_arr = seg[185:205, 530:550].copy()
sub_arr = seg_with_vertices[185:205, 530:550].copy()
sub_arr = dilated_vertices[185:205, 530:550].copy()

lengths = edges_table[:, 3]       # Column index 3: length
widths = edges_table[:, 4]        # Column index 4: avg_width
intensities = edges_table[:, 5]   # Column index 5: avg_int


fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].hist(lengths[lengths > 0], bins=30, color='skyblue', edgecolor='black')
axs[0].set_title("Edge Length Distribution")
axs[0].set_xlabel("Length")
axs[0].set_ylabel("Frequency")

axs[1].hist(widths[widths > 0], bins=30, color='salmon', edgecolor='black')
axs[1].set_title("Average Width Distribution")
axs[1].set_xlabel("Avg Width")

axs[2].hist(intensities[intensities > 0], bins=30, color='lightgreen', edgecolor='black')
axs[2].set_title("Average Intensity Distribution")
axs[2].set_xlabel("Avg Intensity")

plt.tight_layout()
plt.show()


plt.imshow(sub_arr*255)
plt.show()
plt.close()







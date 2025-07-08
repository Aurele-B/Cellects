
from cellects.utils.utilitarian import *
from numba.typed import Dict as TDict
from cellects.image_analysis.network_functions import *
from cellects.image_analysis.fractal_functions import *
from cellects.image_analysis.image_segmentation import *


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
im_nb = -1 #  -1   466
binary_im = binary_video[im_nb, ...]
greyscale_img, _ = generate_color_space_combination(visu[im_nb, ...], list(first_dict.keys()), first_dict, convert_to_uint8=True)
# a = np.array((net_coord))
# a = a[1:,a[0,:]==im_nb]
# import pandas as pd
# pd.DataFrame(a).to_csv(f"/Users/Directory/Scripts/mathematica/training/network_coord_{binary_video.shape[1:]}.csv")


net_vid = np.zeros((720, 1475, 1477), dtype=np.uint8)
net_vid[net_coord[0], net_coord[1], net_coord[2]] = 1

# See(net_vid[im_nb, ...], keep_display=1)

# 0-pad the contour of the image:
pad_network = np.pad(net_vid[im_nb, ...], [(1, ), (1, )], mode='constant')
pad_origin = np.pad(origin, [(1, ), (1, )], mode='constant')

# MAYBE REMOVE : Make sure that the network is only one connected component
pad_network, stats, centers = cc(pad_network)
pad_network[pad_network > 1] = 0
# np.save(f"/Users/Directory/Scripts/mathematica/training/full_network_rep{video_nb}_IMG_4328.npy", full_network)
pad_net_contour = get_contours(pad_network)
# Compute medial axis
pad_skeleton, pad_distances = get_skeleton_and_widths(pad_network, pad_origin)

# nb, shape = cv2.connectedComponents(skeleton)
# skeleton = skeleton[450:525, 775:855]

# Find vertices and edges
pad_skeleton = keep_one_connected_component(pad_skeleton)
pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
branches_coord, tips_coord = get_branches_and_tips_coord(pad_vertices, pad_tips)
connecting_vertices_coord, edge_lengths, edges_pixel_coords = find_closest_vertices(pad_skeleton, branches_coord, tips_coord)
pad_skeleton, pad_distances, edges_labels, numbered_vertices, edges_pixel_coords, tips_coord, branches_coord, vertices_connecting_tips_coord = remove_tipped_edge_smaller_than_branch_width(pad_skeleton, connecting_vertices_coord, pad_distances, edge_lengths, tips_coord, branches_coord, edges_pixel_coords)




nb_e, tempo_numbered_edges, numbered_branches, numbered_tips = get_numbered_edges_and_vertices(pad_skeleton, pad_vertices, pad_tips)

# branch_vertices = (np.array([2, 3]), np.array([3, 7]))  # (y-coords, x-coords)
# branch_vertices = np.nonzero(pad_branches)  # (y-coords, x-coords)
#
# # Coordinates of tips
# tip_vertices = (np.array([1, 3]), np.array([5, 7]))
# tip_vertices = np.nonzero(pad_tips)
#
# pad_edges = (1 - pad_vertices) * pad_skeleton
# branches = pad_vertices - pad_tips
# # Find closest branching vertices
find_closest_vertex(pad_skeleton, branch_vertices, tip_vertices)


numbered_vertices, numbered_edges, vertices_table, edges_labels = get_graph_from_vertices_and_edges(pad_vertices, pad_edges, pad_distances)

network, net_contour, skeleton, distances, vertices, edges = remove_padding([pad_network, pad_net_contour, pad_skeleton, pad_distances, pad_vertices, pad_edges])

numbered_vertices, numbered_edges, vertices_table, edges_labels = add_central_vertex(numbered_vertices, numbered_edges, vertices_table, edges_labels, origin, network_contour)

edges_table = get_edges_table(numbered_edges, distances, greyscale_img)

pathway = Path(f"/Users/Directory/Data/dossier1/plots")
save_network_as_csv(net_vid[im_nb, ...], skeleton, vertices_table, edges_table, edges_labels, pathway)

save_graph_image(binary_im, net_vid[im_nb, ...], numbered_edges, distances, origin, vertices_table, pathway)




a, st, _ = cc(skeleton.astype(np.uint8))
a, st, _ = cc(((numbered_edges+numbered_vertices)>0).astype(np.uint8))




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
plt.imshow(full_network)
plt.show()
plt.imshow(skeleton[260:266, 562:570])
plt.show()
plt.imshow(full_network[260:266, 562:570])
plt.show()
plt.imshow((a>0)[260:266, 562:570])
plt.show()
plt.imshow((a==1))
plt.show()
plt.imshow(skeleton[650:(origin_centroid[0]+1), 650:(origin_centroid[1]+1)])
plt.show()
plt.imsave(f"/Users/Directory/Scripts/mathematica/training/full_description.jpg", skeletonbis, dpi=500)
plt.close()
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







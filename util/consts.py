# %%
# import pandas as pd
import numpy as np
import os
# os.chdir('/home/mil/kawana/workspace/dimr_sapien')

MEAN_COLOR_RGB = np.array([127., 127., 127.], dtype=np.float32)
MEAN_COLOR_RGB_NORMALIZED = np.array([0.5] * 3, dtype=np.float32)
# MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673], dtype=np.float32)

# RFS_labels = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture', 'kitchen_cabinet', 'display', 'trash_bin', 'other_shelf', 'other_table']
# # print(len(RFS_labels))
# CAD_labels = ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']
# # print(len(CAD_labels))
# %%
if os.getenv("BG_FIRST", "0") == "1":
    RFS_labels = [
        'background',
        'dishwasher',
        'trashcan',
        'safe',
        'oven',
        'storagefurniture',
        'table',
        'microwave',
        'refrigerator',
        'washingmachine',
        'box',
    ]
    CAD_labels = RFS_labels[1:]
else:
    RFS_labels = [
        'dishwasher',
        'trashcan',
        'safe',
        'oven',
        'storagefurniture',
        'table',
        'microwave',
        'refrigerator',
        'washingmachine',
        'box',
        'background',
    ]
    CAD_labels = RFS_labels[:1]

# CAD_cnts = [555, 1093, 212, 113, 232, 260, 191, 121]
CAD_cnts = [4370, 4243, 4218, 4379, 4504, 4336, 3941, 4396, 4460, 4229]
CAD_weights = np.sum(CAD_cnts) / np.array(CAD_cnts)

# CAD2ShapeNetID = ['4379243', '3001627', '2871439', '4256520', '2747177', '2933112', '3211117', '2808440']
CAD2ShapeNetID = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009']
CAD2ShapeNet = {k: v for k, v in enumerate(range(len(CAD2ShapeNetID)))} # selected 8 categories from SHAPENETCLASSES
# CAD2ShapeNet = {k: v for k, v in enumerate([1, 7, 8, 13, 20, 31, 34, 43])} # selected 8 categories from SHAPENETCLASSES
ShapeNet2CAD = {v: k for k, v in CAD2ShapeNet.items()}

# # cabinet, display, and bathtub (sink) may fly.
# CADNotFly = [0, 1, 2, 3, 4]
CADNotFly = list(range(len(CAD_labels)))


# assert exist, label map file.
# raw_label_map_file = 'datasets/scannet/rfs_label_map.csv'
# raw_label_map = pd.read_csv(raw_label_map_file)
# # %%
# RFS2CAD = {} # RFS --> cad
# for i in range(len(raw_label_map)):
#     row = raw_label_map.iloc[i]
#     RFS2CAD[int(row['rfs_ids'])] = row['cad_ids']

offset = len(RFS_labels) - len(CAD_labels)
if os.getenv("BG_FIRST", "0") == "1":
    RFS2CAD = {i: i if i < len(CAD_labels) else -1 for i in range(len(RFS_labels))}
else:
    RFS2CAD = {i: (i - 1) if len(RFS_labels) else -1 for i in range(len(RFS_labels))}
RFS2CAD_arr = np.ones(30) * -1
for k, v in RFS2CAD.items():
    RFS2CAD_arr[k] = v
# # %%
# import trimesh
# box1 = trimesh.creation.box()
# box = trimesh.creation.annulus(0.5, 1, 0.5)
# n = trimesh.intersections.slice_mesh_plane(box, plane_normal=[1, 1, 1], plane_origin=[0, 0, 0.5], cap=True)
# n = n.union(box1)
# # n = trimesh.Scene([box1, n])
# n.show()
# # %%

# %%

import os
import numpy as np

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def padding_bounding_box(bb, img_size, padding=32):
    bounding_box = np.zeros(4, dtype=np.int32)

    bounding_box[0] = np.maximum(bb[3] - padding / 2, 0)
    bounding_box[1] = np.maximum(bb[0] - padding / 2, 0)
    bounding_box[2] = np.minimum(bb[1] + padding / 2, img_size[1])
    bounding_box[3] = np.minimum(bb[2] + padding / 2, img_size[0])

    return bounding_box

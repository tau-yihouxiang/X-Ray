import glob
import os
import numpy as np
import tqdm
import PIL
from PIL import Image
import time
import random
from scipy.sparse import csr_matrix
import shutil

def load_xrays( xrays_path):
	loaded_data = np.load(xrays_path)
	loaded_sparse_matrix = csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])
	original_shape = (16, 7, 256, 256)
	restored_array = loaded_sparse_matrix.toarray().reshape(original_shape)
	return restored_array

src_root = "/hdd/taohu/Data/Objaverse/Data/Render/Objaverse_XRay_Raw"
dst_root = "/hdd/taohu/Data/Objaverse/Data/Render/Objaverse_XRay"
xrays_paths = glob.glob(os.path.join(src_root, "**/*.npz"), recursive=True)
# shuffle
# random.shuffle(xrays_paths)

count = 0
count_iou = 0
progress_bar = tqdm.tqdm(xrays_paths)
for xray_path in progress_bar:
    count += 1
    xrays = load_xrays(xray_path)
    image_values_pil = Image.open(xray_path.replace("xrays", "images").replace(".npz", ".png"))
    _, _, _, mask = image_values_pil.split()

    xray = (xrays[0, 0] > 0).astype(np.float32)
    mask = (np.array(mask.resize(xray.shape)) / 255 > 0.5).astype(np.float32)

    delta = np.abs(xray - mask)

    iou = (mask * xray).sum() / np.maximum(mask, xray).sum()
    if iou >= 0.9:
        # copy xray and mask to dst_root
        dst_xray_path = xray_path.replace(src_root, dst_root)
        dst_xray_dir = os.path.dirname(dst_xray_path)
        os.makedirs(dst_xray_dir, exist_ok=True)
        # shutil
        shutil.copy(xray_path, dst_xray_path)

        # copy image to dst_root
        image_path = xray_path.replace("xrays", "images").replace(".npz", ".png")
        dst_image_path = xray_path.replace(src_root, dst_root).replace("xrays", "images").replace(".npz", ".png")
        dst_image_dir = os.path.dirname(dst_image_path)
        os.makedirs(dst_image_dir, exist_ok=True)
        shutil.copy(image_path, dst_image_path)

        count_iou += 1
        rate = count_iou / count
        progress_bar.set_description(f"IOU: {iou:.4f}, rate: {rate:.4f}")
        

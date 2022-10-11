import os, cv2
import tifffile as tiff
from patchify import patchify
from PIL import Image
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

DATA_MAIN_PATH = "./mask"
SAVE_DATA_PATH = "./data"

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data-img-location", dest="data_main_path", required=True)
parser.add_argument("--data-save-location", dest="data_save_path", required=True)
parser.add_argument("--img-pd-path", dest="img_pd_path", required=True)
parser.add_argument("--map-pd-path", dest="map_pd_path", required=True)

args = parser.parse_args()


# DATA_MAIN_PATH = f"PA/mask"
# DATA_XML_PATH = f"PA/XML"

DATA_MAIN_PATH = args.data_main_path
SAVE_DATA_PATH = args.data_save_path
IMG_PD_PATH = args.img_pd_path
MAP_PD_PATH = args.map_pd_path

images = pd.read_pickle(IMG_PD_PATH)
mapper = pd.read_pickle(MAP_PD_PATH)

rgb2gray = eval(mapper.loc[0, "rgb2gray"])
class2rgb = eval(mapper.loc[0, "class2rgb"])
cls_id = eval(mapper.loc[0, "cls_id"])
rgb2cls = eval(mapper.loc[0, "rgb2cls"])
gray2class = eval(mapper.loc[0, "gray2class"])


def mask2onehot(m, cls):
    idx = cls_id[cls]
    return np.where(m != 0, idx, 0)


def onehot2rgb(m, cls):
    idx = cls_id[cls]

    m = m + [[[0, 0, 0]]]
    m = np.where(m != 0, class2rgb[idx], [0, 0, 0])

    return m


def do_work(palatte):
    img_name, masks_info = palatte
    classes = np.unique(masks_info["class"])

    row = masks_info[masks_info["class"] == classes[0]].iloc[0]
    img_path = os.path.join(
        DATA_MAIN_PATH, row["class"], "preview", f"{row['ImgName']}+original.jpg"
    )
    imgs = np.array(Image.open(img_path))
    imgs = np.squeeze(patchify(imgs, (300, 300, 3), step=300), axis=2).reshape(
        -1, 300, 300, 3
    )

    for idx, img in enumerate(imgs):
        np.save(os.path.join(SAVE_DATA_PATH, "imgs", f'{row["ImgName"]}_{idx:04}'), img)

    for mask_label in classes:
        row = masks_info[masks_info["class"] == mask_label].iloc[0]
        mask_path = os.path.join(
            DATA_MAIN_PATH, row["class"], "tiff", f"{row['ImgName']}.tiff"
        )

        masks = tiff.imread(mask_path)
        masks = cv2.resize(
            masks,
            dsize=(0, 0),
            fx=1 / row["img_to_mask_scale"],
            fy=1 / row["img_to_mask_scale"],
            interpolation=cv2.INTER_LINEAR_EXACT,
        )

        masks = patchify(masks, (300, 300), step=300)
        mask_onehots = mask2onehot(np.expand_dims(masks, axis=-1), mask_label).reshape(
            -1, 300, 300, 1
        )

        for idx, mask_onehot in enumerate(mask_onehots):
            mask_sparse = sparse.csr_matrix(mask_onehot.reshape(1, -1))

            sparse.save_npz(
                os.path.join(
                    SAVE_DATA_PATH, "masks", f'{row["ImgName"]}+{mask_label}_{idx:04}'
                ),
                mask_sparse,
            )


if __name__ == "__main__":
    from shutil import rmtree

    rmtree(os.path.join(SAVE_DATA_PATH, "imgs"))
    rmtree(os.path.join(SAVE_DATA_PATH, "masks"))

    os.makedirs(os.path.join(SAVE_DATA_PATH, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DATA_PATH, "masks"), exist_ok=True)

    imgsName = np.unique(images["ImgName"])
    palatte = []
    for img_name in imgsName:
        masks = images.groupby(by="ImgName").get_group(img_name)
        palatte.append((img_name, masks))

    pools = Pool(processes=8)
    for _ in tqdm(pools.imap_unordered(do_work, palatte[3:4])):
        pass

    pools.close()
    pools.join()

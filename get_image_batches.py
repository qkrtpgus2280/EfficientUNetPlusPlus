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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", dest="id", required=True)
parser.add_argument("--startpoint", dest="startpoint", required=True)
parser.add_argument("--endpoint", dest="endpoint", required=True)
parser.add_argument("--data-path", dest="data_main_path", required=True)
parser.add_argument("--save-path", dest="save_data_path", required=True)


args = parser.parse_args()

DATA_MAIN_PATH = args.data_main_path
SAVE_DATA_PATH = args.save_data_path

images = pd.read_pickle("./images.pickle")
mapper = pd.read_pickle("./mapper.pickle")

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


def do_work_move_onemask(palatte):
    img_name, masks_info = palatte
    classes = np.unique(masks_info["class"])

    row = masks_info[masks_info["class"] == classes[0]].iloc[0]
    img_path = os.path.join(
        DATA_MAIN_PATH, row["class"], "preview", f"{row['ImgName']}+original.jpg"
    )
    imgs = np.array(Image.open(img_path))

    """ 
        target image 512사이즈로 맞춰 패치화하기 위해 크기 확장 및 축소
    """
    w_size = imgs.shape[1] // 512
    h_size = imgs.shape[0] // 512

    remain_w = imgs.shape[1] % 512
    remain_h = imgs.shape[0] % 512

    target_w = w_size * 512 if remain_w / 512 < 0.5 else (w_size + 1) * 512
    target_h = h_size * 512 if remain_h / 512 < 0.5 else (h_size + 1) * 512

    imgs = cv2.resize(
        imgs, dsize=(target_w, target_h), interpolation=cv2.INTER_LANCZOS4
    )

    """
        background_rgb, background_onehot 이미지 생성
    """
    background_rgb = np.zeros((target_h, target_w, 3), dtype=np.int64)
    background_one = np.zeros((target_h, target_w, 1), dtype=np.int64)

    """
        마스크를 한 이미지에 옮기기
    """
    for idx, mask_label in enumerate(classes):
        row = masks_info[masks_info["class"] == mask_label].iloc[0]
        mask_path = os.path.join(
            DATA_MAIN_PATH, row["class"], "tiff", f"{row['ImgName']}.tiff"
        )

        masks = tiff.imread(mask_path)
        masks = cv2.resize(
            masks,
            dsize=(target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

        mask_onehots = mask2onehot(np.expand_dims(masks, axis=-1), mask_label)
        mask_onehots = mask_onehots.astype(np.int64)

        background_one += mask_onehots
        background_one = np.where(
            background_one > 6, cls_id[mask_label], background_one
        )

        background_rgb += onehot2rgb(mask_onehots, mask_label)
        background_rgb = np.where(background_rgb > 255, 255, background_rgb)

    # 패치화
    imgs = np.squeeze(patchify(imgs, (512, 512, 3), step=512), axis=2)
    imgs = imgs.reshape(-1, 512, 512, 3)

    mask_onehots = patchify(background_one, (512, 512, 1), step=512).reshape(
        -1, 512, 512, 1
    )

    for idx, img in enumerate(imgs):
        np.save(os.path.join(SAVE_DATA_PATH, "imgs", f'{row["ImgName"]}_{idx:04}'), img)

    for idx, mask_onehot in enumerate(mask_onehots):
        mask_sparse = sparse.csr_matrix(mask_onehot.reshape(1, -1))

        sparse.save_npz(
            os.path.join(SAVE_DATA_PATH, "masks", f'{row["ImgName"]}_{idx:04}'),
            mask_sparse,
        )


def do_work_move_onemask_and_save_only_not_empty(palatte):
    img_name, masks_info = palatte
    classes = np.unique(masks_info["class"])

    row = masks_info[masks_info["class"] == classes[0]].iloc[0]
    img_path = os.path.join(
        DATA_MAIN_PATH, row["class"], "preview", f"{row['ImgName']}+original.jpg"
    )
    imgs = np.array(Image.open(img_path))

    """ 
        target image 512사이즈로 맞춰 패치화하기 위해 크기 확장 및 축소
    """
    w_size = imgs.shape[1] // 512
    h_size = imgs.shape[0] // 512

    remain_w = imgs.shape[1] % 512
    remain_h = imgs.shape[0] % 512

    target_w = w_size * 512 if remain_w / 512 < 0.5 else (w_size + 1) * 512
    target_h = h_size * 512 if remain_h / 512 < 0.5 else (h_size + 1) * 512

    imgs = cv2.resize(
        imgs, dsize=(target_w, target_h), interpolation=cv2.INTER_LANCZOS4
    )

    """
        background_rgb, background_onehot 이미지 생성
    """
    background_rgb = np.zeros((target_h, target_w, 3), dtype=np.int64)
    background_one = np.zeros((target_h, target_w, 1), dtype=np.int64)

    """
        마스크를 한 이미지에 옮기기
    """
    for idx, mask_label in enumerate(classes):
        row = masks_info[masks_info["class"] == mask_label].iloc[0]
        mask_path = os.path.join(
            DATA_MAIN_PATH, row["class"], "tiff", f"{row['ImgName']}.tiff"
        )

        masks = tiff.imread(mask_path)
        masks = cv2.resize(
            masks,
            dsize=(target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

        mask_onehots = mask2onehot(np.expand_dims(masks, axis=-1), mask_label)
        mask_onehots = mask_onehots.astype(np.int64)

        background_one += mask_onehots
        background_one = np.where(
            background_one > 6, cls_id[mask_label], background_one
        )

        background_rgb += onehot2rgb(mask_onehots, mask_label)
        background_rgb = np.where(background_rgb > 255, 0, background_rgb)

    # 패치화
    imgs = np.squeeze(patchify(imgs, (512, 512, 3), step=512), axis=2)
    imgs = imgs.reshape(-1, 512, 512, 3)

    mask_onehots = patchify(background_one.squeeze(), (512, 512), step=512).reshape(
        -1, 512, 512, 1
    )

    for idx, mask_onehot in enumerate(mask_onehots):
        label_masks_local = np.transpose(np.nonzero(mask_onehot))
        if label_masks_local.size != 0:
            mask_sparse = sparse.csr_matrix(mask_onehot.reshape(1, -1))

            np.save(
                os.path.join(SAVE_DATA_PATH, "imgs", f'{row["ImgName"]}_{idx:04}'),
                imgs[idx],
            )
            sparse.save_npz(
                os.path.join(SAVE_DATA_PATH, "masks", f'{row["ImgName"]}_{idx:04}'),
                mask_sparse,
            )


def do_work_sep_mask_image_origin_way(palatte):
    """
    이미지를 512크기로 이미지를 패치로 나눔
    """
    img_name, masks_info = palatte
    classes = np.unique(masks_info["class"])

    row = masks_info[masks_info["class"] == classes[0]].iloc[0]
    img_path = os.path.join(
        DATA_MAIN_PATH, row["class"], "preview", f"{row['ImgName']}+original.jpg"
    )
    imgs = np.array(Image.open(img_path))

    """ 
        target image 512사이즈로 맞춰 패치화하기 위해 크기 확장 및 축소
    """
    w_size = imgs.shape[1] // 512
    h_size = imgs.shape[0] // 512

    remain_w = imgs.shape[1] % 512
    remain_h = imgs.shape[0] % 512

    target_w = w_size * 512 if remain_w / 512 < 0.5 else (w_size + 1) * 512
    target_h = h_size * 512 if remain_h / 512 < 0.5 else (h_size + 1) * 512

    imgs = cv2.resize(
        imgs, dsize=(target_w, target_h), interpolation=cv2.INTER_LANCZOS4
    )

    imgs = np.squeeze(patchify(imgs, (512, 512, 3), step=512), axis=2).reshape(
        -1, 512, 512, 3
    )

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

        masks = patchify(masks, (512, 512), step=512)
        mask_onehots = mask2onehot(np.expand_dims(masks, axis=-1), mask_label).reshape(
            -1, 512, 512, 1
        )

        for idx, mask_onehot in enumerate(mask_onehots):
            mask_sparse = sparse.csr_matrix(mask_onehot.reshape(1, -1))

            sparse.save_npz(
                os.path.join(
                    SAVE_DATA_PATH, "masks", f'{row["ImgName"]}+{mask_label}_{idx:04}'
                ),
                mask_sparse,
            )

    for idx, img in enumerate(imgs):
        np.save(os.path.join(SAVE_DATA_PATH, "imgs", f'{row["ImgName"]}_{idx:04}'), img)


if __name__ == "__main__":
    from shutil import rmtree

    try:
        rmtree(os.path.join(SAVE_DATA_PATH, "imgs"))
        rmtree(os.path.join(SAVE_DATA_PATH, "masks"))
    except:
        pass

    os.makedirs(os.path.join(SAVE_DATA_PATH, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DATA_PATH, "masks"), exist_ok=True)

    imgsName = np.unique(images["ImgName"])
    palatte = []
    for img_name in imgsName:
        masks = images.groupby(by="ImgName").get_group(img_name)
        palatte.append((img_name, masks))

    id = int(args.id)

    pools = Pool(processes=8)
    if id == 0:
        for _ in tqdm(
            pools.imap_unordered(
                do_work_move_onemask, palatte[int(args.startpoint) : int(args.endpoint)]
            )
        ):
            pass
    elif id == 1:
        for _ in tqdm(
            pools.imap_unordered(
                do_work_move_onemask_and_save_only_not_empty,
                palatte[int(args.startpoint) : int(args.endpoint)],
            )
        ):
            pass
    else:
        for _ in tqdm(
            pools.imap_unordered(
                do_work_sep_mask_image_origin_way,
                palatte[int(args.startpoint) : int(args.endpoint)],
            )
        ):
            pass

    pools.close()
    pools.join()

import xmltodict, os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import tifffile as tiff
import pickle
import cv2
import numpy as np
import pandas as pd
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data-img-location", dest="data_main_path", required=True)
parser.add_argument("--data-xml-location", dest="data_xml_path", required=True)

args = parser.parse_args()


# DATA_MAIN_PATH = f"PA/mask"
# DATA_XML_PATH = f"PA/XML"

DATA_MAIN_PATH = args.data_main_path
DATA_XML_PATH = args.data_xml_path

xmls = [xml[:-4] for xml in os.listdir(DATA_XML_PATH)]
no_files = []
idx = 0
images = pd.DataFrame()

for xml in tqdm(xmls):
    target = xml
    xml_string = open(os.path.join(DATA_XML_PATH, f"{target}.xml")).read()
    result = xmltodict.parse(xml_string)
    body = result["object-stream"]["Annotations"]
    image = body["@image"]
    annotations = body["Annotation"]
    comments = body["Comment"]

    if isinstance(annotations, list) == True:
        color_ = []
        class_ = []
        type_ = []
        coords_ = []
        for annotation in annotations:
            color_.append(annotation["@color"])
            class_.append(annotation["@class"])
            type_.append(annotation["@type"])
            coords_.append(
                [
                    [float(coord["@x"]), float(coord["@y"])]
                    for coord in annotation["Coordinates"]["Coordinate"]
                ]
            )

    elif isinstance(annotations, dict) == True:
        color_ = [annotations["@color"]]
        class_ = [annotations["@class"]]
        type_ = [annotations["@type"]]
        coords_ = [
            [
                [float(coord["@x"]), float(coord["@y"])]
                for coord in annotations["Coordinates"]["Coordinate"]
            ]
        ]

    t = np.array(class_)
    first_file_path = t[np.where(t != "")][0]

    origin_img_path = (
        DATA_MAIN_PATH,
        first_file_path,
        "preview",
        target + "+original.jpg",
    )
    origin_mask_paths = (DATA_MAIN_PATH, first_file_path, "tiff", target + ".tiff")
    image_path = os.path.join(*origin_img_path)
    mask_path = os.path.join(*origin_mask_paths)

    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = tiff.imread(mask_path)

        for color, cls, typ, anno in zip(color_, class_, type_, coords_):
            if cls == "":
                continue

            preproc_paths = (DATA_MAIN_PATH, "usable", "infos", target + ".dump")
            coords_paths = (DATA_MAIN_PATH, cls, "dump", target + ".dump")
            preproc_path = os.path.join(*preproc_paths)
            coords_path = os.path.join(*coords_paths)

            coords = pickle.load(open(coords_path, "rb"))
            preproc = pickle.load(open(preproc_path, "rb"))

            anno = list(map((lambda x: (int(x[0]), int(x[1]))), anno))
            anno = np.array(anno, dtype=np.int32)

            images.loc[idx, "ImgName"] = target
            images.loc[idx, "ImageSize"] = str(img.shape)
            images.loc[idx, "MaskSize"] = str(mask.shape)
            images.loc[idx, "class"] = cls
            images.loc[idx, "color"] = color
            images.loc[idx, "preprocessing"] = str(preproc)
            images.loc[idx, "keypoints"] = str(anno.tolist())
            idx += 1

    except FileNotFoundError:
        no_files.append(target)

mask_list = [list(eval(masksz)) for masksz in images["MaskSize"]]
mask_list = np.array(mask_list, np.int32)

img_size = [list(eval(imgsz))[:2] for imgsz in images["ImageSize"]]
img_size = np.array(img_size, np.int32)

scales = (mask_list / img_size).astype(np.int32)[:, 0]

images["img_to_mask_scale"] = scales


cls_id = {v: k + 1 for k, v in enumerate(np.unique(images["class"]))}
cls_id["background"] = 0

class2rgb = {
    k + 1: v
    for k, v in enumerate(
        [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(6)
        ]
    )
}
class2rgb[0] = (0, 0, 0)
rgb2gray = {
    cv2.cvtColor(np.array([[v]], np.uint8), cv2.COLOR_RGB2GRAY)[0][0]: v
    for v in class2rgb.values()
}
rgb2cls = {v: k for k, v in class2rgb.items()}
gray2class = {k: rgb2cls[v] for k, v in rgb2gray.items()}

mapper = pd.DataFrame()

mapper.loc[0, "rgb2gray"] = str(rgb2gray)
mapper.loc[0, "class2rgb"] = str(class2rgb)
mapper.loc[0, "cls_id"] = str(cls_id)
mapper.loc[0, "rgb2cls"] = str(rgb2cls)
mapper.loc[0, "gray2class"] = str(gray2class)

from pprint import pprint

pprint(no_files)

with open("images.pickle", "wb") as f:
    pickle.dump(images, f, protocol=3)

with open("mapper.pickle", "wb") as f:
    pickle.dump(mapper, f, protocol=3)

import argparse
import logging
import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import cv2

from PIL import Image
from torchvision import transforms
from utils.dataset import CODIPAISegmentationDataset
from sklearn.metrics import *
from tqdm import tqdm

import segmentation_models_pytorch.segmentation_models_pytorch as smp

from torch.backends import cudnn


def predict_img(net, dataset_class, full_img, device, scale_factor=1, n_classes=3):
    net.eval()

    img = torch.from_numpy(dataset_class.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if n_classes > 1:
            probs = torch.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor(),
            ]
        )

        full_mask = tf(probs.cpu())

    if n_classes > 1:
        return dataset_class.one_hot2mask(full_mask)
    else:
        return full_mask > 0.5


def get_args():
    parser = argparse.ArgumentParser(
        description="Predict masks from input images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="MODEL.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
        required=True,
    )
    parser.add_argument(
        "--mapper",
        "-ma",
        metavar="pd.DataFrame",
        help="해당 데이터셋의 레이블에 따른 RGB, 클래스 정보등이 담긴 데이터프레임, get_images_info.py로 부터 얻음",
        required=True
    )
    parser.add_argument(
        "--img-path",
        "-i",
        dest="img_path",
        nargs="+",
        help="입력 이미지가 위치한 폴더 경로",
        required=True,
    )
    parser.add_argument(
        "--predict-mask-path",
        "-o",
        dest='mask_path',
        nargs="+",
        help="마스크 예측 파일을 저장할 경로",
        required=True,
    )
    parser.add_argument(
        "-enc",
        "--encoder",
        metavar="ENC",
        type=str,
        default="timm-efficientnet-b0",
        help="Encoder to be used",
        dest="encoder",
    )
    return parser.parse_args()


def calc_metrics(name, in_path, net, dataset_class, scale, device):
    tot_test_precision = 0
    tot_test_recall = 0
    tot_test_roc = 0
    tot_test_iou = 0
    class_1_tot = 0
    class_2_tot = 0
    class_3_tot = 0

    for i, fn in tqdm(enumerate(os.listdir(in_path)), total=len(os.listdir(in_path))):
        image = Image.open(f"{in_path}/{name}/{fn}").convert(mode="RGB")
        img_cv = cv2.imread(f"{in_path}/{name}/{fn}", cv2.IMREAD_COLOR)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        mask = predict_img(
            net=net,
            dataset_class=dataset_class,
            full_img=image,
            scale_factor=scale,
            device=device,
        )
        pred_mask_1d = mask.reshape(
            -1,
        )
        pred_counts = np.bincount(pred_mask_1d)

        mask_ground_truth = Image.open(
            os.path.join(in_path, name, fn.replace(".JPG", ".png"))
        )
        mask_gt_onehot = dataset_class.mask_img2class_mask(mask_ground_truth, 1).astype(
            np.int64
        )
        gt_mask_1d = mask_gt_onehot.reshape(
            -1,
        )
        gt_counts = np.bincount(gt_mask_1d)

        """ calc mIOU"""

        conf_matrix = multilabel_confusion_matrix(
            gt_mask_1d, pred_mask_1d, labels=[0, 1, 2, 3]
        )

        test_data_precision = precision_score(
            gt_mask_1d, pred_mask_1d, average="weighted"
        )
        test_data_recall = recall_score(gt_mask_1d, pred_mask_1d, average="weighted")

        class_1_pred_mask = pred_mask_1d.copy()
        class_1_ground_mask = gt_mask_1d.copy()
        class_2_pred_mask = pred_mask_1d.copy()
        class_2_ground_mask = gt_mask_1d.copy()
        class_3_pred_mask = pred_mask_1d.copy()
        class_3_ground_mask = gt_mask_1d.copy()

        class_1_pred_mask[np.where(class_1_pred_mask == 1)] = 1
        class_1_pred_mask[np.where(class_1_pred_mask == 2)] = 0
        class_1_pred_mask[np.where(class_1_pred_mask == 3)] = 0
        class_1_ground_mask[np.where(class_1_ground_mask == 1)] = 1
        class_1_ground_mask[np.where(class_1_ground_mask == 2)] = 0
        class_1_ground_mask[np.where(class_1_ground_mask == 3)] = 0

        fpr_1, tpr_1, _ = roc_curve(class_1_pred_mask, class_1_ground_mask)
        class_1_roc_score = auc(fpr_1, tpr_1)

        class_2_pred_mask[np.where(class_2_pred_mask == 1)] = 0
        class_2_pred_mask[np.where(class_2_pred_mask == 2)] = 1
        class_2_pred_mask[np.where(class_2_pred_mask == 3)] = 0
        class_2_ground_mask[np.where(class_2_ground_mask == 1)] = 0
        class_2_ground_mask[np.where(class_2_ground_mask == 2)] = 1
        class_2_ground_mask[np.where(class_2_ground_mask == 3)] = 0

        fpr_2, tpr_2, _ = roc_curve(class_2_pred_mask, class_2_ground_mask)
        class_2_roc_score = auc(fpr_2, tpr_2)

        class_3_pred_mask[np.where(class_3_pred_mask == 1)] = 0
        class_3_pred_mask[np.where(class_3_pred_mask == 2)] = 0
        class_3_pred_mask[np.where(class_3_pred_mask == 3)] = 1
        class_3_ground_mask[np.where(class_3_ground_mask == 1)] = 0
        class_3_ground_mask[np.where(class_3_ground_mask == 2)] = 0
        class_3_ground_mask[np.where(class_3_ground_mask == 3)] = 1

        fpr_3, tpr_3, _ = roc_curve(class_3_pred_mask, class_3_ground_mask)
        class_3_roc_score = auc(fpr_3, tpr_3)

        test_data_iou = jaccard_score(gt_mask_1d, pred_mask_1d, average="weighted")
        class_1_tot += class_1_roc_score

        class_2_tot += class_2_roc_score
        class_3_tot += class_3_roc_score
        tot_test_precision += test_data_precision
        tot_test_recall += test_data_recall
        tot_test_iou += test_data_iou

    tot_test_precision /= len(os.listdir(in_path))
    tot_test_recall /= len(os.listdir(in_path))
    tot_test_roc /= len(os.listdir(in_path))
    tot_test_iou /= len(os.listdir(in_path))
    class_1_tot /= len(os.listdir(in_path))
    class_2_tot /= len(os.listdir(in_path))
    class_3_tot /= len(os.listdir(in_path))

    return f"\n\n{name}\ntotal_{name}_precision  : {tot_test_precision}\ntotal_{name}_recall     : {tot_test_recall}\ntotal_{name}_f1         : {tot_test_roc}\ntotal_{name}_iou        : {tot_test_iou}\nclass_1_auc:  {class_1_tot}\nclass_2_auc:     {class_2_tot}\nclass_3_auc:     {class_3_tot}"


# python predict.py -m checkpoints/CP_epoch5.pth --img-path ../CODIPAI/PA/data/test/imgs -- output ../CODIPAI/PA/data/test/predicts

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    in_img_path = args.img_path[0]
    out_mask_path = args.mask_path[0]

    mapper = pickle.load(open(args.mapper, "rb"))
    
    os.makedirs(out_mask_path, exist_ok=True)

    gray2rgb = eval(mapper.loc[0, "rgb2gray"])
    class2rgb = eval(mapper.loc[0, "class2rgb"])
    cls_id = eval(mapper.loc[0, "cls_id"])
    rgb2cls = eval(mapper.loc[0, "rgb2cls"])
    gray2class = eval(mapper.loc[0, "gray2class"])
    n_classes = len(gray2rgb)

    dataset_class = CODIPAISegmentationDataset
    dataset_class.gray2rgb_mapping = gray2rgb
    dataset_class.class2rgb_mapping = class2rgb
    dataset_class.gray2class_mapping = gray2class
    dataset_class.rgb2class_mapping = rgb2cls
    dataset_class.cls_id = cls_id
    dataset_class.n_classes = n_classes

    net = smp.EfficientUnetPlusPlus(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes,
    )
    net = nn.DataParallel(net)

    logging.info("Loading model {}".format(args.model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    net.to(device=device)
    # faster convolutions, but more memory
    cudnn.benchmark = True
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    if device == "cuda":
        score = torch.FloatTensor(1).cuda().zero_()
        weighted_score = torch.FloatTensor(1).cuda().zero_()
    else:
        score = torch.FloatTensor(1).zero_()
        weighted_score = torch.FloatTensor(1).zero_()

    for i, fn in enumerate(os.listdir(in_img_path)):
        img = Image.fromarray(np.load(os.path.join(in_img_path, fn)).astype(np.uint8)).convert(mode="RGB")
        mask = predict_img(
            net=net,
            dataset_class=dataset_class,
            full_img=img,
            scale_factor=1,
            device=device,
        )
        result = dataset_class.mask2image(mask)
        result.save(os.path.join(out_mask_path, f"{fn.replace('.npy', '.jpg')}"))

        logging.info("Mask saved to {}".format(f"{fn}.jpg"))

    # in_path = f"./data/detect_and_segmentation/dataset_folds/dataset_fold_0/images"
    # print(calc_metrics("train", in_path, net, dataset_class, 1, device))

    # in_path = f"./data/detect_and_segmentation/dataset_folds/dataset_fold_0/images"
    # print(calc_metrics("valid", in_path, net, dataset_class, 1, device))

    # in_path = f"./data/detect_and_segmentation/dataset_folds/dataset_fold_0/images"
    # print(calc_metrics("test", in_path, net, dataset_class, 1, device))

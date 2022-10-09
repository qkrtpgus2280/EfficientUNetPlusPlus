from os.path import splitext, join
from os import listdir
from typing import Dict, List
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
import logging
from PIL import Image
import torch.nn.functional as F
from augment import *

r"""
Defines the `BasicSegmentationDataset` and `CoronaryArterySegmentationDatasets`, which extend the `Dataset` and `BasicSegmentationDataset` \ 
classes, respectively. Each class defines the specific methods needed for data processing and a method :func:`__getitem__` to return samples.
"""


class BasicSegmentationDataset(Dataset):
    r"""
    Implements a basic dataset for segmentation tasks, with methods for image and mask scaling and normalization. \
    The filenames of the segmentation ground truths must be equal to the filenames of the images to be segmented, \
    except for a possible suffix.

    Args:
        imgs_dir (str): path to the directory containing the images to be segmented.
        masks_dir (str): path to the directory containing the segmentation ground truths.
        scale (float, optional): image scale, between 0 and 1, to be used in the segmentation.
        mask_suffix (str, optional): suffix to be added to an image's filename to obtain its 
            ground truth filename.
    """

    def __init__(
        self, imgs_dir: str, masks_dir: str, scale: float = 1, mask_suffix: str = ""
    ):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        self.ids = [
            splitext(file)[0]
            for file in listdir(self.masks_dir)
            if not file.startswith(".")
        ]
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self) -> int:
        r"""
        Returns the size of the dataset.
        """
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img: Image, scale: float) -> Image:
        r"""
        Preprocesses an `Image`, rescaling it and returning it as a NumPy array in
        the CHW format.

        Args:
            pil_imgs (Image): object of class `Image` to be preprocessed.
            scale (float): image scale, between 0 and 1.
        """
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i) -> Dict[List[torch.FloatTensor], List[torch.FloatTensor]]:
        r"""
        Returns two tensors: an image and the corresponding mask.
        """
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + ".*")
        img_file = glob(self.imgs_dir + idx + ".*")

        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {idx}: {mask_file}"
        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {idx}: {img_file}"
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert (
            img.size == mask.size
        ), f"Image and mask {idx} should be the same size, but are {img.size} and {mask.size}"

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            "image": [torch.from_numpy(img).type(torch.FloatTensor)],
            "mask": [torch.from_numpy(mask).type(torch.FloatTensor)],
        }


class CODIPAISegmentationDataset(BasicSegmentationDataset):
    r"""
    Implements a dataset for the Coronary Artery Segmentation task, with mappings between grayscale image values to class \
    indices, grayscale image values to RGB image values, and methods for the necessary conversions.
    Args:
        imgs_dir (str): path to the directory containing the images to be segmented.
        masks_dir (str): path to the directory containing the segmentation ground truths.
        scale (float, optional): image scale, between 0 and 1, to be used in the segmentation.
        mask_suffix (str, optional): suffix to be added to an image's filename to obtain its ground truth filename.
        augmentation_ratio (int, optional): number of augmentations to generate per image.
        aug_policy (str, optional): data augmentation policy.
    """

    # Maps maks grayscale value to mask class index
    gray2class_mapping = {}

    # Maps mask grayscale value to mask RGB value
    gray2rgb_mapping = {}
    rgb2class_mapping = {}
    class2rgb_mapping = {}
    cls_id = {}

    # Total number of classes, including the background class
    n_classes = 0

    def __init__(
        self,
        imgs_dir: str,
        masks_dir: str,
        scale: float = 1,
        mask_suffix: str = "",
        augmentation_ratio: int = 0,
        aug_policy: str = "coronary",
        gray2class_mapping: dict = {},
        gray2rgb_mapping: dict = {},
        rgb2class_mapping: dict = {},
        cls_id: dict = {},
    ):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix)
        self.augmentation_ratio = augmentation_ratio
        self.policy = aug_policy

        self.gray2class_mapping = gray2class_mapping
        self.gray2rgb_mapping = gray2rgb_mapping
        self.rgb2class_mapping = rgb2class_mapping
        self.cls_id = cls_id
        self.class2rgb = {v: k for k, v in rgb2class_mapping.items()}
        self.n_classes = len(self.gray2rgb_mapping)

    @classmethod
    def mask_img2class_mask(cls, pil_mask: Image, scale: float) -> np.array:
        r"""
        Preprocesses a grayscale `Image` containing a segmentation mask, rescaling it, converting its grayscale values \
        to class indices and returning it as a NumPy array in the CHW format.
        Args:
            pil_imgs (Image): object of class `Image` to be preprocessed.
            scale (float): image scale, between 0 and 1.
        """
        w, h = pil_mask.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_mask = pil_mask.resize((newW, newH))

        if pil_mask.mode != "L":
            pil_mask = pil_mask.convert(mode="L")
        mask_nd = np.array(pil_mask)

        if len(mask_nd.shape) == 2:
            mask_nd = np.expand_dims(mask_nd, axis=2)

        # HWC to CHW
        mask = mask_nd.transpose((2, 0, 1))
        mask = mask / 255
        mask = mask * 2
        mask = np.around(mask)

        return mask

    @classmethod
    def mask2one_hot(
        cls, mask_tensor: torch.LongTensor, output_shape: str = "NHWC"
    ) -> torch.Tensor:
        r"""
        Returns the received `FloatTensor` in the N1HW shape to a one hot encoded `LongTensor` in the NHWC shape.\
            Can return in NCHW shape is specified.
        Args:
            mask_tensor (LongTensor): N1HW LongTensor to be one-hot encoded.
            output_shape (str): NHWC or NCHW.
        """
        assert (
            output_shape == "NHWC" or output_shape == "NCHW"
        ), "Invalid output shape specified"

        # Assuming tensor in NCHW = N1HW shape
        if output_shape == "NHWC":
            return F.one_hot(mask_tensor, cls.n_classes).squeeze(1)
        # Assuming tensor in N1HW shape
        elif output_shape == "NCHW":
            return torch.transpose(
                torch.transpose(F.one_hot(mask_tensor, cls.n_classes), 2, 3), 1, 2
            )

    @classmethod
    def one_hot2mask(
        cls, one_hot_mask: torch.FloatTensor, shape: str = "CHW"
    ) -> np.array:
        r"""
        Returns the one-channel mask (1HW) corresponding to the CHW one-hot encoded one.
        """
        # Assuming tensor in CHW shape
        if shape == "CHW":
            return np.argmax(one_hot_mask.detach().numpy(), axis=0)
        elif shape == "NCHW":
            return np.argmax(one_hot_mask.detach().numpy(), axis=1)
        return np.argmax(one_hot_mask.detach().numpy(), axis=0)

    @classmethod
    def class2gray(cls, mask: np.array) -> np.array:
        r"""
        Replaces the class labels in a numpy array represented mask by their grayscale values, according to `gray2class_mapping`.
        """
        assert (
            len(cls.gray2class_mapping) == cls.n_classes
        ), f"Number of class mappings - {len(cls.gray2class_mapping)} - should be the same as the number of classes - {cls.n_classes}"
        for color, label in cls.gray2class_mapping.items():
            mask[mask == label] = color
        return mask

    @classmethod
    def gray2rgb(cls, img: Image) -> Image:
        r"""
        Converts a grayscale image into an RGB one, according to gray2rgb_mapping.
        """
        rgb_img = Image.new("RGB", img.size)
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                rgb_img.putpixel((x, y), cls.gray2rgb_mapping[img.getpixel((x, y))])
        return rgb_img

    @classmethod
    def mask2image(cls, mask: np.array) -> Image:
        r"""
        Converts a one-channel mask (1HW) with class indices into an RGB image, according to gray2class_mapping and gray2rgb_mapping.
        """
        return cls.gray2rgb(Image.fromarray(cls.class2gray(mask).astype(np.uint8)))

    def augment(self, image, mask, label):
        """
        Returns a list with the original image and mask and augmented versions of them.
        The number of augmented images and masks is equal to the specified augmentation_ratio.
        The policy is chosen by the policy argument
        """
        tf_imgs = []
        tf_masks = []

        image = self.preprocess(image, self.scale)
        # mask = self.mask_img2class_mask(mask, self.scale)

        self.onehot2rgb(mask, label)
        mask = mask.transpose((2, 0, 1))
        mask = mask / 255
        mask = mask * 2
        mask = np.around(mask)

        image = torch.from_numpy(image).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        tf_imgs.insert(0, image)
        tf_masks.insert(0, mask)

        return (tf_imgs, tf_masks)

    def onehot2rgb(self, m, label):
        idx = self.cls_id[label]

        m = m + [[[0, 0, 0]]]
        m = np.where(m != 0, self.class2rgb[idx], [0, 0, 0])

        return m

    def __getitem__(self, i) -> Dict[List[torch.FloatTensor], List[torch.FloatTensor]]:
        r"""
        Returns two tensors: an image, of shape 1HW, and the corresponding mask, of shape CHW.
        이미지는 중복되기 때문에 마스크 기준으로 loading..
        """
        idx = self.ids[i]
        mask_file = glob(join(self.masks_dir, idx + "*"))
        file_name = idx.split("+")[0]
        patch = mask_file[0].split("+")[-1].replace(".npz", "").split("_")[-1]
        label = "_".join(
            mask_file[0].split("+")[-1].replace(".npz", "").split("_")[:-1]
        )
        img_file = glob(join(self.imgs_dir, file_name + f"*{patch}*"))

        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {idx}: {mask_file}"
        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {idx}: {img_file}"

        image = np.load(img_file[0], allow_pickle=True)
        image = Image.fromarray(image.astype(np.uint8))

        mask = sparse.load_npz(mask_file[0]).toarray()
        mask = mask.reshape(300, 300, 1)

        image, mask = self.augment(image, mask, label)

        return {"image": image, "mask": mask}


if __name__ == "__main__":
    import pandas as pd

    images = pd.read_pickle("../../images.pickle")
    mapper = pd.read_pickle("../../mapper.pickle")

    rgb2gray = eval(mapper.loc[0, "rgb2gray"])
    class2rgb = eval(mapper.loc[0, "class2rgb"])
    cls_id = eval(mapper.loc[0, "cls_id"])
    rgb2cls = eval(mapper.loc[0, "rgb2cls"])
    gray2class = eval(mapper.loc[0, "gray2class"])

    ddd = CODIPAISegmentationDataset(
        "../../data/imgs",
        "../../data/masks",
        1,
        gray2class_mapping=gray2class,
        gray2rgb_mapping=rgb2gray,
        rgb2class_mapping=rgb2cls,
        cls_id=cls_id,
    )
    train_loader = DataLoader(
        ddd, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )
    print(next(iter(train_loader)))

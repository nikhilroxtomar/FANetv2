import os
import json
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
from utils import create_dir

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        h = (y2 - y1)
        w = (x2 - x1)

        if (h * w) > 1000:
            bboxes.append([x1, y1, x2, y2])

    return bboxes

def get_text_prompt(name, image, label, save_path, save=True):
    """ Num to words """
    num_dict = {}
    num_dict[0] = "zero"
    num_dict[1] = "one"
    num_dict[2] = "two"
    num_dict[3] = "three"
    num_dict[4] = "four"
    num_dict[5] = "five"
    num_dict[6] = "six"
    num_dict[7] = "seven"
    num_dict[8] = "eight"
    num_dict[9] = "nine"
    num_dict[10] = "ten"

    """ Generating prompt: How many """
    text_prompt = "a colorectal image with "

    bboxes = mask_to_bbox(label)
    num_boxes = len(bboxes)

    try:
        text_prompt += f"{num_dict[num_boxes]}"
    except KeyError as e:
        text_prompt += f"many"

    """ Generating prompt: Sizes """
    size_dict = {}
    size_dict[0] = "small"
    size_dict[1] = "medium"
    size_dict[2] = "large"

    prev_size = None
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        h = (y2 - y1)
        w = (x2 - x1)
        area = (h * w) / (label.shape[0] * label.shape[1])

        if area < 0.10:
            polyp_size = 0
        elif area >= 0.10 and area < 0.30:
            polyp_size = 1
        elif area >= 0.30:
            polyp_size = 2

        """ Save Image """
        if save == True:
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            image = cv2.putText(image, size_dict[polyp_size], (x1+10, y1+10), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 2, cv2.LINE_AA)

        if prev_size == None:
            text_prompt += f" {size_dict[polyp_size]}"
            prev_size = size_dict[polyp_size]

        else:
            if prev_size == size_dict[polyp_size]:
                pass
            else:
                text_prompt += f", {size_dict[polyp_size]}"

            prev_size = size_dict[polyp_size]

    if num_boxes <= 1:
        text_prompt += " sized polyp"
    else:
        text_prompt += " sized polyps"

    if save == True:
        cv2.imwrite(f"{save_path}/{name}-{num_boxes}.jpg", image)


def get_classification_label(label):
    """ Mask to bboxes """
    bboxes = mask_to_bbox(label)

    """ Classification: one or many """
    num_polyps = 0 if len(bboxes) == 1 else 1

    """ Classification: size -> small, medium and large """
    small, medium, large, = 0, 0, 0
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        h = (y2 - y1)
        w = (x2 - x1)
        area = (h * w) / (label.shape[0] * label.shape[1])

        if area < 0.10:
            small = 1
        elif area >= 0.10 and area < 0.30:
            medium = 1
        elif area >= 0.30:
            large = 1

    return [num_polyps, small, medium, large]


if __name__ == "__main__":
    """ Dataset path """
    dataset_path = "../ML_DATASET/bkai-igh-neopolyp"
    images = sorted(glob(os.path.join(dataset_path, "train", "*.jpeg")))
    labels = sorted(glob(os.path.join(dataset_path, "train_gt", "*.jpeg")))
    print(f"Images: {len(images)} - Labels: {len(labels)}")

    """ Save Path """
    save_path = "prompt"
    create_dir(save_path)

    """ Loop over dataset """
    for x, y in tqdm(zip(images, labels), total=len(images)):
        name = x.split("/")[-1].split(".")[0]

        image = cv2.imread(image, cv2.IMREAD_COLOR)
        label = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        label[label > 0] = 1
        label = label * 255

        # get_text_prompt(name, x, label, save_path)
        [num_polyps], [small, medium, large] = get_classification_label(label)
        print(num_polyps, small, medium, large)

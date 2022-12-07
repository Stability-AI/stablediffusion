import json
import numpy as np

from PIL import Image
from typing import List, Tuple


class Point:
    def __init__(
        self,
        x: int,
        y: int,
    ):
        self.x: int = x
        self.y: int = y

    def __repr__(self) -> str:
        return f"Point ({self.x}, {self.y})"



class BBox:
    def __init__(
        self,
        x1,
        y1,
        x2,
        y2,
    ):
        self.x1 = int(min(x1, x2))
        self.y1 = int(min(y1, y2))
        self.x2 = int(max(x1, x2))
        self.y2 = int(max(y1, y2))

    @property
    def h(self):
        return abs(self.y2 - self.y1)

    @property
    def w(self):
        return abs(self.x2 - self.x1)

    @property
    def center(self) -> Point:
        return Point(int(self.x1 + self.w / 2), int(self.y1 + self.h / 2))

    def add_padding(self, size: int, max_x: int, max_y: int):
        self.x1 = max(0, self.x1 - size)
        self.y1 = max(0, self.y1 - size)
        self.x2 = min(max_x, self.x2 + size)
        self.y2 = min(max_y, self.y2 + size)

class LabeledImage:
    def __init__(
        self,
        name,
        height=None,
        width=None,
        bbox_list: List[BBox] = None,
    ):
        self.name = name
        self.bbox_list: List[BBox] = bbox_list if bbox_list is not None else list()
        self.height = height
        self.width = width



def open_coco(source_coco_path: str, sort_by_name: bool = True) -> List[LabeledImage]:
    """
    Opens annotation in coco-format and returns list of LabeledImage instances
    :param source_coco_path: Path to json file
    :param sort_by_name: If True - LabeledImages are sorted in alphabetical oder by image name
    :return: List of LabeledImage instances
    """
    with open(source_coco_path, "r") as jfile:
        coco_ann = json.load(jfile)

    coco_images = coco_ann["images"]
    coco_categories = coco_ann.get("categories", None)
    coco_labels = coco_ann["annotations"]

    image_dict = dict()  # images from coco
    category_dict = dict()  # categories from coco

    labeled_image_list: List[LabeledImage] = list()  # list with LabeledImage instances
    labeled_image_dict = dict()  # dict to collect data from coco

    # images
    for image_info in coco_images:
        image_dict[image_info["id"]] = image_info["file_name"]

        labeled_image_dict[image_info["file_name"]] = LabeledImage(
            name=image_info["file_name"],
            height=image_info.get("height", None),
            width=image_info.get("width", None),
        )

    # categories
    if coco_categories is not None:

        if isinstance(coco_categories[0], list):
            coco_categories = coco_categories[0]

        for category_info in coco_categories:
            category_name = category_info["name"]

            category_dict[category_info["id"]] = category_name
    else:
        all_labels = set([ann["category_id"] for ann in coco_labels])
        category_dict = {label: label for label in all_labels}

    # labels
    for i, label_info in enumerate(coco_labels):
        x1 = label_info["bbox"][0]
        y1 = label_info["bbox"][1]
        w = label_info["bbox"][2]
        h = label_info["bbox"][3]

        image_name = image_dict[label_info["image_id"]]

        labeled_image_dict[image_name].bbox_list.append(
            BBox(x1, y1, x1 + w, y1 + h)
        )

    # converting image dict to list
    for image_name, labeled_image in labeled_image_dict.items():
        labeled_image_list.append(labeled_image)

    # sort by image name
    if sort_by_name:
        labeled_image_list.sort(key=lambda x: x.name)

    return labeled_image_list


def insert_image(
    src_image: Image, 
    img_to_insert: Image, 
    x: int, 
    y: int, 
) -> Tuple[np.ndarray, np.ndarray]:
    src_image_mat = np.array(src_image)
    img_to_insert_mat = np.array(img_to_insert)
    h, w, c = img_to_insert_mat.shape
    src_image_mat[y:y+h, x:x+w] = img_to_insert_mat
    src_image = Image.fromarray(src_image_mat)
    return src_image


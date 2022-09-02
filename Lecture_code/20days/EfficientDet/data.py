import os
import re
from glob import glob

WNIDS = "./data/tiny-imagenet-200/wnids.txt"
WORDS = "./data/tiny-imagenet-200/words.txt"
TRAIN_DIR = "./data/tiny-imagenet-200/train/"


# INFO: [image, (label, bbox)]
def search_class(fd=None, match=""):
    fd.seek(0)
    for line in fd:
        if re.search(match, line):
            class_name = line.strip().split("\t")[1]
            return class_name


# INFO: {"n02795169": "cat"}
def class_name():
    # {"wnids": "class_name"}
    wnids = open(WNIDS, "r")
    words = open(WORDS, "r")

    class_names = []
    while True:
        wnid = wnids.readline().strip()
        class_dict = {wnid: search_class(words, wnid)}
        if not wnid:
            break
        class_names.append(class_dict)

    wnids.close()
    words.close()
    return class_names


def _bbox(fd=None) -> list:
    bboxes = []
    for line in fd:
        line = line.strip()
        fname, xmin, ymin, xmax, ymax = line.split("\t")
        bbox_dict = {
            "fname": fname,
            "bbox": (int(xmin), int(ymin), int(xmax), int(ymax)),
        }
        bboxes.append(bbox_dict)
    return bboxes


# INFO: {"fname": "n02795169_210.JPEG", "bbox": (0, 0, 1, 1)}
# TODO: -> {"fname": "n02795169_210.JPEG", "bbox": (0, 0, 1, 1), "image": img}
def bboxes():
    bbox_texts = glob(TRAIN_DIR + "*/*_boxes.txt")

    bbox_annotations = []
    for bbox_txt in bbox_texts:
        fd = open(bbox_txt, "r")
        bbox_annotations.extend(_bbox(fd))
        fd.close()
    return bbox_annotations


def _train():
    bbox = bboxes()
    cls_name = class_name()

    # INFO: [image, (label, bbox)]
    # INFO: {"fname": "n02795169_210.JPEG", "bbox": (0, 0, 1, 1)}
    # INFO: -> {"fname": "n02795169_210.JPEG", "bbox": (0, 0, 1, 1), "class": "cat"}
    for b in bbox:
        key = b["fname"].split("_")[0]
        for name_dict in cls_name:
            if key in name_dict:
                b.update({"class": name_dict.get(key)})
        print(b)
        break


if __name__ == "__main__":
    _train()

"""voc_custom dataset."""
import os
import xml.etree.ElementTree as ET
from PIL import Image
from collections import namedtuple

import tensorflow_datasets as tfds

_DESCRIPTION = """
CUSTOM
This dataset contains the data from the PASCAL Visual Object Classes Challenge 2012, a.k.a. VOC2012.
A total of 11540 images are included in this dataset, where each image contains a set of objects, out of 20 different classes, making a total of 27450 annotated objects.
"""

_TRAIN_TXT = "ImageSets/Segmentation/train.txt"
_VAL_TXT = "ImageSets/Segmentation/val.txt"
_IMG_DIR = "JPEGImages"
_SEG_IMG_DIR = "SegmentationClass"
_ANNOTATION_DIR = "Annotations"
_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'


class VocCustom(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for voc_custom dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image':
                tfds.features.Image(shape=(None, None, 3)),
                'segmentation_image':
                tfds.features.Image(shape=(None, None, 3)),
                'objects':
                tfds.features.Sequence({
                    'bbox':
                    tfds.features.BBox(shape=(4, )),
                    'label':
                    tfds.features.ClassLabel(shape=(), num_classes=20)
                })
            }),
            supervised_keys=None,
            homepage=None,
            citation=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(_URL)

        with open(os.path.join(path, _TRAIN_TXT)) as train_fd:
            with open(os.path.join(path, _VAL_TXT)) as val_fd:
                train_filenames = [fname.strip() for fname in train_fd]
                val_filenems = [fname.strip() for fname in train_fd]

        return {
            'train': self._generate_examples(train_filenames),
            'val': self._generate_examples(val_filenems)
        }

    def _generate_objects(self, fname):

        tree = ET.parse(os.path.join(_ANNOTATION_DIR, fname))
        xml = tree.getroot()

        for obj in xml.findall("object"):
            bbox = obj.find("bndbox")
            ymin, xmin = int(bbox.findtext("ymin")), int(bbox.findtext("xmin"))
            ymax, xmax = int(bbox.findtext("ymax")), int(bbox.findtext("xmax"))
            Bbox = namedtuple("ymin", "xmin", "ymax", "xmax")
            yield {
                "bbox": Bbox(ymin, xmin, ymax, xmax),
                "label": obj.findtext("label"),
            }

    def _generate_examples(self, filenames):
        for fname in filenames:
            yield {
                'image': os.path.join(_IMG_DIR, fname + ".jpg"),
                'segmentation_image': os.path.join(_SEG_IMG_DIR,
                                                   fname + ".jpg"),
                'objects': self._generate_objects(fname)
            }

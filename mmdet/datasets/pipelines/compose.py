import collections
import os
import cv2
import numpy as np

from torchvision import transforms
from matplotlib import pyplot as plt
from mmcv.utils import build_from_cfg
from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None

        ##################################
        # Save augmented image (my code) #
        # image = data['img']._data
        # matplotlib_imshow(image, one_channel=True)
        
        
        # save_root = '/home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/visualize'
        # file_id = 0
        # while os.path.exists(os.path.join(save_root, f'img{file_id}.png')) is True:
        #     file_id += 1
        #     continue
        # plt.savefig(os.path.join(save_root, f'img{file_id}.png'))
        # plt.close()
        ##################################

        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

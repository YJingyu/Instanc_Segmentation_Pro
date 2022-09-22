import os.path as osp

import mmcv
import numpy as np
import numpy
import pycocotools.mask as maskUtils
import cv2, os
import random

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

from pycocotools import mask as mask_util


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles:
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadProposals:
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(num_max_proposals={self.num_max_proposals})'


@PIPELINES.register_module()
class FilterAnnotations:
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
    """

    def __init__(self, min_gt_bbox_wh):
        # TODO: add more filter options
        self.min_gt_bbox_wh = min_gt_bbox_wh

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            return None
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results

@PIPELINES.register_module()
class LoadAnnotationsCopyPasteSingle:
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 copy_paste=True,
                 data_dir='/lnt/lipei_algo/lengyu.yb/datasets/segmentation/MMSports22/total_aug'):
                #  data_dir='/lnt/lipei_algo/lengyu.yb/datasets/segmentation/MMSports22/train_aug'):
                # data_dir='/lnt/lipei_algo/lengyu.yb/datasets/segmentation/MMSports22/challenge_crop/'): 
                #  data_dir='/lnt/lipei_algo/lengyu.yb/datasets/segmentation/MMSports22/test_crop/'): 
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_label = with_label
        self.file_client_args = dict(backend='disk').copy()
        self.file_client = None

        # CopyPaste
        if copy_paste:
            print('LoadAnnotationsCopyPaste from ', data_dir)
            self.data_dir = data_dir
            self.copy_paste = copy_paste

            self.crop_image_dir = 'c_images'
            self.crop_label_dir = 'c_labels'

            import glob
            self.src_names_0 = glob.glob(f'{data_dir}/{self.crop_image_dir}/0/*.png')

            with open(f'{data_dir}/{self.crop_label_dir}/0.txt') as f:
                self.labels = {}
                for line in f.readlines():
                    line = line.rstrip().split(' ')
                    self.labels[line[0]] = line[1:]

    @staticmethod
    def _load_boxes(results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_boxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_boxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_boxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    @staticmethod
    def _load_labels(results):
        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        if self.copy_paste:
            img = results['img']

            # print(results['img_info'])
            # image_name = results['img_info']['filename'].split('/')[-1]
            # cv2.imwrite('mmsportspreview/images/'+'bef_'+image_name, img)
            # print('before label', results['gt_labels'])

            img, label, boxes, masks = self.add_objects(img)

            # cv2.imwrite('mmsportspreview/images/'+'add_'+image_name, img)
            # print('add label', label)

            masks.extend(results['ann_info']['masks'])
            label.extend(results['gt_labels'].tolist())
            boxes.extend(results['gt_bboxes'].tolist())

            results['img'] = img
            results['gt_labels'] = numpy.array(label, numpy.int64)
            results['gt_bboxes'] = numpy.array(boxes, numpy.float32)
        else:
            masks = results['ann_info']['masks']
        gt_masks = BitmapMasks([self._poly2mask(mask, h, w) for mask in masks], h, w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    @staticmethod
    def _poly2mask(mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            rle = mask_util.merge(mask_util.frPyObjects(mask_ann, img_h, img_w))
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = mask_util.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = mask_util.decode(rle)
        return mask

    def add_objects(self, img):

        gt_label = []
        gt_masks = []
        gt_boxes = []

        dst_h, dst_w = img.shape[:2]
        # num_0 = numpy.random.randint(5, 15)
        # num_0 = numpy.random.randint(2, 8)
        num_0 = numpy.random.randint(1, 6)
        y_c_list = numpy.random.randint(dst_h // 2 - 256, dst_h // 2 + 256, num_0)
        x_c_list = numpy.random.randint(256, dst_w - 256, num_0)
        src_names = numpy.random.choice(self.src_names_0, num_0).tolist()

        mask_list = []
        poly_list = []
        src_img_list = []
        src_name_list = []
        for src_name in src_names:
            poly = []
            label = self.labels[os.path.basename(src_name)]
            src_img = cv2.imread(src_name)
            for i in range(0, len(label), 2):
                poly.append([int(label[i]), int(label[i + 1])])
            src_mask = numpy.zeros(src_img.shape, src_img.dtype)
            cv2.fillPoly(src_mask, [numpy.array(poly)], (255, 255, 255))
            mask_list.append(src_mask)
            poly_list.append(poly)
            src_img_list.append(src_img)
            src_name_list.append(src_name)

        for i, (x_c, y_c) in enumerate(zip(x_c_list, y_c_list)):
            dst_poly = []
            for p in poly_list[i]:
                dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
            dst_mask = numpy.zeros(img.shape, img.dtype)
            cv2.fillPoly(dst_mask, [numpy.array(dst_poly, int)], (255, 255, 255))
            x_min, y_min, w, h = cv2.boundingRect(numpy.array([dst_poly], int))
            gt_boxes.append([x_min, y_min, x_min + w, y_min + h])
            src = src_img_list[i].copy()
            h, w = src.shape[:2]
            mask = mask_list[i].copy()
            img[dst_mask > 0] = 0
            img[y_c:y_c + h, x_c:x_c + w] += src * (mask > 0)
            gt_label.append(0)
            dst_point = []
            for p in dst_poly:
                dst_point.append(p[0])
                dst_point.append(p[1])
            gt_masks.append([dst_point])
        return img, gt_label, gt_boxes, gt_masks

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_boxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class LoadAnnotationsCopyPaste:
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 copy_paste=True,
                 data_dir='/lnt/lipei_algo/lengyu.yb/datasets/segmentation/VIPriors2022/datasets'):
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_label = with_label
        self.file_client_args = dict(backend='disk').copy()
        self.file_client = None

        # CopyPaste
        if copy_paste:
            self.data_dir = data_dir
            self.copy_paste = copy_paste

            self.crop_image_dir = 'images_crop'
            self.crop_label_dir = 'labels_crop'

            import glob
            self.src_names_0 = glob.glob(f'{data_dir}/{self.crop_image_dir}/0/*.png')
            self.src_names_1 = glob.glob(f'{data_dir}/{self.crop_image_dir}/1/*.png')

            with open(f'{data_dir}/{self.crop_label_dir}/0.txt') as f:
                self.labels = {}
                for line in f.readlines():
                    line = line.rstrip().split(' ')
                    self.labels[line[0]] = line[1:]
            with open(f'{data_dir}/{self.crop_label_dir}/1.txt') as f:
                for line in f.readlines():
                    line = line.rstrip().split(' ')
                    self.labels[line[0]] = line[1:]

    @staticmethod
    def _load_boxes(results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_boxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_boxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_boxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    @staticmethod
    def _load_labels(results):
        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        if self.copy_paste:
            img = results['img']
            img, label, boxes, masks = self.add_objects(img)

            masks.extend(results['ann_info']['masks'])
            label.extend(results['gt_labels'].tolist())
            boxes.extend(results['gt_bboxes'].tolist())

            results['img'] = img
            results['gt_labels'] = numpy.array(label, numpy.int64)
            results['gt_bboxes'] = numpy.array(boxes, numpy.float32)
        else:
            masks = results['ann_info']['masks']
        gt_masks = BitmapMasks([self._poly2mask(mask, h, w) for mask in masks], h, w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    @staticmethod
    def _poly2mask(mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            rle = mask_util.merge(mask_util.frPyObjects(mask_ann, img_h, img_w))
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = mask_util.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = mask_util.decode(rle)
        return mask

    def add_objects(self, img):

        gt_label = []
        gt_masks = []
        gt_boxes = []

        dst_h, dst_w = img.shape[:2]
        num_0 = numpy.random.randint(5, 15)
        num_1 = numpy.random.randint(5, 15)
        y_c_list = numpy.random.randint(dst_h // 2 - 256, dst_h // 2 + 256, num_0 + num_1)
        x_c_list = numpy.random.randint(256, dst_w - 256, num_0 + num_1)
        src_names = numpy.random.choice(self.src_names_0, num_0).tolist()
        src_names.extend(numpy.random.choice(self.src_names_1, num_1).tolist())

        mask_list = []
        poly_list = []
        src_img_list = []
        src_name_list = []
        for src_name in src_names:
            poly = []
            label = self.labels[os.path.basename(src_name)]
            src_img = cv2.imread(src_name)
            for i in range(0, len(label), 2):
                poly.append([int(label[i]), int(label[i + 1])])
            src_mask = numpy.zeros(src_img.shape, src_img.dtype)
            cv2.fillPoly(src_mask, [numpy.array(poly)], (255, 255, 255))
            mask_list.append(src_mask)
            poly_list.append(poly)
            src_img_list.append(src_img)
            src_name_list.append(src_name)
        for i, (x_c, y_c) in enumerate(zip(x_c_list, y_c_list)):
            dst_poly = []
            for p in poly_list[i]:
                dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
            dst_mask = numpy.zeros(img.shape, img.dtype)
            cv2.fillPoly(dst_mask, [numpy.array(dst_poly, int)], (255, 255, 255))
            x_min, y_min, w, h = cv2.boundingRect(numpy.array([dst_poly], int))
            gt_boxes.append([x_min, y_min, x_min + w, y_min + h])
            src = src_img_list[i].copy()
            h, w = src.shape[:2]
            mask = mask_list[i].copy()
            img[dst_mask > 0] = 0
            img[y_c:y_c + h, x_c:x_c + w] += src * (mask > 0)
            if 'human' in os.path.basename(src_name_list[i]):
                gt_label.append(0)
            else:
                gt_label.append(1)
            dst_point = []
            for p in dst_poly:
                dst_point.append(p[0])
                dst_point.append(p[1])
            gt_masks.append([dst_point])
        return img, gt_label, gt_boxes, gt_masks

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_boxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class LoadAnnotationsSpecificCopyPaste:
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 copy_paste=True,
                 data_dir='/lnt/lipei_algo/lengyu.yb/datasets/segmentation/MMSports22/trainval_aug/'):
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_label = with_label
        self.file_client_args = dict(backend='disk').copy()
        self.file_client = None

        # CopyPaste
        if copy_paste:
            print('LoadAnnotationsCopyPaste from ', data_dir)
            self.data_dir = data_dir
            self.copy_paste = copy_paste

            self.crop_image_dir = 'crop_images'
            self.crop_label_dir = 'crop_labels'

            import glob
            self.src_names_0 = glob.glob(f'{data_dir}/{self.crop_image_dir}/0/*.png')

            with open(f'{data_dir}/{self.crop_label_dir}/0.txt') as f:
                self.labels = {}
                for line in f.readlines():
                    line = line.rstrip().split(' ')
                    self.labels[line[0]] = line[1:]

    @staticmethod
    def _load_boxes(results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_boxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_boxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_boxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    @staticmethod
    def _load_labels(results):
        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        if self.copy_paste:
            img = results['img']
            # print(results['img_info'])
            # image_name = results['img_info']['filename'].split('/')[-1]
            # cv2.imwrite('mmsportspreview/images/'+'bef_'+image_name, img)
            # print('before label', results['gt_labels'])
            gt_bboxes = results['gt_bboxes'].tolist() # x1,y1,x2,y2
            num_bboxes = len(gt_bboxes)
            if num_bboxes > 5:
                cp_num = numpy.random.randint(3, num_bboxes)
                cp_bboxes = random.sample(gt_bboxes, cp_num)
                # if image_name == 'camcourt2_1513370019281_0.png':
                #     print(gt_bboxes)
                #     print(cp_bboxes)
                # cv2.rectangle(img, (int(gt_bboxes[0][0]),int(gt_bboxes[0][1])), (int(gt_bboxes[0][2]),int(gt_bboxes[0][3])), (255, 0, 255), 2)
                # cv2.imwrite('mmsportspreview/bboxes/'+'box_'+image_name, img)
                y_c_list = []
                x_c_list = []
                for cp_bbox in cp_bboxes:
                    y_min = int(1.5*cp_bbox[1] - 0.5*cp_bbox[3])
                    y_min = max(0, y_min)
                    y_max = int(0.5*cp_bbox[1] + 0.5*cp_bbox[3])
                    y_c = numpy.random.randint(y_min, y_max)
                    x_min = int(2*cp_bbox[0] - cp_bbox[2])
                    x_min = max(0, x_min)
                    x_max = cp_bbox[2]
                    x_c = numpy.random.randint(x_min, x_max)
                    y_c_list.append(y_c)
                    x_c_list.append(x_c)
            else:            
                dst_h, dst_w = img.shape[:2]
                num_0 = numpy.random.randint(5, 15)
                y_c_list = numpy.random.randint(int(dst_h*0.25), int(dst_h*0.8), num_0)
                x_c_list = numpy.random.randint(int(dst_w*0.2), int(dst_w*0.8), num_0)

            img, label, boxes, masks = self.add_objects(img, y_c_list, x_c_list)
            # cv2.imwrite('mmsportspreview/specopy/'+'add_'+image_name, img)
            # print('add label', label)
            masks.extend(results['ann_info']['masks'])
            label.extend(results['gt_labels'].tolist())
            boxes.extend(results['gt_bboxes'].tolist())
            results['img'] = img
            results['gt_labels'] = numpy.array(label, numpy.int64)
            results['gt_bboxes'] = numpy.array(boxes, numpy.float32)
        else:
            masks = results['ann_info']['masks']
        gt_masks = BitmapMasks([self._poly2mask(mask, h, w) for mask in masks], h, w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    @staticmethod
    def _poly2mask(mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            rle = mask_util.merge(mask_util.frPyObjects(mask_ann, img_h, img_w))
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = mask_util.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = mask_util.decode(rle)
        return mask

    def add_objects(self, img, y_c_list, x_c_list):
        dst_h, dst_w = img.shape[:2]
        gt_label = []
        gt_masks = []
        gt_boxes = []
        src_names = numpy.random.choice(self.src_names_0, len(y_c_list)).tolist()

        mask_list = []
        poly_list = []
        src_img_list = []
        src_name_list = []
        for src_name in src_names:
            poly = []
            label = self.labels[os.path.basename(src_name)]
            src_img = cv2.imread(src_name)
            for i in range(0, len(label), 2):
                poly.append([int(label[i]), int(label[i + 1])])
            src_mask = numpy.zeros(src_img.shape, src_img.dtype)
            cv2.fillPoly(src_mask, [numpy.array(poly)], (255, 255, 255))
            mask_list.append(src_mask)
            poly_list.append(poly)
            src_img_list.append(src_img)
            src_name_list.append(src_name)

        for i, (x_c, y_c) in enumerate(zip(x_c_list, y_c_list)):
            src = src_img_list[i].copy()
            h, w = src.shape[:2]
            if y_c + h < dst_h and x_c + w < dst_w:
                dst_poly = []
                for p in poly_list[i]:
                    dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
                dst_mask = numpy.zeros(img.shape, img.dtype)
                cv2.fillPoly(dst_mask, [numpy.array(dst_poly, int)], (255, 255, 255))
                x_min, y_min, w, h = cv2.boundingRect(numpy.array([dst_poly], int))
                gt_boxes.append([x_min, y_min, x_min + w, y_min + h])
                mask = mask_list[i].copy()
                img[dst_mask > 0] = 0
                img[y_c:y_c + h, x_c:x_c + w] += src * (mask > 0)
                gt_label.append(0)
                dst_point = []
                for p in dst_poly:
                    dst_point.append(p[0])
                    dst_point.append(p[1])
                gt_masks.append([dst_point])
        return img, gt_label, gt_boxes, gt_masks

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_boxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        return results

    def __repr__(self):
        return self.__class__.__name__

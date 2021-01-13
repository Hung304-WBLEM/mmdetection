_base_ = '../nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py'

model = dict(bbox_head=dict(num_classes=2))

# Dataset path
DDSM_TRAIN_DATASET = '/home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train'
DDSM_TRAIN_ANNOTATION = DDSM_TRAIN_DATASET + '/annotation_coco_with_classes.json'
DDSM_TEST_DATASET = '/home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test'
DDSM_TEST_ANNOTATION = DDSM_TEST_DATASET + '/annotation_coco_with_classes.json'


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('malignant', 'benign')
data = dict(
    train=dict(
        img_prefix=DDSM_TRAIN_DATASET,
        classes=classes,
        ann_file=DDSM_TRAIN_ANNOTATION),
    val=dict(
        img_prefix=DDSM_TEST_DATASET,
        classes=classes,
        ann_file=DDSM_TEST_ANNOTATION),
    test=dict(
        img_prefix=DDSM_TEST_DATASET,
        classes=classes,
        ann_file=DDSM_TEST_ANNOTATION))
total_epochs = 24

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth'

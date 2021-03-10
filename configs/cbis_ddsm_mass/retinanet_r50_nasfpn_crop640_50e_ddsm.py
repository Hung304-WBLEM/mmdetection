_base_ = '../nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py'

model = dict(bbox_head=dict(num_classes=2))

# Dataset path
DDSM_TRAIN_DATASET = '/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train_train'
DDSM_TRAIN_ANNOTATION = DDSM_TRAIN_DATASET + '/annotation_coco_with_classes.json'
DDSM_VAL_DATASET = '/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train_val'
DDSM_VAL_ANNOTATION = DDSM_VAL_DATASET + '/annotation_coco_with_classes.json'
DDSM_TEST_DATASET = '/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test'
DDSM_TEST_ANNOTATION = DDSM_TEST_DATASET + '/annotation_coco_with_classes.json'


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('malignant-mass', 'benign-mass')
data = dict(
    samples_per_gpu=4,
    workes_per_gpu=4,
    train=dict(
        img_prefix=DDSM_TRAIN_DATASET,
        classes=classes,
        ann_file=DDSM_TRAIN_ANNOTATION),
    val=dict(
        img_prefix=DDSM_VAL_DATASET,
        classes=classes,
        ann_file=DDSM_VAL_ANNOTATION),
    test=dict(
        img_prefix=DDSM_TEST_DATASET,
        classes=classes,
        ann_file=DDSM_TEST_ANNOTATION))
total_epochs = 50

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth'
work_dir = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_mass/retinanet_r50_nasfpn_crop640_50e_ddsm'

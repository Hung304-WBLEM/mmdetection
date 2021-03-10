_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=2))
)

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
    workers_per_gpu=4,
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

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth'
work_dir = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm'
# resume_from = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/latest.pth'

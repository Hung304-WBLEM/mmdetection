_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=2))
)
# Dataset path
DDSM_TRAIN_DATASET = '/home/hqvo2/Projects/Breast_Cancer/data/processed_data2/calc/train_train'
DDSM_TRAIN_ANNOTATION = DDSM_TRAIN_DATASET + '/annotation_coco_with_classes.json'
DDSM_VAL_DATASET = '/home/hqvo2/Projects/Breast_Cancer/data/processed_data2/calc/train_val'
DDSM_VAL_ANNOTATION = DDSM_VAL_DATASET + '/annotation_coco_with_classes.json'
DDSM_TEST_DATASET = '/home/hqvo2/Projects/Breast_Cancer/data/processed_data2/calc/test'
DDSM_TEST_ANNOTATION = DDSM_TEST_DATASET + '/annotation_coco_with_classes.json'

albu_train_transforms = [
    dict(
        type='CLAHE',
        tile_grid_size=(32, 32),
        p=1
    )
]
albu_test_transforms = [
    dict(
        type='CLAHE',
        tile_grid_size=(32, 32),
        p=1
    )
]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Albu',
                transforms=albu_test_transforms,
                update_pad_shape=False,
                skip_img_without_anno=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('malignant-calc', 'benign-calc')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        pipeline=train_pipeline,
        img_prefix=DDSM_TRAIN_DATASET,
        classes=classes,
        ann_file=DDSM_TRAIN_ANNOTATION),
    val=dict(
        pipeline=test_pipeline,
        img_prefix=DDSM_VAL_DATASET,
        classes=classes,
        ann_file=DDSM_VAL_ANNOTATION),
    test=dict(
        pipeline=test_pipeline,
        img_prefix=DDSM_TEST_DATASET,
        classes=classes,
        ann_file=DDSM_TEST_ANNOTATION))


total_epochs = 24

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth'
work_dir = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/calc/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu-clahe32x32'
# resume_from = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/latest.pth'

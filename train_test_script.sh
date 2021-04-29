module load cudatoolkit/10.1

############################
########## Train ###########
############################
cd /home/hqvo2/Projects/Breast_Cancer/libs/mmdetection

# Train on Mass data
# bash tools/dist_train.sh configs/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py 4
# bash tools/dist_train.sh configs/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu.py 4

# Train on Calcification data
# bash tools/dist_train.sh configs/cbis_ddsm_calc/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py 4
bash tools/dist_train.sh configs/cbis_ddsm_calc/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu.py 4

############################
###### Get Best Ckpt #######
############################
cd /home/hqvo2/Projects/Breast_Cancer/source/evaluation

mass_detection_root="/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/mass"
calc_detection_root="/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/calc"

python eval_mmdet_models.py --parent_root $mass_detection_root --metric bbox_mAP_50
python eval_mmdet_models.py --parent_root $calc_detection_root --metric bbox_mAP_50


############################
######### Predict ##########
############################
cd /home/hqvo2/Projects/Breast_Cancer/libs/mmdetection

# Test on Mass data
# PORT=29501 ./tools/dist_test.sh configs/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py \
#     ${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/best_ckpt.pth 2 \
#     --format-only --eval-options "jsonfile_prefix=${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/test_bboxes"

# PORT=29501 ./tools/dist_test.sh configs/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu.py \
#     ${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu/best_ckpt.pth 2 \
#     --format-only --eval-options "jsonfile_prefix=${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu/test_bboxes"

# Test on Calc data
# PORT=29501 ./tools/dist_test.sh configs/cbis_ddsm_calc/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py \
#     ${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/best_ckpt.pth 4 \
#     --format-only --eval-options "jsonfile_prefix=${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/test_bboxes"

PORT=29501 ./tools/dist_test.sh configs/cbis_ddsm_calc/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu.py \
    ${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu-clahe32x32/best_ckpt.pth 4 \
    --format-only --eval-options "jsonfile_prefix=${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu-clahe32x32/test_bboxes"


############################
####### Evaluation #########
############################
cd /home/hqvo2/Projects/Breast_Cancer/source/evaluation

mass_test_gt="/home/hqvo2/Datasets/processed_data2/mass/test/annotation_coco_with_classes.json"
calc_test_gt="/home/hqvo2/Datasets/processed_data2/calc/test/annotation_coco_with_classes.json"

# Plot for Mass data
# python plot_eval_curve.py -gt ${mass_test_gt} -p ${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/test_bboxes.bbox.json -bb all -s ${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/
# python plot_eval_curve.py -gt ${mass_test_gt} -p ${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu/test_bboxes.bbox.json -bb all -s ${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu/

# Plot for Calc data
# python plot_eval_curve.py -gt ${calc_test_gt} -p ${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/test_bboxes.bbox.json -bb all -s ${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/
python plot_eval_curve.py -gt ${calc_test_gt} -p ${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu-clahe32x32/test_bboxes.bbox.json -bb all -s ${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu-clahe32x32/

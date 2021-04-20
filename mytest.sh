# ./tools/dist_test.sh configs/ddsm/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py ../../experiments/mmdet_processed_data/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/epoch_15.pth 1 --format-only --eval-options "jsonfile_prefix=./res_best"
# ./tools/dist_test.sh configs/ddsm/faster_rcnn_r50_caffe_fpn_mstrain_1x_ohem_ddsm.py ../../experiments/mmdet_processed_data/faster_rcnn_r50_caffe_fpn_mstrain_1x_default_ohem_ddsm/epoch_17.pth 1 --format-only --eval-options "jsonfile_prefix=./res_ohem"

mass_detection_root="/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/mass"
calc_detection_root="/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/calc"

# PORT=29501 ./tools/dist_test.sh configs/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py \
#     ${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/best_ckpt.pth 1 \
#     --format-only --eval-options "jsonfile_prefix=${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/test_bboxes"

PORT=29501 ./tools/dist_test.sh configs/cbis_ddsm_calc/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py \
    ${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/best_ckpt.pth 1 \
    --format-only --eval-options "jsonfile_prefix=${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/test_bboxes"

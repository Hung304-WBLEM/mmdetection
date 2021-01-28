#!/bin/bash

#SBATCH -J train_high_res
#SBATCH -o result.o%j
#SBATCH -N 1 
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=30
#SBATCH -t 4:00:00
#SBATCH --mem-per-cpu=2048

#SBATCH --mail-user=voquochung304@gmail.com
#SBATCH --mail-type=all

module load cudatoolkit/10.1

cd /home/hqvo2/Projects/Breast_Cancer/libs/mmdetection

# bash tools/dist_train.sh configs/ddsm/faster_rcnn_r50_caffe_fpn_mstrain_1x_large_ddsm.py 4
# bash tools/dist_train.sh configs/ddsm/faster_rcnn_r50_caffe_fpn_mstrain_1x_verylarge_ddsm.py 8 
# bash tools/dist_train.sh configs/ddsm/faster_rcnn_r50_caffe_fpn_mstrain_1x_ohem_ddsm.py 4 
bash tools/dist_train.sh configs/ddsm/faster_rcnn_r50_caffe_fpn_mstrain_1x_200eps_ddsm.py 4 
# bash tools/dist_train.sh configs/ddsm/retinanet_r50_nasfpn_crop640_50e_ddsm_test.py 8
# bash tools/dist_train.sh configs/ddsm/retinanet_r50_nasfpn_crop640_50e_large_ddsm.py 4
# bash tools/dist_train.sh configs/ddsm/retinanet_r50_nasfpn_crop640_50e_verylarge_ddsm.py 8
# bash tools/dist_train.sh configs/ddsm/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_ultralarge_ddsm.py 8
# bash tools/dist_train.sh configs/ddsm/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_crop_ddsm.py 8
# bash tools/dist_train.sh configs/ddsm/mask_rcnn_r49_caffe_fpn_mstrain-poly_1x_crop2_ddsm.py 8
# bash tools/dist_train.sh configs/ddsm/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_crop3_ddsm.py 8

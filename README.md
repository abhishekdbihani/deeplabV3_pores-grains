# MudrockNet: Semantic Segmentation of Mudrock SEM Images through Deep learning

This repository contains the deep learning SEM image segmentation model, MudrockNet, based on the [DeeplabV3+ architecture](https://arxiv.org/abs/1802.02611) and applied using [TensorFlow in Python](https://github.com/tensorflow/models/tree/master/research/deeplab) for semantic segmentation of pores and grains.

The repository was created by Abhishek Bihani, Hugh Daigle, Javier E. Santos, Christopher Landry, Masa Prodanovic and Kitty Milliken.

## Description
The trained MudrockNet model can be used for detection of pores (green) and large i.e. silt size grains (red) from SEM images of shales or mudrocks. An example is shown in the below image. The original training dataset can be found in [Milliken et al. (2016)]( https://www.digitalrocksportal.org/projects/42), and training images with associated ground truth data (segmented images) are available in [Bihani et al. (2020)]( https://www.digitalrocksportal.org/projects/259)

<img src="https://github.com/abhishekdbihani/MudrockNet/blob/master/images/sem_sample1.5.png" align="middle" width="800" height="900" alt="SEM image: pores and grains" >

Figure 1 shows the overlay mask of ground truth data (A), MudrockNet model predictions (B), and trainable Weka model predictions in ImageJ (C), on four selected SEM images from the test set. The silt grains are in red, pores in green, clay in transparent color, and the truth images show a scale bar for reference. 
 

# Workflow

## 1) Dataset and MudrockNet model download:
This repository contains large trained model files which require [LTS](https://git-lfs.github.com/) for downloading. The repository with the trained MudrockNet model and the test images can be downloaded after installing LFS in the folder using following commands. 

Command:
```bash
git lfs install

git lfs clone https://github.com/abhishekdbihani/MudrockNet                                                                                               
```

## 2) Dataset creation: 
The images ([raw](https://github.com/abhishekdbihani/MudrockNet/tree/master/dataset/data/JPEGImages) + [label/ground truth](https://github.com/abhishekdbihani/MudrockNet/tree/master/dataset/data/SegmentationClassRaw)) need to be converted to TensorFlow TFRecords before conducting training. The images were split randomly into [training, validation and test datasets](https://github.com/abhishekdbihani/MudrockNet/tree/master/dataset/data) in a ratio 70:15:15.

Command:
```bash
python create_pascal_tf_record.py --data_dir DATA_DIR \
                                  --image_data_dir IMAGE_DATA_DIR \
                                  --label_data_dir LABEL_DATA_DIR
```
Ready TFRecords of training and evaluation datasets are located [here](https://github.com/abhishekdbihani/MudrockNet/tree/master/dataset/tfrecord).

## 3) Training: 
The MudrockNet model can be trained by using a pre-trained checkpoint such as a pre-trained [Resnet101 v2 model](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz) used here. The final trained model checkpoint files are located [here](https://github.com/abhishekdbihani/MudrockNet/tree/master/dataset/model).

Command:
```bash
python train.py --model_dir MODEL_DIR --pre_trained_model PRE_TRAINED_MODEL
```

The training process can be checked with Tensor Board.
Command:
```bash
tensorboard --logdir MODEL_DIR
```
The metrics for training and validation are seen below in Figure 2:
<img src="https://github.com/abhishekdbihani/MudrockNet/blob/master/images/deeplab_metrics1.png" align="middle" width="600" height="900" alt="SEM image model: metrics" >

MudrockNet was trained on the images using a NVIDIA GeForce GTX 1070 GPU with 8 GB memory. The training was stopped after 50 epochs, once the training and validation loss became constant (values 13.25 and 13.52 respectively), and the training and validation pixel-accuracy reached a plateau (values 0.9205 and 0.8898 respectively).

## 4) Evaluation:
The MudrockNet model was evaluated for intersection over union (IoU) of different classes (pores and large grains) with following results (Table 1):


| Mean IoU values  | Training | Validation |  Test  |
|:----------------:|:--------:|:----------:|:-------:
| Silt grains      |  0.7169  |  0.6443    | 0.6591
| Pores            |  0.6844  |  0.6744    | 0.6642  

Command:

```bash
python evaluate.py --help
```

## 5) Inference:
The trained MudrockNet model can be used for segmentation of any SEM mudrock images. The results of segmentation on test data by the trained model are available [here](https://github.com/abhishekdbihani/MudrockNet/tree/master/dataset/data/output_test) and the comparison of the results of ground truth data and predictions using overlay masks are available [here](https://github.com/abhishekdbihani/MudrockNet/tree/master/dataset/data/masks_test). 

Command:
```bash
python inference.py --data_dir DATA_DIR --infer_data_list INFER_DATA_LIST --model_dir MODEL_DIR 
```
The trainable Weka model using the random forest classifier in ImageJ is uploaded [here](https://github.com/abhishekdbihani/MudrockNet/blob/master/weka_classifier.model) for comparison. It is trained on three classes: silt, pore, clay.

The IoU comparisons with ground truth data for silt and pore values from the MudrockNet and Weka model for images in Figure 1 are given below in Table 2:

|IoU values |	MudrockNet |MudrockNet |Weka       | Weka|
|:---------:|:----------:|:---------:|:---------:|:----|
|Image      | Silt grains| Pores     |Silt grains|Pores|
|1          | 0.892	     | 0.729     |	0.702    |0.581|
|2          | 0.822	     | 0.655     |	0.665    |0.667|
|3          | 0.889	     | 0.667     |	0.578    |0.707|
|4          | 0.881	     | 0.816     |	0.543    |0.497|


## Citation:

If you use our model, please cite as:
Bihani A., Daigle H., Santos J. E., Landry C., Prodanovic M., Milliken K. MudrockNet: Semantic Segmentation of Mudrock SEM Images through Deep learning. Git code (2019) https://github.com/abhishekdbihani/MudrockNet.

## Author dataset publications:

1) Bihani, A., Daigle, H., Prodanovic, M., Milliken, K., & E. Santos, J. (2020). Mudrock images from Nankai Trough [Data set]. Digital Rocks Portal. https://doi.org/10.17612/BVXS-BC79.

2) Milliken, K. L., Prodanovic, M., Nole, M., & Daigle, H. (2016). Unconsolidated muds from the Nankai Trough [Data set]. Digital Rocks Portal University of Texas at Austin. https://doi.org/10.17612/P7F59W.

## Acknowledgments

1) MudrockNet was implemented using the [TensorFlow in Python](https://github.com/tensorflow/models/tree/master/research/deeplab), and the implemented model and the architecture (DeepLab-v3+) is based on the variant by [rishizek]( http:/github.com/rishizek/tensorflow-deeplab-v3-plus) made for the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

2) Samples and data were provided by the Integrated Ocean Drilling Program (IODP). Funding for sample preparation and SEM imaging was provided by a post-expedition award (Milliken, P.I.) from the Consortium for Ocean Leadership.

3) The authors are grateful to https://github.com/anuragbihani for his assistance in creating MudrockNet.

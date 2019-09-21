# deeplabV3_pores-grains

This repository contains a model of [DeeplabV3+ Model](https://arxiv.org/abs/1802.02611) applied using [TensorFlow in Python](https://github.com/tensorflow/models/tree/master/research/deeplab) for semantic segmentation of pores and grains. Deeplab is a state-of-the-art segmentation model created and released as open source by Google.

## Acknowledgment
It has been modified from https://github.com/rishizek/tensorflow-deeplab-v3-plus which was made for segmentation of [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/)

Note: It was not directly forked as size of the trained model files required [LTS](https://git-lfs.github.com/) , which is not supported for forked public repositories. The authors are grateful to https://github.com/anuragbihani for his assistance.

## Description
This trained model can be used for detection of pores (green) and large i.e. silt size grains (red) from SEM images of shales or mudrocks. An example is shown in the below image. The original dataset can be found here: https://www.digitalrocksportal.org/projects/42

<img src="https://github.com/abhishekdbihani/deeplabV3_pores-grains/blob/master/images/sem_sample1.png" align="middle" width="800" height="400" alt="SEM image: pores and grains" >

Figure shows the overlay mask of ground truth data (A) and predictions (B) on the SEM images with silt grains in red and pores in green with the image grain IoU equal to 0.89 and pore IoU equal to 0.69.

# Workflow

## 1) Dataset creation: 
The images ([raw](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/data/JPEGImages) + [label/ground truth](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/data/SegmentationClassRaw)) need to be converted to TensorFlow TFRecords before conducting training. The images were split randomly into [training, validation and test datasets](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/data) in a ratio 80:15:5.

Command:
```bash
python create_pascal_tf_record.py --data_dir DATA_DIR \
                                  --image_data_dir IMAGE_DATA_DIR \
                                  --label_data_dir LABEL_DATA_DIR
```
Ready TFRecords of training and evaluation datasets are located [here](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/tfrecord).

## 2) Training: 
The model was trained by using a pre-trained checkpoint such as a pre-trained [Resnet v2 model](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz) used here. The final trained model checkpoint files are located [here](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/model).

Command:
```bash
python train.py --model_dir MODEL_DIR --pre_trained_model PRE_TRAINED_MODEL
```

The training process can be checked with Tensor Board.
Command:
```bash
tensorboard --logdir MODEL_DIR
```
The metrics for training and validation are seen below:
<img src="https://github.com/abhishekdbihani/deeplabV3_pores-grains/blob/master/images/deeplab_metrics.png" align="middle" width="600" height="400" alt="SEM image model: metrics" >

The model was trained on the images using a NVIDIA GeForce GTX 1070 GPU with 8 GB memory for ~3 hours. The training was stopped after a total of 10920 steps (33 epochs), once the decreasing total training and validation loss became constant (values 19.45 and 19.22 respectively), and the training and validation pixel accuracy reached a plateau (values 0.8971 and 0.8912 respectively). 

## 3) Evaluation:
The model was evaluated for Intersection over Union (IoU) of different classes (pores and large grains) with following results:


| Mean IoU values  | Training | Validation |  Test  |
|:----------------:|:--------:|:----------:|:-------:
| Silt grains      |  0.6807  |  0.5556    | 0.5732
| Pores            |  0.6394  |  0.6148    | 0.6229  

Command:

```bash
python evaluate.py --help
```

## 4) Inference:
The trained model can be used for segmentation of any SEM mudrock images. The dataset used to test our model is located [here](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/data/JPEG_test).

Command:
```bash
python inference.py --data_dir DATA_DIR --infer_data_list INFER_DATA_LIST --model_dir MODEL_DIR 
```

## Citation
If you use our model, please cite as: 
Bihani A., Daigle H., Santos J. E., Landry C., Prodanovic M., Milliken K. (2019). Insight into the Sealing Capacity of Mudrocks determined using a Digital Rock Physics Workflow. TACC Symposium for Texas Researchers (TACCSTER), 26-27 September, Austin, TX, USA. 




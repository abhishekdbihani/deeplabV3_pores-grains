# Semantic Segmentation of Mudrock SEM Images through Deep learning (deeplabV3_pores-grains)

This repository contains a model of [DeeplabV3+ Model](https://arxiv.org/abs/1802.02611) applied using [TensorFlow in Python](https://github.com/tensorflow/models/tree/master/research/deeplab) for semantic segmentation of pores and grains. Deeplab is a state-of-the-art segmentation model created and released as open source by Google.

The repository was created by Abhishek Bihani in collaboration with Hugh Daigle, Javier E. Santos, Christopher Landry, Masa Prodanovic and Kitty Milliken.

## Description
The trained model can be used for detection of pores (green) and large i.e. silt size grains (red) from SEM images of shales or mudrocks. An example is shown in the below image. The original dataset can be found here: https://www.digitalrocksportal.org/projects/42

<img src="https://github.com/abhishekdbihani/deeplabV3_pores-grains/blob/master/images/sem_sample1.1.png" align="middle" width="800" height="400" alt="SEM image: pores and grains" >

Figure shows the overlay mask of ground truth data (A) and predictions (B) on the SEM images with silt grains in red and pores in green with the image grain Intersection over Union (IoU) equal to 0.89 and pore IoU equal to 0.69.

## Acknowledgments
1) The workflow has been modified from https://github.com/rishizek/tensorflow-deeplab-v3-plus which was made for segmentation of [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).
2) Samples and data were provided by the Integrated Ocean Drilling Program (IODP). Funding for sample preparation and SEM imaging was provided by a post-expedition award (Milliken, P.I.) from the Consortium for Ocean Leadership.
3) The authors are grateful to https://github.com/anuragbihani for his assistance.

Note: The repository was not directly forked as size of the trained model files required [LTS](https://git-lfs.github.com/) , which is not supported for forked public repositories. 

# Workflow

## 1) Dataset and model download:
The repository with the trained model and the test images can be downloaded after installing LFS in the folder using following commands. 

Command:
```bash
git lfs install

git lfs clone https://github.com/abhishekdbihani/deeplabV3_pores-grains                                                                                                
```

## 2) Dataset creation: 
The images ([raw](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/data/JPEGImages) + [label/ground truth](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/data/SegmentationClassRaw)) need to be converted to TensorFlow TFRecords before conducting training. The images were split randomly into [training, validation and test datasets](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/data) in a ratio 70:15:15.

Command:
```bash
python create_pascal_tf_record.py --data_dir DATA_DIR \
                                  --image_data_dir IMAGE_DATA_DIR \
                                  --label_data_dir LABEL_DATA_DIR
```
Ready TFRecords of training and evaluation datasets are located [here](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/tfrecord).

## 3) Training: 
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
<img src="https://github.com/abhishekdbihani/deeplabV3_pores-grains/blob/master/images/deeplab_metrics1.png" align="middle" width="600" height="400" alt="SEM image model: metrics" >

The model was trained on the images using a NVIDIA GeForce GTX 1070 GPU with 8 GB memory. The network was fully trained after 45 epochs, once the training and validation loss became constant (values 18.89 and 18.97 respectively), and the training and validation pixel-accuracy reached a plateau (values 0.9418 and 0.9089 respectively).

## 4) Evaluation:
The model was evaluated for IoU of different classes (pores and large grains) with following results:


| Mean IoU values  | Training | Validation |  Test  |
|:----------------:|:--------:|:----------:|:-------:
| Silt grains      |  0.6626  |  0.5669    | 0.5308
| Pores            |  0.6725  |  0.6806    | 0.6551  

Command:

```bash
python evaluate.py --help
```

## 5) Inference:
The trained model can be used for segmentation of any SEM mudrock images. The results of segmentation on test data by the trained model are available [here](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/data/output_test) and the comparison of the results of of ground truth data and predictions using overlay masks are available [here](https://github.com/abhishekdbihani/deeplabV3_pores-grains/tree/master/dataset/data/masks_test). 

Command:
```bash
python inference.py --data_dir DATA_DIR --infer_data_list INFER_DATA_LIST --model_dir MODEL_DIR 
```
## Citation:

If you use our model, please cite as:
Bihani A., Daigle H., Santos J. E., Landry C., Prodanovic M., Milliken K. Semantic Segmentation of Mudrock SEM Images through Deep learning. Git code (2019).

## Author publications:

1) Bihani A., Daigle H., Santos J. E., Landry C., Prodanovic M., Milliken K. (2019). Insight into the Sealing Capacity of Mudrocks determined using a Digital Rock Physics Workflow. TACC Symposium for Texas Researchers (TACCSTER), 26-27 September, Austin, TX, USA. 

2) Bihani A., Daigle H., Santos J., Landry C., Prodanović M., Milliken K. (2019). H44B-06: Insight into the Sealing Capacity of Mudrocks determined using a Digital Rock Physics Workflow. AGU Fall Meeting, 9-13 December, San Francisco, USA.

# Chest X-Ray Classification with ResNet-50
This project implements a deep learning model for classifying chest X-ray images into three categories: COVID-19, Normal, and Viral Pneumonia. It utilizes a ResNet-50 model with transfer learning, leveraging pre-trained ImageNet weights for improved accuracy.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyincavusbi/ChestResNet/blob/main/ChestResNet.ipynb)

## Dataset

The model is trained and evaluated on the "Covid19-dataset" available on Kaggle: [Covid19-dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

The dataset contains chest X-ray images in JPEG format, divided into 'train' and 'test' folders, with each folder containing subfolders for the three classes:

*   Covid
*   Normal
*   Viral Pneumonia

## Model

The model used in this project is ResNet-50, a deep convolutional neural network known for its strong performance in image classification tasks. Transfer learning is employed by using the ResNet-50 model pre-trained on the ImageNet dataset. The final fully connected layer of the ResNet-50 model is replaced with a new layer to perform 3-class classification specific to this project.

## Pretrained Model

This project uses a pretrained ResNet-50 model, which is downloaded directly from Kaggle using `kagglehub`. The model can be found at the following link: [ResNet-50 on Kaggle](https://www.kaggle.com/models/tensorflow/resnet-50/frameworks/tensorFlow2/variations/classification/versions/1)

## Requirements

This notebook is intended to be run on Google Colab. The following libraries are used:

*   TensorFlow
*   Keras
*   Scikit-learn
*   Matplotlib
*   PIL
*   Numpy
*   Pathlib
*   Shutil
*   Kagglehub
*   OS

## Usage

1. Open the notebook in Google Colab: [Open in Colab](https://colab.research.google.com/github/huseyincavusbi/ChestResNet/blob/main/ChestResNet.ipynb)
2. Ensure the runtime is set to GPU (Runtime -> Change runtime type -> Hardware accelerator -> GPU).
3. Run all cells in the notebook sequentially.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

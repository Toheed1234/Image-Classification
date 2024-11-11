# Fruit Image Classifier using MobileNetV2
This project implements an image classifier using MobileNetV2 architecture for classifying fruits and vegetables from the Fruits 360 dataset. The objective of this project is to compare the performance of a pre-trained model and a modified version of the model, and analyze the results to improve the model's accuracy.

Project Overview
In this project, I used a Convolutional Neural Network (CNN) model based on MobileNetV2 for the classification task. The dataset consists of 90,483 images of fruits and vegetables, with 131 classes. The goal was to build a model that can predict the fruit or vegetable category based on an input image.

Key Objectives:
Load and preprocess the Fruits 360 dataset.
Train a MobileNetV2 model and evaluate its performance.
Modify the pre-trained model by changing the architecture (e.g., adjusting the head layer).
Compare the performance of the original pre-trained model and the modified version.
Visualize training metrics (accuracy, loss, confusion matrix).
Dataset
The Fruits 360 dataset consists of:

Total number of images: 90,483
Training set size: 67,692 images (one fruit or vegetable per image).
Test set size: 22,688 images (one fruit or vegetable per image).
Number of classes: 131 classes (fruits and vegetables).
Image size: 100x100 pixels.
Key Techniques
Data Augmentation
Data augmentation was applied to the training data to increase diversity and reduce overfitting. The following techniques were used:

Flips (horizontal/vertical)
Rotation (90-degree and finer angles)
Translation (shifting images along x/y axes)
Scaling (resizing images)
Salt and pepper noise addition
Normalization and Standardization
Normalization and standardization were used to preprocess the images for improved model performance. The pixel values were scaled to a range between 0 and 1 to assist in faster convergence during training.

Model Architecture
The model is based on MobileNetV2, which is known for being lightweight and efficient, making it suitable for mobile and edge devices.

Convolutional Layers: To detect patterns like edges or textures.
Pooling Layers: To reduce the size of feature maps while retaining important information.
Flattening: To convert the 2D feature maps into 1D vectors.
Activation Functions: ReLU was used to introduce non-linearity.
Linear Layer: Outputs the final classification probabilities.
Softmax: Used in the output layer to classify images into one of the 131 categories.
Regularization Techniques
To prevent overfitting and improve generalization, the following techniques were employed:

Dropout: Randomly deactivating neurons during training.
Batch Normalization: Stabilizing the learning process by normalizing layer inputs.
Training
Hyperparameters
Optimizer: Adam
Learning Rate: Adaptive, adjusted during training
Batch Size: 32
Epochs: 50 (adjustable)
Loss Function
The Cross-Entropy Loss function was used for the classification task, as it works well for multi-class problems.

Results
Performance
Training Accuracy: 97.00%
Validation Accuracy: 84.72%
Test Accuracy: 80.94%
The model showed signs of overfitting, as indicated by the high training accuracy and the gap between training and validation performance. However, the test accuracy being similar to validation suggests that the model generalizes reasonably well.

Visualization
The following visualizations were provided during the project:

Accuracy and Loss Curves: To track model training and validation performance.
Confusion Matrix: To evaluate the misclassifications.
Improvements
Several techniques could be used to improve the model:

Data Augmentation: To generate more diverse training data.
Regularization: To prevent overfitting.
Hyperparameter Tuning: To find optimal model configurations.
Learning Rate Scheduling: To improve convergence during training.
Ensemble Methods: Combining multiple models for better predictions.
Conclusion
This project demonstrates the use of MobileNetV2 for classifying fruits and vegetables using the Fruits 360 dataset. While the model performed reasonably well, there is room for improvement, particularly in addressing overfitting and fine-tuning hyperparameters. Future work may involve exploring alternative models, further data augmentation, and advanced regularization techniques.

Requirements
To run this project, you will need:

Python 3.x
TensorFlow or PyTorch
Matplotlib
Seaborn


License
This project is licensed under the MIT License - see the LICENSE file for details.

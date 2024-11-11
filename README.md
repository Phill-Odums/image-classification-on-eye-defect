# image-classification-on-eye-defect
# Eye Defect Detection Model Documentation

This documentation outlines the process of building a model to detect various eye defects using Python and machine learning libraries.

### Introduction

The model is designed to detect eye defects such as bulging eyes, cataracts, crossed eyes, glaucoma, and uveitis to assist ophthalmologists in diagnostic decision-making, surgical planning, patient education, and research.

### Libraries Used

The model utilizes TensorFlow, Keras, Tuna, Layers, Hyperparameter, CV2P, and Gloop for importing image datasets and building the classification model.

### Data Import and Processing

- Image datasets are imported and stored in separate folders for each eye defect.
- Data is processed by converting images to RGB format, resizing, and normalizing pixel values.
- Processed images and labels are stored in NumPy arrays for model training.

### Data Splitting

The dataset is split into training and testing sets with an 80/20 split ratio for model evaluation.

### Convolutional Neural Network (CNN) Model

- A CNN model is constructed with BatchNormalization and MaxPooling layers for image classification.
- Hyperparameters are tuned using Tuna for optimal model performance.

### Model Training

- The model is trained with multiple trials to achieve the best accuracy.
- Training parameters such as epochs and batch size are set to optimize learning.

### Model Evaluation

- The model undergoes multiple trials to improve performance.
- Challenges such as data quality and regulatory approvals are highlighted for future enhancements.

### Conclusion


# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
import keras_tuner as kt
from keras_tuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
import cv2,PIL,glob,pathlib
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import random

# Define dataset paths
for dirname,_, filenames in os.walk(r'C:/Users/phill/Downloads/normal eye/Eye_diseases'):
    for filename in filenames:
      pass

Bulging_Eyes = pathlib.Path('C:/Users/phill/Downloads/normal eye/Eye_diseases/Bulging_Eye')
Cataracts = pathlib.Path('C:/Users/phill/Downloads/normal eye/Eye_diseases/Glaucoma')
Crossed_Eyes = pathlib.Path('C:/Users/phill/Downloads/normal eye/Eye_diseases/Crossed_Eyes')
Glaucoma = pathlib.Path('C:/Users/phill/Downloads/normal eye/Eye_diseases/Cataracts')
Uveitis = pathlib.Path('C:/Users/phill/Downloads/normal eye/Eye_diseases/Uveitis')
Eye_defects = pathlib.Path('/content/drive/MyDrive/Eye_defect/Eye_defects')


# Define image dictionary
image_dic1= {"Bulging_Eyes":list(Bulging_Eyes.glob("*jpeg")),
            "Cataracts":list(Cataracts.glob("*jpeg")),
            "Crossed_Eyes":list(Crossed_Eyes.glob("*jpeg")),
            "Glaucoma":list(Glaucoma.glob("*jpeg")),
            "Uveitis":list(Uveitis.glob("*jpeg")),
            "Eye_defects":list(Eye_defects.glob("*jpeg"))
}

# Define labels dictionary
labels_dic1={
    "Bulging_Eyes":0,
    "Cataracts":1,
    "Crossed_Eyes":2,
    "Glaucoma":3,
    "Uveitis":4,
    "Eye_defects":5
}

# Define image dimensions
img_height, img_width = 105, 105

# preprocessing of data for trainig model
mode1 = []
mode2 = []
for label, img1 in image_dic1.items():
  for img in img1:
    img = cv2.imread(str(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.0
    mode1.append(img)
    mode2.append(labels_dic1[label])


# Convert lists to numpy arrays
mode1 = np.array(mode1)
mode2 = np.array(mode2)

#spliting dataset into training and testing sets
train_mode1,test_mode1,train_mode2,test_mode2 = train_test_split(mode1,mode2,test_size = 0.2)
train_mode1 = train_mode1.astype("float32")
test_mode1 = test_mode1.astype("float32")


# Define a function to build your model with tunable hyperparameters
def model_builder(hp):
    model = keras.Sequential([
    keras.layers.Input(shape=(img_height, img_width, 3)),

    layers.Conv2D(128, (2, 2), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),

    layers.Conv2D(32, (2, 2), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    keras.layers.GaussianNoise(0.1),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(50, activation='softmax'),
    layers.Dropout(0.2),
  # layers.BatchNormalization(),
])


    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    model.load_weights('//my_dir/tuner/trial_0', by_name=True, skip_mismatch=True)



#tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Initialize a BayesianOptimization tuner
tuner = kt.BayesianOptimization(
    model_builder,
    objective='val_accuracy',
    max_trials=5,

    directory='/drive/',
    project_name='tune_61'
)

# Search for the best hyperparameters
tuner.search(train_mode1, train_mode2, epochs=100, batch_size=0, validation_data=(test_mode1, test_mode2))
print("Search complete.")

# Get the best model
try:
    best_model = tuner.get_best_models(num_models=1)[0]
except Exception as e:
    print(f"Error: {e}")
    best_model = None


# Evaluate the best model
test_loss, test_acc = best_model.evaluate(test_mode1, test_mode2)
print(f'Test result: {test_acc:.5f}')
result = best_model.evaluate(test_mode1, test_mode2)
print(f'Test accuracy: {int(result[1]*100)}%')

#prediction class
predictions = best_model.predict(test_mode1)
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)

# Print classification report
print(classification_report(test_mode2, predicted_labels))

#print the summary
best_model.summary()
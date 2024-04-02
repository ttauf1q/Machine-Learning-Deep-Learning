from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

image_size = [244, 244]

vgg = VGG16(input_shape = image_size + [3], weights = 'imagenet', include_top =  False)
for layer in vgg.layers:
    layer.trainable = False
    
from glob import glob
folders = glob('D:/Deep Learning/Tomato Leaf Disease Detection/tomato/train')

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation = 'softmax')(x)

model = Model(inputs = vgg.input, outputs = prediction)

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories for training and testing images
train_dir = 'D:/Deep Learning/Tomato Leaf Disease Detection/tomato/train'
test_dir = 'D:/Deep Learning/Tomato Leaf Disease Detection/tomato/val'

# Define image dimensions and batch size
img_height, img_width = 224, 224  # Example dimensions
batch_size = 32

# Use ImageDataGenerator for loading and preprocessing images
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalizing pixel values
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training images from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(244, 244),
    batch_size=batch_size,
    class_mode='categorical')  # Assuming categorical labels

# Load testing images from directory
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(244, 244),
    batch_size=batch_size,
    class_mode='categorical')  # Assuming categorical labels

history = model.fit(
  train_generator,
  validation_data=test_generator,
  epochs=20,
  steps_per_epoch=len(train_generator),
  validation_steps=len(test_generator)
)


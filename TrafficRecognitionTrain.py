from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Activation, Flatten, Dense
from keras import utils
from numpy import array
from tensorflow.keras.optimizers import RMSprop
import cv2
import numpy as np

img_width, img_height = 32, 32
train_data_dir = 'Final_Training/Images'
validation_data_dir = 'Online-Test-sort'
test_images = 39209
validation_images = 12569
epochs = 20
batch_size = 16

model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.1),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.1),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.1),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(43, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc']
)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

model.fit_generator(
    train_generator,
    steps_per_epoch=test_images // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_images // batch_size
)

model.save('TrafficRecognitionModel_8.h5')
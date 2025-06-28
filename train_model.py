# train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
import os

# Configuration

DATA_DIR = "garbage-dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_PATH = "waste_model.h5"
LABELS_PATH = "labels.txt"

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class names
class_names = list(train_data.class_indices.keys())
with open(LABELS_PATH, "w") as f:
    f.write("\n".join(class_names))

# Build model using MobileNetV2
base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base.trainable = False
x = GlobalAveragePooling2D()(base.output)
output = Dense(len(class_names), activation="softmax")(x)
model = Model(inputs=base.input, outputs=output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

# Save model
model.save(MODEL_PATH)
print("✅ Model trained and saved successfully!")

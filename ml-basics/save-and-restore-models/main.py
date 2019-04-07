from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import feature_column
from tensorflow.keras import layers
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import os
print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
def create_model():
      model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
      ])
      model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
      return model
model = create_model()
model.summary()

# Save checkpoints during training
print("Checkpoint callback")
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)
model = create_model()
model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("Callback options")
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,period=5)
model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Manually save weights
print("Manually save weights")
model.save_weights('./checkpoints/my_checkpoint')
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Save the entire model
print("Save as HDF5")
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('my_model.h5')
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("Save as saved_model")
saved_model_path = "./saved_models/{}".format(int(time.time()))
tf.keras.experimental.export_saved_model(model, saved_model_path)
new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()
model.predict(test_images).shape
new_model.compile(optimizer=model.optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
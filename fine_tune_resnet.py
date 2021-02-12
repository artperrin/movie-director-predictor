# import the necessary packages
import logging

logging.getLogger().setLevel(logging.INFO)

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import tensorflow as tf
import config
import time
from model_builder import Mod
from model_builder import Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# to compute with GPU
gpus = tf.config.experimental.list_physical_devices(
    "GPU"
)

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-p",
    "--plot",
    type=str,
    default="plot.png",
    help="path to output loss/accuracy plot",
)
args = vars(ap.parse_args())

### Beginning of the program ###
start = time.time()
# initialize the learning rate, number of epochs to train for, and batch size
LR = config.LEARNING_RATE
EPOCHS = config.EPOCHS
BS = config.BATCH_SIZE

# initialize the data settings
path = config.TRN_DIR
size = config.IMG_SIZE
test_size = config.TEST_SIZE

# read the images and take the labels from the folder names
data = Dataset(path, size)
NbDir = data.read_data()
label_encoder = data.encode_data()
(trainX, testX, trainY, testY) = data.split_data(test_size)

# construct the training image generator for data augmentation
logging.info("Augmenting the data...")
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

# build and compile the model
model = Mod()
model.model_build(class_number=NbDir)
model = model.model_compile(LR)

# begin training
logging.info("Training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
)

# make predictions for tests
logging.info("Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# find the index of max probability
predIdxs = np.argmax(predIdxs, axis=1)
# show classification report
print(
    classification_report(
        testY.argmax(axis=1), predIdxs, target_names=label_encoder.classes_
    )
)

# serialize the model to disk
logging.info("Saving mask detector model...")
model.save("model.h5", save_format="h5")
# and serialize the label encoder to disk
logging.info("Saving label encoder...")
with open("encoder.pickle", "wb") as f:
    f.write(pickle.dumps(label_encoder))

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

### End of the program ###

print(f"Ended within {time.time()-start} seconds.")

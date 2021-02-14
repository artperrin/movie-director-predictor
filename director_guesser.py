from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import argparse
import config
import pickle
import cv2
import time
import logging
import os

logging.getLogger().setLevel(logging.INFO)

# to compute with GPU
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
	
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-s", "--save", default=None,
	help="path to prediction if one wants to save it")
args = vars(ap.parse_args())

outpath = args["save"]

### Beginning of the program ###
start = time.time()

# load the model and label binarizer from disk
logging.info("Loading model and label binarizer...")
model = load_model('model.h5')
lb = pickle.loads(open('encoder.pickle', "rb").read())

# load the input image from disk and pre-process it
image = cv2.imread(args["image"])
saved = image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, config.IMG_SIZE)
image = img_to_array(image)
image = preprocess_input(image)
image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

# make the prediction
proba = model.predict(image)

# find the resulting label
label = lb.classes_[np.argmax(proba, axis=1)]

# show the result
print(f'a movie directed by {label[0]}.')
logging.info(f"End of the prediction within {round(time.time()-start, 2)} seconds.")
X, Y = config.IMG_SIZE
X, Y = int(X/2), int(Y/2)
font = cv2.FONT_HERSHEY_SIMPLEX 
cv2.putText(saved, f"a movie directed by {label[0]}", (X,Y), font, 2, (255, 0, 0), 4) 
cv2.imshow('prediction (press any key to close)', saved)
cv2.waitKey(0)
cv2.destroyAllWindows()

if not outpath==None:
  outpath = os.path.join(outpath, label[0])
  cv2.imwrite(outpath+'.jpg', saved)
  logging.info(f"The prediction has been saved at directory {outpath}.")
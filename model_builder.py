import logging
import os
import numpy as np
from imutils import paths
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Mod:

    def model_build(self, class_number, input_shape=(224,224,3)):
        logging.info('Building the model...')
        # load the MobileNetV2 network, ensuring the head FC layer sets are left off
        baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
        # construct the head of the model that will be placed on top of the the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(class_number, activation="softmax")(headModel)
        # place the head FC model on top of the base model (this will become the actual model we will train)
        self.model = Model(inputs=baseModel.input, outputs=headModel)
        # loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
        for layer in baseModel.layers:
            layer.trainable = False

    def model_compile(self, learning_rate=1e-4):
        # compile our model
        logging.info("Compiling model...")
        opt = Adam(lr=learning_rate)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return self.model

class Dataset:

    def __init__(self, path, imgSize):
        logging.info("Loading images...")
        self.path = path
        self.lenPath = len(path)
        self.imagePaths = list(paths.list_images(path))
        self.size = imgSize
    
    def read_data(self):
        logging.info("Reading data...")
        self.data = []
        self.labels = []
        # loop over the image paths
        i = 0;
        end = len(self.imagePaths)
        for imagePath in self.imagePaths:
            # extract the class label from the filename
            label = imagePath.split(os.path.sep)[-2]
            label = label[self.lenPath:]
            image = load_img(imagePath, target_size=self.size)
            image = img_to_array(image)
            image = preprocess_input(image)
            self.data.append(image)
            self.labels.append(label)

            # visualize the progress
            progress = np.ceil(i / end * 25)  # progress from 0 to 25
            progress_line = ""
            for k in range(26):
                if k <= progress:
                    progress_line += "="
                else:
                    progress_line += "."
            print("Progress of reading : [" + progress_line + "]", end="\r", flush=True,)
            i+=1

        self.data = np.array(self.data, dtype="float32")
        self.labels = np.array(self.labels)
        print("End of reading.", end='\n')
        # return the number of directors
        return len(next(os.walk(self.path))[1])
    
    def encode_data(self):
        # perform one-hot encoding on the labels
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)
        self.labels = to_categorical(self.labels)
        return label_encoder
    
    def split_data(self, test_size=0.2):
        (trainX, testX, trainY, testY) = train_test_split(self.data, self.labels, test_size=test_size, stratify=self.labels)
        return (trainX, testX, trainY, testY)
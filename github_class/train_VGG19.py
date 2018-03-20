from keras.models import Model, Input
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.initializers import RandomNormal, Zeros
from keras.initializers import Zeros, RandomNormal
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, \
UpSampling2D, Activation, AveragePooling2D, merge, Flatten, Dense, Dropout
import numpy as np
from random import shuffle
import keras.utils
from keras import utils as np_utils
from src.data_prepare import data_preprocess

# define model hyperparameters
KERNEL_SIZE = 21
NUM_CLASSES = 3
LEARN_RATE = 10e-4
EPOCHS = 20
DECAY = LEARN_RATE/EPOCHS
IMG_SIZE = 128

# define input shape
input_shape = [
    128,
    128,
    1,
]


# define input shape
inputs = Input(shape = input_shape)

# first conv block
conv1 = Convolution2D(64, KERNEL_SIZE, padding="same", activation="relu")(inputs)
conv1 = Convolution2D(64, KERNEL_SIZE, padding="same", activation="relu")(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

KERNEL_SIZE = 3
# second conv block
conv2 = Convolution2D(128, KERNEL_SIZE, padding="same", activation="relu")(pool1)
conv2 = Convolution2D(128, KERNEL_SIZE, padding="same", activation="relu")(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

# third conv block
conv3 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu")(pool2)
conv3 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu")(conv3)
conv3 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu")(conv3)
conv3 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu")(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

# fourth conv block
conv4 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu")(pool3)
conv4 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu")(conv4)
conv4 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu")(conv4)
conv4 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu")(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)

# fourth conv block
conv5_0 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu")(pool4)
conv5_1 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu")(conv5_0)
conv5_2 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu")(conv5_1)
conv5_3 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu")(conv5_2)
pool5 = MaxPooling2D(pool_size=(2, 2))(conv5_3)


# fully connected layer
flat = Flatten(name = 'flatten')(pool5)
fc = Dense(1024, activation="relu",name = 'fc1')(flat)  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
fc = Dropout(0.5)(fc)
fc = Dense(1024, activation="relu", name = 'fc2')(fc)  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
fc = Dropout(0.5)(fc)


pred = Dense(NUM_CLASSES, activation = "softmax", name = 'predictions')(fc)

# define model and compile
model = Model(inputs=inputs, outputs=pred, name='vgg19')

model.compile(
    loss = categorical_crossentropy,
    optimizer=SGD(
            lr=LEARN_RATE,
            decay=DECAY,
            momentum=0.1,
    ),
    metrics = ['accuracy']
)


# data load
'''
training_data = np.load('/data/train_data_all.npy')
test_data = np.load('/data/test_data_all.npy')
X = np.array([i[0] for i in training_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in training_data]
Y = np.array(Y)
print('train set is done!')
test_x = np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test_data]
test_y = np.array(test_y)
'''

train,test = data_preprocess(X,test_x)

# fit the models
model.fit(train, Y, epochs=EPOCHS, verbose=1, validation_data = (test,test_y), batch_size = 8, shuffle = True)

# save the model
model.save('.../train_VGG19.h5')

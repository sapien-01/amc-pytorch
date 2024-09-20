import time
import numpy as np
import sys
import random
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Input
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from util import *

# Set constants
train_path = sys.argv[1]
test_path = sys.argv[2]
resolution = 36
tmp = train_path.split('_')
tmp = tmp[2].split('.')
SNR = tmp[0]
R = 0   # Read Polar Model
duplicate = 0

# Wrap the training function in tf.function to enable graph mode
@tf.function
def train(model, x_train, y_train, x_val, y_val, x_test, y_test):
    start_time = time.time()
    result = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=100, epochs=20, shuffle=True, verbose=1)
    print("--- Train Time: %s seconds ---" % (time.time() - start_time))

    # Load best weights
    model.load_weights('model.h5')
    start_time = time.time()
    y_pred = model.predict(x_test)
    print("--- Test Time: %s seconds ---" % (time.time() - start_time))

    # Evaluate model
    scores = model.evaluate(x_test, y_test)
    print('Scores: ', scores)
    return model

def build_model(input_shape, t):
    # CNN Model Architecture
    cnn_input = Input(shape=input_shape)
    cnn_batch = BatchNormalization()(cnn_input)
    conv1 = Convolution2D(2 * t, (3, 3), padding='same', activation='relu')(cnn_batch)
    max1 = MaxPooling2D((3, 3))(conv1)
    bat1 = BatchNormalization()(max1)
    conv2 = Convolution2D(t, (3, 3), padding='same', activation='relu')(bat1)
    max2 = MaxPooling2D((3, 3))(conv2)
    flat = Flatten()(max2)
    den1 = Dense(2 * t, activation='relu')(flat)
    den1 = BatchNormalization()(den1)
    den2 = Dense(t, activation='relu')(den1)
    den2 = BatchNormalization()(den2)
    den3 = Dense(4, activation='softmax')(den2)
    model = Model(cnn_input, den3)
    
    # Compile Model
    sgd = SGD(learning_rate=0.1, weight_decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    # model.summary()
    return model

def main():
    t = int(2 ** int(sys.argv[3]))
    
    # Load and preprocess the data
    (x_train, y_train) = load_mat(train_path, 0)
    (x_train, y_train), (x_val, y_val) = split_data(x_train, y_train, 0.2)
    
    x_train = sig2pic(x_train, -3, 3, resolution)
    x_val = sig2pic(x_val, -3, 3, resolution)
    (x_test, y_test) = load_mat(test_path, 0)
    x_test = sig2pic(x_test, -3, 3, resolution)
    
    # Convert to tf.data.Dataset for performance
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(100)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(100)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

    # Build and train model
    model = build_model((36, 36, 1), t)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_acc', patience=8, mode='max')
    checkpoint = ModelCheckpoint(filepath='model.keras', save_best_only=True, monitor='val_acc', mode='max')

    # Train model
    model = train(model, train_dataset, val_dataset, test_dataset)

    # Save model structure
    model_json = model.to_json()
    with open("Model/cnn.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(f"Model/cnn_{SNR}.h5")

    # Test model
    for i in range(4):
        n = int(x_test.shape[0] / 4)
        scores = model.evaluate(x_test[n * i:n * (i + 1), :], y_test[n * i:n * (i + 1), :])
        print('Scores: ', scores)

if __name__ == "__main__":
    main()


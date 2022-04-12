# Convolutional Neural Network
# Importing the libraries

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

# Preprocessing the Training set

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('datasets/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Preprocessing the Test set

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('datasets/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


# Part 2 - Building the CNN

def build_train_model():
    model = tf.keras.models.Sequential()

    #  adding Convolution layer

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

    # adding  Pooling layer

    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Adding a second convolutional layer

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    #  Flattening

    model.add(tf.keras.layers.Flatten())

    #  Full Connection

    model.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Output Layer

    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Training the CNN

    # Compiling the CNN

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the CNN

    model.fit(x=training_set, validation_data=test_set, epochs=25)
    return model


cnn = build_train_model()


# Making a single prediction
def predict_sample(path):
    test_image = image.load_img(path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    print(training_set.class_indices)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    return prediction


path = 'datasets/single_prediction/cat_or_dog_1.jpg'
prediction=predict_sample(path)
print(prediction)

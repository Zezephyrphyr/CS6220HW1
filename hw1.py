#The code refers to 
#https://github.com/CodeRed1704/Cat-Vs-Dog-Keras-Tensorflow
#https://medium.com/@parthvadhadiya424/hello-world-program-in-keras-with-cnn-dog-vs-cat-classification-efc6f0da3cc5
#Thanks to their help. 

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Initialising the CNN
model = Sequential()

# Convolution
model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))

# Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


#Adding third convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

#Adding fourth convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())

#Full connection
model.add(Dense(units = 128, activation = 'relu'))

model.add(Dense(units = 128, activation = 'relu'))

model.add(Dense(units = 128, activation = 'relu'))

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('../../../../OneDrive/Fall2018/BigData/HW1/dataset/train',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

model.fit_generator(training_set,
                         steps_per_epoch = 500,
                         epochs = 5,
                         validation_steps = 100)

test_image_1 = image.load_img('../../../../OneDrive/Fall2018/BigData/HW1/dataset/test/cat.588.jpg', target_size = (50, 50))
test_image_1 = image.img_to_array(test_image_1)
test_image_1 = np.expand_dims(test_image_1, axis = 0)
result = model.predict(test_image_1)
print "Test Cat Picture One is: "
print result[0][0]


test_image_2 = image.load_img('../../../../OneDrive/Fall2018/BigData/HW1/dataset/test/dog.11147.jpg', target_size = (50, 50))
test_image_2 = image.img_to_array(test_image_2)
test_image_2 = np.expand_dims(test_image_2, axis = 0)
result2 = model.predict(test_image_2)
print "Test Dog Picture Two is: "
print result2[0][0]

test_image_3 = image.load_img('../../../../OneDrive/Fall2018/BigData/HW1/dataset/test/cat.4694.jpg', target_size = (50, 50))
test_image_3 = image.img_to_array(test_image_3)
test_image_3 = np.expand_dims(test_image_3, axis = 0)
result3 = model.predict(test_image_3)
print "Test Cat Picture Three is: "
print result3[0][0]

test_image_4 = image.load_img('../../../../OneDrive/Fall2018/BigData/HW1/dataset/test/dog.3315.jpg', target_size = (50, 50))
test_image_4 = image.img_to_array(test_image_4)
test_image_4 = np.expand_dims(test_image_4, axis = 0)
result4 = model.predict(test_image_4)
print "Test Dog Picture Four is: "
print result4[0][0]

print(training_set.class_indices)

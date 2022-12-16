# Dataset MNIST: Handwritten letters, 28x28 gray images for 10 digits, 60K training and 10K test images
# 1D fully connected neural network
# Input layer: 28x28 = 784 neurons
# 1 hidden layer: experimental 1024 neurons and sigmoid activation function
# Output layer: 10 neurons for 10 digits (0-9) and softmax activation function
# Optimiser: Stochastic gradient descent
# Loss: Categorical cross entropy
# Batch: 128 minibatches, 60 epochs

import keras.utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt

# Loading dataset and vectorizing them
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

# Generate class labels in one-hot coding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Defining network
# Model: Sequential (Feed Forward without Feedback)
model = Sequential()

#model.add(Dense(1024, input_shape=(784, ), activation='relu'))   #accuracy=0.11
#model.add(Dense(1024, input_shape=(784, ), activation='softmax'))  #accuracy=0.39
#model.add(Dense(1024, input_shape=(784, ), activation='tanh'))  #accuracy=0.81

# sigmoid = loss: 0.1111 - accuracy: 0.9650 - val_loss: 0.1393 - val_accuracy: 0.9589 (best)
model.add(Dense(1024, input_shape=(784, ), activation='sigmoid'))   #accuracy=0.96
model.add(Dense(10, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.1, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=60, verbose=1, validation_data=(x_test, y_test))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
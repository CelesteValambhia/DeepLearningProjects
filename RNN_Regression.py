# ---------- Recurrent Neural Network: ----------
# One way to deal with problem of having different amount of input values is by RNN. Examples include: Speech
# recognition, language translation, stock prediction, image recognition to describe content in pictures etc. RNNs
# are good for predicting sequential data. RNN has a looping mechanism that acts as a highway to allow information to
# flow from one step to the next. This information is a hidden state which is a representation of previous inputs.

# ---------- Weather prediction with Recurrent Neural Networks ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import Callback

# humidity = pd.read_csv("./weather-dataset/humidity.csv")
temp = pd.read_csv("./weather-dataset/temperature.csv")
# pressure = pd.read_csv("./weather-dataset/pressure.csv")

# humidity_SF = humidity[['datetime', 'San Francisco']]
temp_SF = temp[['datetime', 'San Francisco']]
# pressure_SF = pressure[['datetime', 'San Francisco']]

# Dropping NaN values
pd.set_option('mode.chained_assignment', None) # ignore warnings
# humidity_SF.dropna(inplace=True)
temp_SF.dropna(inplace=True)
# pressure_SF.dropna(inplace=True)

# print(humidity_SF.shape)
print(temp_SF.shape)
# print(pressure_SF.shape)

def plot_train_points(quantity, datapoints):
    plt.figure(figsize=(15,4))
    if quantity == 'humidity':
        plt.title("Humidity of first {} data points".format(datapoints))
        plt.plot(humidity_SF['San Francisco'].iloc[:datapoints], c='k', lw=1)
    if quantity == 'temperature':
        plt.title("Temperature of first {} data points".format(datapoints))
        plt.plot(temp_SF['San Francisco'].iloc[:datapoints], c='k', lw=1)
    if quantity == 'pressure':
        plt.title("Pressure of first {} data points".format(datapoints))
        plt.plot(pressure_SF['San Francisco'].iloc[:datapoints], c='k', lw=1)
    plt.grid(True)
    plt.show()


# We only use first 22000 datapoints not all
datapoints = 22000
#plot_train_points('temperature', datapoints)

# Split training and testing dataset
train = np.array(temp_SF['San Francisco'].iloc[:datapoints])
test = np.array(temp_SF['San Francisco'].iloc[datapoints:])
print("Train data length:", train.shape)
print("Test data length:", test.shape)

train = train.reshape(-1, 1)
test = test.reshape(-1, 1)

# Here, we choose step=9. In more complex RNN and in particular for text processing, this is also called embedding size.
# The idea here is that we are assuming that 9 hours of weather data can effectively predict the 10th hour data...
step = 15
# add step elements into train and test
test = np.append(test, np.repeat(test[-1, ], step))
train = np.append(train, np.repeat(train[-1, ], step))
print("Train data length:", train.shape)
print("Test data length:", test.shape)

# convert test and train data into the matrix with step value
# basically if we have X.T=[1 2 3 4 5 6 7 8 9 10]
# and if step = 3
# then after conversion we get
# trainX.T = [[1 2 3] [2 3 4] [3 4 5] ... and so on]
def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i+step
        X.append(data[i:d, ])
        Y.append(data[d, ])
    return np.array(X), np.array(Y)


trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)
print("Training data shape:", trainX.shape, ', ', trainY.shape)
print("Test data shape:", testX.shape, ', ', testY.shape)

# When you're dealing with RNNs your dataset should have a shape that looks like
# (nb_samples, nb_timesteps, nb_features) Translating this to your use case means that each account is a sample
# (what you'll iterate when doing mini-batching), each week is a timestep (what your rnn will iterate over)
# and your features are.... Features.
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print("Training data shape:", trainX.shape, ', ', trainY.shape)
print("Test data shape:", testX.shape, ', ', testY.shape)

# We build a simple function to define the RNN model. It uses a single neuron for the output layer because we are
# predicting a real-valued number here. As activation, it uses the ReLU function. Following arguments are supported.
# neurons in the RNN layer
# embedding length (i.e. the step length we chose)
# nenurons in the densely connected layer
# learning rate
def build_simple_rnn(num_units, embedding, num_dense, lr):
    """
    Builds and compiles a simple RNN model
    Arguments:
              num_units: Number of units of a the simple RNN layer
              embedding: Embedding length
              num_dense: Number of neurons in the dense layer followed by the RNN layer
              lr: Learning rate (uses RMSprop optimizer)
    Returns:
              A compiled Keras model.
    """
    model = Sequential()
    model.add(SimpleRNN(units=num_units, input_shape=(1, embedding), activation="linear"))
    model.add(Dense(num_dense, activation="linear"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=lr))
    return model

def simple_LSTM():
    # for LSTM The three dimensions of this input are:
    # Samples. One sequence is one sample. A batch is comprised of one or more samples.
    # Time Steps. One time step is one point of observation in the sample.
    # Features. One feature is one observation at a time step.
    model = Sequential()
    model.add(LSTM(128, input_shape=(1, step), activation='linear', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dense(1))
    optimizer = RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


model_temperature = build_simple_rnn(num_units=128, num_dense=32, embedding=step, lr=0.0005)
model_temperature.summary()

batch_size = 8
num_epochs = 1000

history = model_temperature.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, verbose=1)

plt.figure(figsize=(7,5))
plt.title("RMSE loss over epochs",fontsize=16)
plt.plot(np.sqrt(model_temperature.history.history['loss']),c='k',lw=2)
plt.grid(True)
plt.xlabel("Epochs",fontsize=14)
plt.ylabel("Root-mean-squared error",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

trainPredict = model_temperature.predict(trainX)
testPredict = model_temperature.predict(testX)
predicted = np.concatenate((trainPredict, testPredict), axis=0)

#plotting ground truth vs prediction
index = temp_SF.index.values
plt.figure(figsize=(15, 5))
plt.title("Temperature: Ground truth and prediction together", fontsize=18)
plt.plot(index, temp_SF['San Francisco'], c='blue')
plt.plot(index, predicted, c='orange', alpha=0.75)
plt.legend(['True data', 'Predicted'], fontsize=15)
plt.axvline(x=datapoints, c="r")
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# LSTM
lstm_temp = simple_LSTM()
lstm_temp.summary()
batch_size = 8
num_epochs = 1000
history_lstm = lstm_temp.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, verbose=1)

plt.figure(figsize=(7, 5))
plt.title("RMSE loss over epochs", fontsize=16)
plt.plot(np.sqrt(lstm_temp.history_lstm.history_lstm['loss']), c='k', lw=2)
plt.grid(True)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Root-mean-squared error", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

trainPredict = lstm_temp.predict(trainX)
testPredict = lstm_temp.predict(testX)
predicted = np.concatenate((trainPredict, testPredict), axis=0)

#plotting ground truth vs prediction
index = temp_SF.index.values
plt.figure(figsize=(15, 5))
plt.title("Temperature: Ground truth and prediction together", fontsize=18)
plt.plot(index, temp_SF['San Francisco'], c='blue')
plt.plot(index, predicted, c='orange', alpha=0.75)
plt.legend(['True data', 'Predicted'], fontsize=15)
plt.axvline(x=datapoints, c="r")
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

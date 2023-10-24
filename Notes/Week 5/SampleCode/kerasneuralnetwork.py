import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import activations
from tensorflow import keras

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# construct a 2-2-1 neural net
model = Sequential()
model.add(Dense(2, input_dim=2, activation=activations.sigmoid))
model.add(Dense(1, activation=activations.sigmoid))

opt = keras.optimizers.SGD(learning_rate=0.2)
model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['binary_accuracy'])
model.fit(X, y, epochs=20000, verbose=2)

print(model.predict(X).round())
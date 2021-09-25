import matplotlib.pyplot as plt

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])

print(X_train[0].shape)

# reshape data to fit model
X_train = X_train.reshape(60000, 28, 28, 1) # 60000 images, 28x28 shape, grayscale
X_test = X_test.reshape(10000, 28, 28, 1)

# one-hot encode target col (convert to numerical data)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# creating model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))  # 64 nodes
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten()) # connects convolution and dense layers
model.add(Dense(10, activation='softmax')) # output layer, 10 nodes for output

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
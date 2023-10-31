# import the necessary packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        # second conv layer
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        model.add(Conv2D(16, (3,3), padding="same"))
        model.add(Activation("relu"))

        # second conv layer
        model.add(Conv2D(8, (3,3), padding="same"))
        model.add(Activation("relu"))

        # second conv layer
        model.add(Conv2D(8, (3,3), padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(4, (3,3), padding="same"))
        model.add(Activation("relu"))


        # softmax classifier
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        model.summary()
        return model


    
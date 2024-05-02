from tensorflow.keras.layers import *
from tensorflow.keras import Sequential



#모델 구조 지정
def get_model(p_input_shape):
    model = Sequential()
    model.add(Dense(units=4000, activation='relu', input_shape=p_input_shape))
    #model.add(Dropout(0.3))
    model.add(Dense(units=7, activation='softmax'))
    return model

#모델 구조 지정
def get_model_gen(p_input_shape):
    model = Sequential()
    model.add(Dense(units=5184, activation='relu', input_shape=p_input_shape))
    model.add(BatchNormalization())
    model.add(Reshape((9,9,64)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2,2)))
    #model.add(Conv2D(filters=16, kernel_size = 3, padding='same', activation='relu'))
    #model.add(BatchNormalization())
    #model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'))
    #CNN
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3,3)))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))
    #model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    #model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))
    return model



#모델 구조 지정
def get_model_gen_v2(p_input_shape):
    model = Sequential()
    model.add(Dense(units=1568, activation='relu', input_shape=p_input_shape))
    model.add(BatchNormalization())
    model.add(Reshape((7,7,32)))

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'))

    #CNN
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3,3)))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(units=300, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    return model
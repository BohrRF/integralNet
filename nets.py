import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
from keras import backend as k
from keras import optimizers
from keras import activations
from keras.models import Model, Sequential
from keras.layers import Input, Activation, BatchNormalization, Dropout, Flatten, Dense
from keras.layers.merge import add, multiply, concatenate, maximum, average
from keras.utils import plot_model
import lossplt
from sklearn.model_selection import train_test_split

train = []

train_x, train_y, train_x_e = [], [], []
val_x, val_x_e, val_y = [], [], []
for x in np.linspace(-np.pi, np.pi, 1000):
    train.append((x, 1, 0.65 * np.sin(2.6 * x) + 0.35 * np.cos(4.7 * x - 8.3)))

data_t_0, data_v_0 = train_test_split(train, test_size=0.05)
for i in data_t_0:
    train_x.append(i[0])
    train_x_e.append(i[1])
    train_y.append(i[2])

for i in data_v_0:
    val_x.append(i[0])
    val_x_e.append(i[1])
    val_y.append(i[2])

@tf.function
def swish(x):
    return x * activations.sigmoid(x)

@tf.function
def swish_d(x):
    return swish(x) + activations.sigmoid(x) * (1 - swish(x))


def relu_d(x):
    return tf.cond(tf.greater(x, 0.0), lambda: 1.0, lambda: 0.0)

def relu(x):
    return tf.maximum(x, 0.0)

def acti(x):
    if x == "sine":
        return [k.sin, k.cos]
    elif x == "swish":
        return [swish, swish_d]
    elif x == "relu":
        return [relu, relu_d]


input = Input(shape=(1,), name="input")
input_e = Input(shape=(1,), name="input_e")

activation = "swish"
activation0 = Activation(acti(activation)[0], name="activation0")
activation0d = Activation(acti(activation)[1], name="activation0d")
activation1 = Activation(acti(activation)[0], name="activation1")
activation1d = Activation(acti(activation)[1], name="activation1d")
activation2 = Activation(acti(activation)[0], name="activation2")
activation2d = Activation(acti(activation)[1], name="activation2d")




dense0 = Dense(256, name="dense0")
dense1 = Dense(256, name="dense1")
dense2 = Dense(256, name="dense2")
dense3 = Dense(1, name="dense3")


multiply0_input0 = dense0(input_e)
multiply0_input1 = activation0d(dense0(input))
multiply0 = multiply([multiply0_input0, multiply0_input1])

multiply1_input0 = dense1(multiply0)
multiply1_input1 = activation1d(dense1(activation0(dense0(input))))
multiply1 = multiply([multiply1_input0, multiply1_input1])

multiply2_input0 = dense2(multiply1)
multiply2_input1 = activation2d(dense2(activation1(dense1(activation0(dense0(input))))))
multiply2 = multiply([multiply2_input0, multiply2_input1])

output = dense3(multiply2)

GradNet = Model(inputs=[input, input_e], outputs=output)
plot_model(GradNet, to_file="GradNet.png", show_shapes=True, show_layer_names=True)

GradNet.compile(loss="mean_squared_error",
                optimizer=optimizers.RMSprop(lr=0.001))

higher_better_metrics = ['acc']
visualize_cb = lossplt.LearningVisualizationCallback(higher_better_metrics)
callbacks = [visualize_cb, ]

history = GradNet.fit([train_x, train_x_e],
                      train_y,
                      validation_data=([val_x, val_x_e], val_y),
                      epochs=1000,
                      batch_size=200,
                      callbacks=callbacks)



IntNet = Sequential()

IntNet.add(GradNet.get_layer("input"))
IntNet.add(GradNet.get_layer("dense0"))
IntNet.add(activation0)
IntNet.add(GradNet.get_layer("dense1"))
IntNet.add(activation1)
IntNet.add(GradNet.get_layer("dense2"))
IntNet.add(activation2)
IntNet.add(GradNet.get_layer("dense3"))

plot_model(IntNet, to_file="IntNet.png", show_shapes=True, show_layer_names=True)
IntNet.compile(optimizer="adam", loss="mean_squared_error")



fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharex="col", sharey="row")


test_x = np.linspace(-np.pi, np.pi, 1000)
test_x_e = np.ones(1000)
test_y = 0.65 * np.sin(2.6 * test_x) + 0.35 * np.cos(4.7 * test_x - 8.3)


axes[0].plot(test_x, test_y, label="GT")

grad = GradNet.predict({"input":test_x, "input_e":test_x_e})
axes[0].plot(test_x, grad, label="GradNet")
axes[0].set_title("function")
axes[0].legend()


def func(x):
    return (-0.65/2.6)*np.cos(2.6*x) + (0.35/4.7)*np.sin(4.7*x-8.3)
gt = func(test_x)
axes[1].plot(test_x, gt - func(0), label="GT")


a = IntNet.predict(test_x)
axes[1].plot(test_x, a - IntNet.predict(np.array([0])), label="IntNet")
axes[1].set_title("integral")
axes[1].legend()

plt.show()


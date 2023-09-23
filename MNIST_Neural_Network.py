"""
Artificial Neural Network MNIST dataset

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

train_path = 'mnist_train.csv'
test_path = 'mnist_test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Preprocessing
x_train = tf.constant(train_data.iloc[:, 1:].values.astype(np.float32))
y_train = tf.constant(pd.get_dummies(train_data.iloc[:, 0]).values.astype(np.float32))  # One-hot encode the labels

x_test = tf.constant(test_data.iloc[:, 1:].values.astype(np.float32))
y_test = tf.constant(test_data.iloc[:, 0].values.astype(np.float32))


y_train_max = tf.reduce_max(y_train, axis=1)


# Neural Network Configuration
input_layer = x_train.shape[1]
hidden_layer1 = 32
hidden_layer2 = 48
hidden_layer3 = 32
output_layer = 10


# Initialize the weights using TensorFlow
w1 = tf.random.normal((input_layer, hidden_layer1), stddev=0.01)
w2 = tf.random.normal((hidden_layer1, hidden_layer2), stddev=0.01)
w3 = tf.random.normal((hidden_layer2, hidden_layer3), stddev=0.01)
w4 = tf.random.normal((hidden_layer3, output_layer), stddev=0.01)


def relu(x):
    return tf.nn.relu(x)


def softmax(x):
    return tf.nn.softmax(x)


def relu_derivative(x):
    return tf.cast(x > 0, dtype=tf.float32)


def forward_propagation(x, w1, w2, w3, w4):
    z1 = tf.matmul(x, w1)
    a1 = relu(z1)
    z2 = tf.matmul(a1, w2)
    a2 = relu(z2)
    z3 = tf.matmul(a2, w3)
    a3 = relu(z3)
    out = tf.matmul(a3, w4)
    y_ = softmax(out)
    return y_, z1, a1, z2, a2, z3, a3



def back_propagation(x, y, y_pred, z1, a1, z2, a2, z3, a3):
    dloss = 2 * (y_pred - y) / tf.cast(tf.shape(y)[0], tf.float32)
    dw4 = tf.matmul(tf.transpose(a3), dloss)
    da3 = tf.matmul(dloss, tf.transpose(w4))
    dz3 = tf.multiply(da3, relu_derivative(a3))
    dw3 = tf.matmul(tf.transpose(a2), dz3)
    da2 = tf.matmul(dz3, tf.transpose(w3))
    dz2 = tf.multiply(da2, relu_derivative(a2))
    dw2 = tf.matmul(tf.transpose(a1), dz2)
    da1 = tf.matmul(dz2, tf.transpose(w2))
    dz1 = tf.multiply(da1, relu_derivative(a1))
    dw1 = tf.matmul(tf.transpose(x), dz1)
    return dw1, dw2, dw3, dw4


def update_weight(w1, w2, w3, w4, dw1, dw2, dw3, dw4, learning_rate):
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    w3 -= learning_rate * dw3
    w4 -= learning_rate * dw4
    return w1, w2, w3, w4


num_iterations = 10000
learning_rate = 0.001

mse = tf.keras.losses.MeanSquaredError()

train_losses = tf.constant([])
test_losses = tf.constant([])

# Training loop
for i in range(num_iterations):
    y_train_pred, z1, a1, z2, a2, z3, a3 = forward_propagation(x_train, w1, w2, w3, w4)

    dw1, dw2, dw3, dw4 = back_propagation(x_train, y_train, y_train_pred, z1, a1, z2, a2, z3, a3)

    w1, w2, w3, w4 = update_weight(w1, w2, w3, w4, dw1, dw2, dw3, dw4, learning_rate)

    y_train_pred_max = tf.reduce_max(y_train_pred, axis=1)
    training_loss = mse(1,y_train_pred_max).numpy()

    train_losses = tf.concat([train_losses, [training_loss]], axis=0)
    
    y_test_pred, _, _, _, _, _, _ = forward_propagation(x_test, w1, w2, w3, w4)
    y_test_pred_max = tf.reduce_max(y_test_pred, axis=1)
    testing_loss = mse(1,y_test_pred_max).numpy()

    test_losses = tf.concat([test_losses, [testing_loss]], axis=0)

    if i%10 ==0 :
        loss = train_losses[-1]
        accuracy = 1 - loss
        print('loss = ', loss, '\t', 'accuracy = ', accuracy)


np.save('model_weights.npy', [w1, w2, w3, w4])

iterations = np.arange(1, num_iterations + 1)
plt.plot(iterations, train_losses, label='Train Loss')
plt.plot(iterations, test_losses, label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()



import tensorflow as tf
import numpy as np
# grader-required-cell
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# GRADED FUNCTION: house_model
def house_model():
    ### START CODE HERE

    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember to explictly set the dtype as float
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    # Define your model (should be a model with 1 dense layer and 1 unit)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train your model for 1000 epochs by feeding the i/o tensors
    history=model.fit(xs, ys, epochs=1000)

    ### END CODE HERE
    return model,history


def print_loss(history):
    loss = history.history['loss']

    epochs = range(len(loss))

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)

    plt.show()

def save_model(model):
    model.save('C:\Progetti\Personali\MachineLearning\Basic\Cousera\Housing_price\model_saved')
model,history = house_model()
new_x = 7.0
prediction = model.predict([new_x])[0]

print(prediction)
print_loss(history)
save_model(model)


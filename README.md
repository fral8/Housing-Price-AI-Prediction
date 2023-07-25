# Housing Price AI Regressor

## Overview

This repository contains code for a simple AI Regressor that predicts housing prices based on the number of bedrooms. The model is built using TensorFlow and Keras and utilizes a single dense layer with one unit. The model is trained using Stochastic Gradient Descent (SGD) optimizer and Mean Squared Error (MSE) as the loss function.

## Dataset

The dataset used for training and testing the Housing Price AI Regressor consists of sample data for houses with 1 up to 6 bedrooms. The input data (number of bedrooms) is stored in the `xs` array, and the corresponding output data (housing prices) is stored in the `ys` array.

## Model Architecture

The AI Regressor model consists of one dense layer with one unit. The input shape is [1], which corresponds to the number of bedrooms as the only feature for prediction.

## Training

The model is trained for 1000 epochs using the training data. The training process involves minimizing the Mean Squared Error loss by adjusting the model parameters with the SGD optimizer.

## Prediction

After training, the model is used to predict the housing price for a new input, which is a house with 7 bedrooms. The predicted price is displayed in the output.

## Training Performance

The training loss over the epochs is plotted to visualize the model's performance during training.

## Saving the Model

After training, the model is saved to a file named "model_saved" in the specified directory.

## Usage

To use this AI Regressor for predicting housing prices based on the number of bedrooms, simply load the saved model and call the `model.predict([new_x])` method, where `new_x` is the number of bedrooms for which you want to predict the price.

Feel free to modify the code, extend the dataset, or experiment with different hyperparameters to enhance the model's prediction capabilities.

For any questions or suggestions, please contact [Francesco Alotto](mailto:franalotto94@gmail.com). Happy coding! 😊


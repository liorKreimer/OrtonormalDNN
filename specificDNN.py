
import numpy as np
import random
np.random.seed()
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models,callbacks
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import BatchNormalization,Dropout
from keras.models import Sequential
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import SGD,Adam,Adagrad
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.losses import sparse_categorical_crossentropy
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.constraints import UnitNorm



# Set the global random seed for TensorFlow
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)


# load data
orthonormal_data_new_with_labels = pd.read_csv('orthonormal_data_sum1.csv')
#orthonormal_data_new_with_labels['Labels'] = np.where(orthonormal_data_new_with_labels['Labels'] == -1, 0, orthonormal_data_new_with_labels['Labels'])
labels = orthonormal_data_new_with_labels['Labels']
X_variables = orthonormal_data_new_with_labels.drop('Labels', axis=1)
# Scale or normalize features to have consistent ranges
scaler = StandardScaler()
X_variables = scaler.fit_transform(X_variables)
X_variables = np.array(X_variables)

# iterating over each feature in your dataset,
# performing the Shapiro-Wilk test,
# and printing the index of the features that reject the null hypothesis (i.e., the ones that are not normally distributed)
# Initialize a list to store the indices of non-normally distributed features
non_normal_features = []

# Assuming your dataset is stored in X_train
for feature_index in range(X_variables.shape[1]):
    # Extract the specific feature
    feature_data = X_variables[:, feature_index]

    # Perform Shapiro-Wilk test
    stat, p_value = shapiro(feature_data)

    # Set significance level
    alpha = 0.05

    # Check if null hypothesis is rejected
    if p_value < alpha:
        print(f'Feature {feature_index} is not normally distributed (p-value: {p_value:.4f})')
        non_normal_features.append(feature_index)

# Remove non-normally distributed features from the dataset
X_variables = np.delete(X_variables, non_normal_features, axis=1)
# get the new shape of my data set
num_features = X_variables.shape[1]
print(num_features)

# I have a dataset I can start testing it with machine learning models
# test/train
X_train, X_test, y_train, y_test = train_test_split(X_variables, labels, test_size=0.2, random_state=42)

# plot to see balance labels
plt.hist(y_train)
plt.ylabel('output labels of train set ')
plt.show()
plt.hist(y_test)
plt.ylabel('output labels of validation set ')
plt.show()


# calculate the mean and standard deviation of each feature in the train dataset
mean_values = np.mean(X_train, axis=0)
std_dev_values = np.std(X_train, axis=0)

# Normalize the training set using mean and std of training set
X_train = (X_train - mean_values) / std_dev_values
# Normalize the validation set using mean and std of training set
X_test = (X_test - mean_values) / std_dev_values


# building loss functions
def hinge_loss(y_true, y_pred):
    return tf.maximum(0., 1. - tf.cast(y_true, dtype=tf.float32) * y_pred)


# building activation functions
def shifted_hard_sigmoid(x):
    return tf.maximum(-1.0, tf.minimum(1.0, x))


def tanH(x):
    return tf.math.tanh(x)


def leaky_ReLU(x):
    return tf.maximum(0.2 * x, x)


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smooth approximation of the rectifier function and is used
    to introduce non-linearity in the neural network.
    """
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


# preform decay learning rate
def step_decay(epoch, current_learning_rate):
    if epoch < 5:
        return current_learning_rate
    else:
        return current_learning_rate * 0.5  # Adjust this multiplier as needed


class DegenerateBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3):
        super(DegenerateBatchNormalization, self).__init__()
        self.epsilon = epsilon
        self.bn_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        normalized_inputs = self.bn_layer(inputs, training=training)
        # Handle degenerate case where variance is too small
        normalized_inputs = tf.where(tf.math.is_nan(normalized_inputs), tf.zeros_like(normalized_inputs), normalized_inputs)
        return normalized_inputs


class BoundedBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3, clip_value=1.0):
        super(BoundedBatchNormalization, self).__init__()
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.bn_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        normalized_inputs = self.bn_layer(inputs, training=training)
        clipped_inputs = tf.clip_by_value(normalized_inputs, -self.clip_value, self.clip_value)
        return clipped_inputs


# Custom regularization layer for Singular Value Bounding (SVB)
class SingularValueBounding(layers.Layer):
    def __init__(self, epsilon=0.01):
        super(SingularValueBounding, self).__init__()
        self.epsilon = epsilon

    def call(self, inputs):
        # Compute SVD of weight matrix
        s, u, v = tf.linalg.svd(inputs)
        # Compute spectral normalization factor
        max_singular_value = tf.reduce_max(s)
        scale = 1. / (max_singular_value + self.epsilon)
        # Apply spectral normalization
        w_normalized = inputs * scale
        return w_normalized

# visualize the impact of adding hidden layers on the accuracy - find optimal deep - average result on num_runs
# Number of runs for each model configuration
num_runs = 5
num_epochs = 50

# List to store accuracy results for this model configuration
accuracy_results = []
# List to store accuracy results for number of epochs
accuracy_results_epochs = []
# Store validation accuracy for each epoch
validation_accuracy_results_epochs = []
# Store training loss for this model configuration
loss_results_epochs = []
# Store validation loss for this model configuration
validation_loss_results_epochs = []
gradients_first_layer = []
gradients_last_layer = []

# Repeat the training and evaluation process for each configuration multiple times
for _ in range(num_runs):
    # Build the model
    model = Sequential()
    # add first layer
    model.add(layers.Dense(num_features,input_dim=num_features, activation=shifted_hard_sigmoid, kernel_initializer='glorot_uniform'))
    model.add(SingularValueBounding())  # Apply SVB regularization after each layer
    # Add hidden layers
    for i in range(20):
        model.add(layers.Dense(num_features, activation=shifted_hard_sigmoid))
        # model.add(DegenerateBatchNormalization())
        # model.add(BoundedBatchNormalization())
        # model.add(SingularValueBounding())  # Apply SVB regularization after each layer
        model.add(Dropout(0.3))
    # add output layer
    model.add(layers.Dense(1, activation=tanH))

    # optimizers
    # Customize SGD optimizer
    learning_rate = 0.001
    momentum = 0.8
    optimizer1 = SGD(learning_rate=learning_rate,momentum=momentum, nesterov=True)
    # Adagrad
    # optimizer2 = Adagrad(learning_rate=learning_rate)
    # adam
    #optimizer3 = Adam(learning_rate=learning_rate)
    # adamW
    #optimizer4 = AdamW(learning_rate=learning_rate)
    # Compile the model
    model.compile(optimizer=optimizer1, loss=hinge_loss, metrics=['accuracy'])

    # list of predictions on validation set
    # print(f'model predictions {model.predict(X_test)}')
    # print(f'y test  {y_test}')

    # Define the learning rate schedule callback
    lr_schedule = callbacks.LearningRateScheduler(step_decay)

    # Train the model and Pass validation data
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=64, verbose=0,
                        validation_data=(X_test,y_test),callbacks=[lr_schedule])

    '''
    # Compute gradients for first and last layers
    with tf.GradientTape() as tape:
        predictions = model(X_train)
        loss = hinge_loss(y_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients_first_layer.append(gradients[0].numpy())  # First layer
    gradients_last_layer.append(gradients[-2].numpy())  # Last layer
    '''

    # Append the training accuracy for each epoch
    accuracy_results_epochs.append(history.history['accuracy'])
    print(f' accuracy_results_epochs {accuracy_results_epochs}')
    # Append the validation accuracy for each epoch
    validation_accuracy_results_epochs.append(history.history['val_accuracy'])
    print(f' validation_accuracy_results_epochs {validation_accuracy_results_epochs}')
    # Append the loss on training for each epoch
    loss_results_epochs.append(history.history['loss'])
    print(f' loss_results_epochs {loss_results_epochs}')
    # Append the loss on validation for each epoch
    validation_loss_results_epochs.append(history.history['val_loss'])
    print(f' validation_loss_results_epochs {validation_loss_results_epochs}')

    # Evaluate the model on the test set
    accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f" iteration{ _ + 1} num hidden layers 20 accuracy{accuracy}")

    # Append accuracy to the results list
    accuracy_results.append((accuracy[1]) * 100)

'''
# Calculate average gradients for first and last layers
avg_gradients_first_layer = [np.mean(items) for items in zip(*gradients_first_layer)]
all_gradients_first_layer.append(avg_gradients_first_layer)
print(all_gradients_first_layer)
avg_gradients_last_layer = [np.mean(items) for items in zip(*gradients_last_layer)]
all_gradients_last_layer.append(avg_gradients_last_layer)
print(all_gradients_last_layer)
'''

# Calculate the average accuracy and standard deviation for this model configuration
avg_accuracy = np.mean(accuracy_results)
print(avg_accuracy)
std_dev = np.std(accuracy_results)
print(std_dev)

# Calculate the average training accuracy on num_runs for each epoch
avg_accuracy_results_epochs = [np.mean(items) for items in zip(*accuracy_results_epochs)]

# Calculate the average validation accuracy on num_runs for each epoch
avg_validation_accuracy_results_epochs = [np.mean(items) for items in zip(*validation_accuracy_results_epochs)]

# Calculate the average training loss on num_runs for each epoch
avg_loss_results_epochs = [np.mean(items) for items in zip(*loss_results_epochs)]

# Calculate the average validation loss on num_runs for each epoch
avg_validation_loss_results_epochs = [np.mean(items) for items in zip(*validation_loss_results_epochs)]


'''
# Plot distribution plot for gradients of the first layer
plt.figure(figsize=(10, 5))
plt.hist(all_gradients_first_layer, bins=50, alpha=0.7)
plt.xlabel('Gradient Value')
plt.ylabel('Frequency')
plt.title('Distribution of Gradients for First Layer')
plt.grid(True)
plt.show()

# Plot distribution plot for gradients of the last layer
plt.figure(figsize=(10, 5))
plt.hist(all_gradients_last_layer, bins=50, alpha=0.7)
plt.xlabel('Gradient Value')
plt.ylabel('Frequency')
plt.title('Distribution of Gradients for Last Layer')
plt.grid(True)
plt.show()
'''

# Plot the average training accuracy and validation accuracy against epochs for different numbers of hidden layers
plt.figure(figsize=(16, 6))  # Set the figure size
# Plot accuracy graph
# Evaluation accuracy over time
plt.subplot(1, 2, 1)  # Create subplot for accuracy
plt.plot(range(num_epochs), avg_accuracy_results_epochs, label='Training', color='blue')
plt.plot(range(num_epochs), avg_validation_accuracy_results_epochs, '--', label='Validation', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Epochs for Different Numbers of Hidden Layers')
plt.legend()
plt.grid(True)

# Plot loss graph
plt.subplot(1, 2, 2)  # Create subplot for accuracy
plt.plot(range(num_epochs), avg_loss_results_epochs, label='Training', color='red')
plt.plot(range(num_epochs), avg_validation_loss_results_epochs, '--', label='Validation', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epochs for Different Numbers of Hidden Layers')
plt.legend()
plt.grid(True)
# plot together
plt.tight_layout()  # Adjust subplot layout to fit labels
plt.show()

# print results
print(f"Optimal Accuracy: {avg_accuracy}%")



# basic imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#libraries for  classification

from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# Extract features and target
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target variable (0: setosa, 1: versicolor, 2: virginica)

from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


def softmax(z):
    e_z = np.exp(z - np.max(z))  # Subtracting max(z) for numerical stability
    return e_z / e_z.sum(axis=1, keepdims=True)

# Number of features and classes
num_features = X.shape[1]
num_classes = len(np.unique(y))

# Initialize weights and bias
np.random.seed(0)  # For reproducibility
weights = np.random.randn(num_features, num_classes)
bias = np.zeros(num_classes)

# Hyperparameters
learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1]
batch_size = 30
epochs = 50
# Lists to store validation and test accuracies for each learning rate
validation_accuracies = []
test_accuracies = []

for learning_rate in learning_rates:

    for epoch in range(epochs):
        # Shuffle the training data
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, len(X_train), batch_size):
            # Mini-batch sampling
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            # Forward pass
            logits = np.dot(X_batch, weights) + bias
            probabilities = softmax(logits)

            # Compute loss (cross-entropy)
            loss = -np.log(probabilities[range(len(X_batch)), y_batch]).mean()

            # Compute gradients
            d_logits = probabilities
            d_logits[range(len(X_batch)), y_batch] -= 1
            d_logits /= len(X_batch)

            d_weights = np.dot(X_batch.T, d_logits)
            d_bias = np.sum(d_logits, axis=0)

            # Update weights and bias
            weights -= learning_rate * d_weights
            bias -= learning_rate * d_bias


    # Calculate accuracy on the validation set
    logits_val = np.dot(X_val, weights) + bias
    predictions_val = np.argmax(logits_val, axis=1)
    val_accuracy = np.mean(predictions_val == y_val)
    validation_accuracies.append(val_accuracy)

    # Calculate accuracy on the test set
    logits_test = np.dot(X_test, weights) + bias
    predictions_test = np.argmax(logits_test, axis=1)
    test_accuracy = np.mean(predictions_test == y_test)
    test_accuracies.append(test_accuracy)

# Plot validation and test accuracies vs. learning rates
plt.figure(figsize=(10, 5))
plt.plot(learning_rates, validation_accuracies, label='Validation Accuracy')
plt.plot(learning_rates, test_accuracies, label='Test Accuracy')
plt.xscale('log')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Accuracy')
plt.title('Validation and Test Accuracies vs. Learning Rate')
plt.legend()
plt.grid(True)
plt.show()

print("Training complete.")

np.argmax(logits_test,axis=1)

X_train_set_0 = X_train[y_train == 0]
X_train_set_1 = X_train[y_train == 1]
X_train_set_2 = X_train[y_train == 2]
# set_
y_train_set_0 = y_train[y_train == 0]
y_train_set_1 = y_train[y_train == 1]
y_train_set_2 = y_train[y_train == 2]

learning_rate=0.1

X_0_probs = []
X_1_probs = []
X_2_probs = []

for epoch in range(epochs):
    # Shuffle the training data
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    for i in range(0, len(X_train), batch_size):
        # Mini-batch sampling
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        # Forward pass
        logits = np.dot(X_batch, weights) + bias
        probabilities = softmax(logits)

        # Compute loss (cross-entropy)
        loss = -np.log(probabilities[range(len(X_batch)), y_batch]).mean()

        # Compute gradients
        d_logits = probabilities
        d_logits[range(len(X_batch)), y_batch] -= 1
        d_logits /= len(X_batch)

        d_weights = np.dot(X_batch.T, d_logits)
        d_bias = np.sum(d_logits, axis=0)

        # Update weights and bias
        weights -= learning_rate * d_weights
        bias -= learning_rate * d_bias

    # Calculate mean probabilities for each class within each dataset
    mean_prob_0 = np.mean(softmax(np.dot(X_train_set_0, weights) + bias), axis=0)
    mean_prob_1 = np.mean(softmax(np.dot(X_train_set_1, weights) + bias), axis=0)
    mean_prob_2 = np.mean(softmax(np.dot(X_train_set_2, weights) + bias), axis=0)

    # Append the mean probabilities to the respective lists
    X_0_probs.append(mean_prob_0)
    X_1_probs.append(mean_prob_1)
    X_2_probs.append(mean_prob_2)

plt.figure(figsize=(10, 6))
for class_label in range(num_classes):
    plt.plot(np.arange(epochs), [mean[class_label] for mean in X_0_probs], label=f'Class {class_label}')

plt.title('Mean Probabilities for Setosa vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Probability')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for class_label in range(num_classes):
    plt.plot(np.arange(epochs), [mean[class_label] for mean in X_1_probs], label=f'Class {class_label}')

plt.title('Mean Probabilities for Versicolor vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Probability')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for class_label in range(num_classes):
    plt.plot(np.arange(epochs), [mean[class_label] for mean in X_2_probs], label=f'Class {class_label}')

plt.title('Mean Probabilities for Virgininca vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Probability')
plt.legend()
plt.show()

logits_test = np.dot(X_test, weights) + bias
y_pred = np.argmax(logits_test, axis=1)

y_pred

y_test

from sklearn import metrics

labels = [0,1,2]
cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

print(metrics.classification_report(y_test, y_pred))


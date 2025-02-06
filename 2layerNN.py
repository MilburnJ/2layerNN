import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# For reproducibility
np.random.seed(42)

# Generate dataset: 200 points in the range [-10, 10]
X = np.linspace(-10, 10, 200).reshape(-1, 1)
y = 2 * X + 1  # The target function: y = 2x + 1

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Network architecture parameters
input_dim = 1      # one feature (x)
hidden_dim = 10    # 10 neurons in the hidden layer
output_dim = 1     # one output (y)

# Initialize weights and biases randomly
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))

# Define the activation function and its derivative (using tanh)
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Hyperparameters
learning_rate = 0.001
n_epochs = 10000  # Number of training iterations

# For recording the loss at each epoch
losses = []

# Training loop
for epoch in range(n_epochs):
    # -------- Forward Pass --------
    # Hidden layer: linear combination + activation
    z1 = np.dot(X_train, W1) + b1       # shape: (n_samples, hidden_dim)
    a1 = tanh(z1)                      # activated outputs of hidden layer

    # Output layer: linear combination (no activation, since we want a linear output)
    z2 = np.dot(a1, W2) + b2             # shape: (n_samples, output_dim)
    y_pred = z2                        # predicted output

    # -------- Compute Loss (Mean Squared Error) --------
    loss = np.mean((y_pred - y_train) ** 2)
    losses.append(loss)

    # -------- Backpropagation --------
    n_samples = X_train.shape[0]
    # Derivative of MSE loss with respect to y_pred:
    d_y_pred = (2 * (y_pred - y_train)) / n_samples  # shape: (n_samples, output_dim)
    
    # Gradients for the output layer parameters
    dW2 = np.dot(a1.T, d_y_pred)          # shape: (hidden_dim, output_dim)
    db2 = np.sum(d_y_pred, axis=0, keepdims=True)  # shape: (1, output_dim)
    
    # Backpropagate into hidden layer
    d_a1 = np.dot(d_y_pred, W2.T)          # shape: (n_samples, hidden_dim)
    d_z1 = d_a1 * tanh_derivative(z1)      # shape: (n_samples, hidden_dim)
    
    # Gradients for the hidden layer parameters
    dW1 = np.dot(X_train.T, d_z1)          # shape: (input_dim, hidden_dim)
    db1 = np.sum(d_z1, axis=0, keepdims=True)  # shape: (1, hidden_dim)
    
    # -------- Update Weights and Biases --------
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Optionally print the loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Plot the training loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.show()

# -------- Evaluate on Test Data --------
# Forward pass for test data
z1_test = np.dot(X_test, W1) + b1
a1_test = tanh(z1_test)
z2_test = np.dot(a1_test, W2) + b2
y_test_pred = z2_test

# Plot the true vs. predicted outputs on the test set
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, label="True", color="blue")
plt.scatter(X_test, y_test_pred, label="Predicted", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Test Data: True vs Predicted")
plt.legend()
plt.show()

# -------- Visualize the Fit over the Entire Range --------
X_range = np.linspace(-10, 10, 200).reshape(-1, 1)
z1_range = np.dot(X_range, W1) + b1
a1_range = tanh(z1_range)
z2_range = np.dot(a1_range, W2) + b2
y_range_pred = z2_range

plt.figure(figsize=(10, 5))
plt.plot(X_range, 2 * X_range + 1, label="True Function: 2x+1", color="blue")
plt.plot(X_range, y_range_pred, label="Network Prediction", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function Fit")
plt.legend()
plt.show()

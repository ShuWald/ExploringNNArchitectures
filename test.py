import numpy as np
import matplotlib.pyplot as plt

# Target function: y = 2*x1 + 3*x2
# We'll compare different architectures to learn this simple linear relationship

class SimpleNetwork:
    def __init__(self, architecture_type="wide", learning_rate=0.1):
        self.lr = learning_rate
        self.losses = []
        
        if architecture_type == "wide":
            # Wide: 2 inputs -> 4 hidden -> 1 output (single layer, more neurons)
            self.W1 = np.random.normal(0, 0.5, (2, 4))  # 2x4
            self.W2 = np.random.normal(0, 0.5, (4, 1))  # 4x1
            self.architecture = "Wide (2->4->1)"
            
        elif architecture_type == "deep":
            # Deep: 2 inputs -> 2 hidden -> 2 hidden -> 1 output (more layers, fewer neurons per layer)
            self.W1 = np.random.normal(0, 0.5, (2, 2))  # 2x2
            self.W2 = np.random.normal(0, 0.5, (2, 2))  # 2x2  
            self.W3 = np.random.normal(0, 0.5, (2, 1))  # 2x1
            self.architecture = "Deep (2->2->2->1)"
            
        elif architecture_type == "minimal":
            # Minimal: 2 inputs -> 1 output (direct linear)
            self.W1 = np.random.normal(0, 0.5, (2, 1))  # 2x1
            self.architecture = "Minimal (2->1)"
    
    def forward(self, X):
        if hasattr(self, 'W3'):  # Deep network
            self.z1 = X @ self.W1  # No activation for simplicity
            self.a1 = self.z1
            self.z2 = self.a1 @ self.W2
            self.a2 = self.z2
            self.z3 = self.a2 @ self.W3
            return self.z3
        elif self.W2.shape[0] == 4:  # Wide network
            self.z1 = X @ self.W1
            self.a1 = self.z1  # No activation
            self.z2 = self.a1 @ self.W2
            return self.z2
        else:  # Minimal network
            return X @ self.W1
    
    def backward(self, X, y, y_pred):
        m = X.shape[0]
        dL_dy = 2 * (y_pred - y) / m  # MSE derivative
        
        if hasattr(self, 'W3'):  # Deep network
            # Backprop through 3 layers
            dL_dW3 = self.a2.T @ dL_dy
            dL_da2 = dL_dy @ self.W3.T
            
            dL_dW2 = self.a1.T @ dL_da2
            dL_da1 = dL_da2 @ self.W2.T
            
            dL_dW1 = X.T @ dL_da1
            
            # Update weights
            self.W3 -= self.lr * dL_dW3
            self.W2 -= self.lr * dL_dW2
            self.W1 -= self.lr * dL_dW1
            
        elif self.W2.shape[0] == 4:  # Wide network
            # Backprop through 2 layers
            dL_dW2 = self.a1.T @ dL_dy
            dL_da1 = dL_dy @ self.W2.T
            dL_dW1 = X.T @ dL_da1
            
            # Update weights
            self.W2 -= self.lr * dL_dW2
            self.W1 -= self.lr * dL_dW1
            
        else:  # Minimal network
            # Direct update
            dL_dW1 = X.T @ dL_dy
            self.W1 -= self.lr * dL_dW1
    
    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = np.mean((y_pred - y) ** 2)
        self.losses.append(loss)
        self.backward(X, y, y_pred)
        return loss

# Generate training data
np.random.seed(42)
X = np.random.normal(0, 1, (100, 2))
y = 2 * X[:, 0:1] + 3 * X[:, 1:2]  # True relationship: y = 2*x1 + 3*x2

# Initialize networks
networks = {
    'Minimal': SimpleNetwork('minimal', 0.01),
    'Wide': SimpleNetwork('wide', 0.01), 
    'Deep': SimpleNetwork('deep', 0.01)
}

# Print initial weights to see starting point
print("=== INITIAL WEIGHTS ===")
for name, net in networks.items():
    print(f"\n{name} Network ({net.architecture}):")
    print(f"W1 shape: {net.W1.shape}")
    print(f"W1:\n{net.W1}")
    if hasattr(net, 'W2'):
        print(f"W2 shape: {net.W2.shape}")
        print(f"W2:\n{net.W2}")
    if hasattr(net, 'W3'):
        print(f"W3 shape: {net.W3.shape}")
        print(f"W3:\n{net.W3}")

# Training
epochs = 1000
print(f"\n=== TRAINING FOR {epochs} EPOCHS ===")

for epoch in range(epochs):
    for name, net in networks.items():
        loss = net.train_step(X, y)
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} - {name:7s}: Loss = {loss:.6f}")

# Final weights and analysis
print("\n=== FINAL WEIGHTS & ANALYSIS ===")
for name, net in networks.items():
    print(f"\n{name} Network:")
    
    # Show final weights
    if hasattr(net, 'W3'):  # Deep
        print(f"W1:\n{net.W1}")
        print(f"W2:\n{net.W2}")
        print(f"W3:\n{net.W3}")
        
        # Calculate effective linear transformation
        effective_weights = net.W1 @ net.W2 @ net.W3
        print(f"Effective linear transformation: {effective_weights.flatten()}")
        
    elif net.W2.shape[0] == 4:  # Wide
        print(f"W1:\n{net.W1}")
        print(f"W2:\n{net.W2}")
        
        # Calculate effective linear transformation
        effective_weights = net.W1 @ net.W2
        print(f"Effective linear transformation: {effective_weights.flatten()}")
        
    else:  # Minimal
        print(f"W1:\n{net.W1}")
        print(f"Direct transformation: {net.W1.flatten()}")
    
    # Test prediction
    test_input = np.array([[1, 1]])
    pred = net.forward(test_input)
    expected = 2 * 1 + 3 * 1  # Should be 5
    print(f"Test: f([1,1]) = {pred[0,0]:.4f} (expected: {expected})")
    print(f"Final loss: {net.losses[-1]:.6f}")

# Plot convergence
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
for name, net in networks.items():
    plt.plot(net.losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Convergence')
plt.legend()
plt.yscale('log')

plt.subplot(1, 2, 2)
for name, net in networks.items():
    plt.plot(net.losses[-200:], label=name)  # Last 200 epochs
plt.xlabel('Epoch (Last 200)')
plt.ylabel('MSE Loss')
plt.title('Final Convergence Detail')
plt.legend()

plt.tight_layout()
plt.show()

print("\n=== KEY INSIGHTS ===")
print("1. TARGET: y = 2*x1 + 3*x2")
print("2. All networks learn the same linear relationship but through different paths")
print("3. Wide networks: Multiple redundant pathways, often faster initial convergence")
print("4. Deep networks: Must coordinate across layers, potentially slower but more expressive")
print("5. Minimal networks: Direct path, guaranteed to work for linear problems")
print("\nNote: Without activation functions, all networks are computing linear transformations!")
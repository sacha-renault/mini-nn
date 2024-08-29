import torch
import torch.nn as nn

class CustomDense(nn.Module):
    def __init__(self, in_features, out_features, activation_func=None):
        super(CustomDense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation_func = activation_func
        # Initialize weights and biases to 0.1
        nn.init.constant_(self.linear.weight, 0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, x):
        x = self.linear(x)
        if self.activation_func:
            x = self.activation_func(x)
        return x

# Define the network
class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.l1 = CustomDense(5, 5, activation_func=torch.tanh)
        self.l2 = CustomDense(5, 5, activation_func=torch.tanh)
        self.l3 = CustomDense(5, 5, activation_func=torch.tanh)
        self.l4 = CustomDense(5, 1, activation_func=torch.tanh)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x

# Instantiate the network
model = CustomNetwork()

# Input tensor with requires_grad=True to track gradients
inputs = torch.ones(5, requires_grad=True)

# Forward pass
output = model(inputs)

# Manually create a dummy loss for demonstration purposes
# Here, we'll use the mean of the output tensor as a dummy "loss"
dummy_loss = output.mean()

# Perform backward pass on the dummy loss
dummy_loss.backward()

# If you need to inspect the gradients of the input tensor
if inputs.grad is not None:
    print(f'inputs.grad: {inputs.grad}')

# Print gradients of all parameters
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'{name}: Grad={param.grad}')

print(dummy_loss)


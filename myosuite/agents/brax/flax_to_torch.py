import torch
import numpy as np

from myosuite.agents.brax.distribution import NormalTanhDistribution


class TorchModel(torch.nn.Module):
    def __init__(self, params):
        super(TorchModel, self).__init__()

        self.dist = NormalTanhDistribution(
            event_size=5
        )
        
        # Unpack the params tuple
        normalization_params, network_params = params[0], params[1]
        
        # Store mean and std for normalization
        self.mean = torch.tensor(np.array(normalization_params.mean), dtype=torch.float32)
        self.std = torch.tensor(np.array(normalization_params.std), dtype=torch.float32)
        
        # Extract the parameters dictionary
        # params_dict = network_params.policy['params']
        params_dict = network_params['params']
        
        # Create a ModuleList to hold the layers
        self.layers = torch.nn.ModuleList()
        
        # Iterate over the params_dict to create layers
        for key, layer_params in params_dict.items():
            # Extract kernel and bias
            kernel = layer_params['kernel'].T
            bias = layer_params['bias']
            
            # Ensure the kernel and bias are writable
            kernel = np.copy(kernel)
            bias = np.copy(bias)
            
            # Create a linear layer
            layer = torch.nn.Linear(len(kernel[0]), len(bias))
            
            # Convert to PyTorch tensors and set as parameters
            layer.weight = torch.nn.Parameter(torch.tensor(kernel, dtype=torch.float32))
            layer.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
            
            # Add the layer to the ModuleList
            self.layers.append(layer)

    def forward(self, x):
        # Normalize the input
        x = (x - self.mean) / self.std
        
        # Pass through each layer
        for layer in self.layers[:-1]:
            x = torch.nn.functional.silu(layer(x))
        
        # Last layer without activation
        x = self.layers[-1](x)
        x = self.dist.mode(x)
        return x
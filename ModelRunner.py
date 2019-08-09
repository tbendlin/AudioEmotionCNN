import torch
import numpy as np
from torchvision import transforms
from ConvNeuralNetwork import *

MODEL_PATH = "./saved_model.pt"

class ModelRunner:
    def __init__(self, device_type="cpu", map_location="cuda:0"):
        
        # Setting the device. Default is CPU because Pi has no cuda support
        device = torch.device(device_type)
        
        # Using the model defined in ConvNeuralNetwork
        model = ConvNeuralNetwork()
        
        # Since we are just classifying we don't need the whole model
        # just the params for inference
        model.load_state_dict(torch.load(MODEL_PATH, map_location={map_location:device_type}))

        # Must set to evaluation mode to ensure consistent results
        model.eval()
        
        self.trained_model = model
        self.data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((129, 129)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    def __evaluate_class__(self, input_spectrogram):
        
        # Transform into the form expected by the model
        tensor = self.data_transforms(input_spectrogram)
        
        # Run through the model
        output = self.trained_model(tensor)
        return torch.argmax(output)
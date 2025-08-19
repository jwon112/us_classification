
import wandb
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model import EnhancedResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_gradcam_examples(model, dataloader, grad_cam):
    model.eval()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        image_tensor = inputs[0]
        class_idx = preds[0].item()

        # Use GradCAM to generate the heatmap
        heatmap = grad_cam.generate_cam(image_tensor, class_idx)
        image = inputs[0].cpu().numpy().transpose(1, 2, 0)

        # Log the Grad-CAM to WandB
        log_gradcam_to_wandb(image, heatmap)
        break


def overlay_heatmap_on_image(image, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return overlay


def log_gradcam_to_wandb(image, heatmap):
    overlay_image = overlay_heatmap_on_image(image, heatmap)
    wandb.log({"gradcam_overlay": [wandb.Image(overlay_image, caption="Grad-CAM Overlay")]})

def generate_gradcam_heatmap(model, input_image, target_layer):
    # Forward pass
    output = model(input_image.unsqueeze(0))
    target_class = output.argmax().item()

    # Compute the gradients
    model.zero_grad()
    output[0, target_class].backward()

    # Get gradients and activations
    gradients = target_layer.gradients
    activations = target_layer.activations

    # Global Average Pooling
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(len(pooled_gradients)):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Generate heatmap
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1  # Avoid division by zero

    return heatmap
# Updated main script

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Use register_full_backward_hook instead of register_backward_hook
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, class_idx=None):
        self.model.eval()
        input_image = input_image.unsqueeze(0)  # Add batch dimension

        # Forward pass
        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)

        # Compute Grad-CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.size(2), input_image.size(3)))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

def visualize_cam(image, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(image)
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)


def show_cam_on_image(img, heatmap):
    # Ensure heatmap is non-negative
    heatmap = np.maximum(heatmap, 0)

    # Normalize the heatmap to the range [0, 1], avoiding division by zero
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap

    # Convert heatmap to uint8 format (values range [0, 255])
    heatmap = np.uint8(255 * heatmap)

    # Apply color map (convert grayscale heatmap to colored heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Normalize heatmap to the range [0, 1] as float32
    heatmap = np.float32(heatmap) / 255

    # Superimpose heatmap on the original image (assuming 'img' is in range [0, 1])
    cam = heatmap + np.float32(img)

    # Normalize the result to the range [0, 1] again, avoiding division by zero
    cam = cam / np.max(cam) if np.max(cam) > 0 else cam

    # Convert the final result to uint8 format (values range [0, 255])
    return np.uint8(255 * cam)

if __name__ == "__main__":
    # Initialize your model, dataloaders, etc.
    # Assuming you have defined your model as EnhancedResNet and it is loaded with weights

    # Define target layer based on your model structure
    model = EnhancedResNet(num_classes=num_classese).to(device)  # Example of loading model
    dataloaders = {"test": test_dataloader}  # Example of loading dataloader

    # Define the target layer for Grad-CAM (Adjust based on your model's structure)
    target_layer = model.residual_block  # Replace with the appropriate layer
    grad_cam = GradCAM(model, target_layer)

    # Now use the Grad-CAM integration to log examples
    log_gradcam_examples(model, dataloaders['test'], grad_cam)
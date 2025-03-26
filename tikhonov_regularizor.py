import torch
import torch.nn.functional as F

def compute_gradients(image_tensor):
    device = image_tensor.device
    # Define the Sobel operators for gradient calculation
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 3, 3)
    
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 3, 3)
    
    # Convert image_tensor to (N, C, H, W) and ensure it's a float tensor
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).float()  # Shape: (1, 3, H, W)
    
    # Initialize tensors to accumulate gradients
    grad_x_total = torch.zeros_like(image_tensor).to(device)
    grad_y_total = torch.zeros_like(image_tensor).to(device)

    # Apply the Sobel filters to compute gradients for each channel
    for i in range(image_tensor.size(1)):  # Loop over each channel
        channel_tensor = image_tensor[:, i:i+1, :, :]  # Extract the single channel
        grad_x = F.conv2d(channel_tensor, sobel_x, padding=1)
        grad_y = F.conv2d(channel_tensor, sobel_y, padding=1)
        grad_x_total[:, i, :, :] = grad_x
        grad_y_total[:, i, :, :] = grad_y
    
    return grad_x_total.squeeze(0), grad_y_total.squeeze(0)

def tikhonov_regularization(image_tensor):
    grad_x, grad_y = compute_gradients(image_tensor)
    
    # Compute the squared gradients
    grad_x_squared = grad_x ** 2
    grad_y_squared = grad_y ** 2
    
    # Compute the Tikhonov regularization term (sum of squared gradients)
    regularization_term = grad_x_squared.sum() + grad_y_squared.sum()
    
    return regularization_term

if __name__ == '__main__':
    # Example usage
    W, H = 256, 256
    image_tensor = torch.randn(W, H, 3)  # Random image tensor for demonstration

    # Compute Tikhonov regularization term
    reg_term = tikhonov_regularization(image_tensor)
    print(f'Tikhonov Regularization Term: {reg_term.item()}')


import torch
import torch.nn.functional as F


class GradCAM_Dynamic:
    """A dimension-agnostic GradCAM extractor that works on 2D or 3D layers."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)

        score = output[0, 0]
        score.backward()

        # Dynamically find which dimensions to pool (everything except dim 1, which is Channels)
        # If 4D [B, C, H, W], pool_dims = [0, 2, 3]
        # If 5D [B, C, D, H, W], pool_dims = [0, 2, 3, 4]
        pool_dims = [0] + list(range(2, self.gradients.dim()))

        pooled_grads = torch.mean(self.gradients, dim=pool_dims, keepdim=True)

        cam = self.activations * pooled_grads
        cam = torch.sum(cam, dim=1).squeeze()

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach()
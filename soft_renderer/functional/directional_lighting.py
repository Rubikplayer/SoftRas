import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def directional_lighting(light, normals, light_intensity=0.5, light_color=(1,1,1), 
                         light_direction=(0,1,0)):
    """
    Args:
        light: [nb, :, 3]
        normals: [nb, :, 3]
        light_intensity: scalar, as a factor applied on light_color, or [nb,] for differentiable
        light_color: [nb, 3], if variable then differentiable
        light_direction: [nb, 3], if variable then differentiable
    Returns:
        light: [nb, :, 3], lighting effect already applied
    """
    device = light.device

    if isinstance(light_color, tuple) or isinstance(light_color, list):
        light_color = torch.tensor(light_color, dtype=torch.float32, device=device)
    elif isinstance(light_color, np.ndarray):
        light_color = torch.from_numpy(light_color).float().to(device)
    elif torch.is_tensor(light_color):
        light_color = light_color.to(device) # differentiable
    else:
        raise RuntimeError(f"invalid light_color type = {type(light_color)}")

    if isinstance(light_direction, tuple) or isinstance(light_direction, list):
        light_direction = torch.tensor(light_direction, dtype=torch.float32, device=device)
    elif isinstance(light_direction, np.ndarray):
        light_direction = torch.from_numpy(light_direction).float().to(device)
    elif torch.is_tensor(light_direction):
        light_direction = light_direction.to(device) # differentiable
    else:
        raise RuntimeError(f"invalid light_direction type = {type(light_direction)}")

    # fix shape
    if light_color.ndimension() == 1:
        light_color = light_color[None, :]
    if light_direction.ndimension() == 1:
        light_direction = light_direction[None, :] #[nb, 3]

    # normalize direction to proper directional vector
    light_direction = F.normalize(light_direction, eps=1e-6)
    # import ipdb; ipdb.set_trace()

    # apply diffusive shading
    cosine = F.relu(torch.sum(normals * light_direction, dim=2)) #[]
    light += light_intensity * (light_color[:, None, :] * cosine[:, :, None])
    return light # [nb, :, 3]
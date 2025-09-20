
"""
U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection
Implementation for Team 1: The Isolationists - Subject & Background Separation Specialists

This module implements the complete U²-Net architecture with RSU blocks for 
pixel-perfect background removal and salient object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
import os
from typing import Tuple, List, Optional

class REBNCONV(nn.Module):
    """Basic convolution block with batch normalization and ReLU activation"""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, dirate: int = 1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))

class RSU(nn.Module):
    """
    Residual U-block (RSU) - Core building block of U²-Net

    Args:
        height: Number of levels in the RSU block
        in_ch: Input channels
        mid_ch: Middle channels
        out_ch: Output channels  
        dilated: Whether to use dilated convolutions
    """

    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int, dilated: bool = False):
        super(RSU, self).__init__()
        self.height = height
        self.dilated = dilated

        # Input convolution
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        # First encoder layer
        self.encoder_layers.append(REBNCONV(out_ch, mid_ch, dirate=1))
        self.pool_layers.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))

        # Subsequent encoder layers
        for i in range(2, height):
            dilation = 2**(i-1) if dilated else 1
            self.encoder_layers.append(REBNCONV(mid_ch, mid_ch, dirate=dilation))
            if not dilated and i < height - 1:
                self.pool_layers.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))

        # Bridge layer (bottom of U)
        bridge_dilation = 2**(height-2) if dilated else 1
        self.bridge = REBNCONV(mid_ch, mid_ch, dirate=bridge_dilation)

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(height-2, 0, -1):
            dilation = 2**(i-1) if dilated else 1
            self.decoder_layers.append(REBNCONV(mid_ch*2, mid_ch, dirate=dilation))

        # Final decoder layer
        self.decoder_final = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection
        hx = x
        hxin = self.rebnconvin(hx)

        # Encoder path - store features for skip connections
        encoder_features = [hxin]
        hx = hxin

        for i, (encoder, pool) in enumerate(zip(self.encoder_layers[:-1], self.pool_layers)):
            hx = encoder(hx)
            encoder_features.append(hx)
            hx = pool(hx)

        # Final encoder layer
        if len(self.encoder_layers) > 0:
            hx = self.encoder_layers[-1](hx)
            if not self.dilated:
                encoder_features.append(hx)

        # Bridge
        hx = self.bridge(hx)

        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoder_layers):
            if not self.dilated:
                hx = F.interpolate(hx, size=(encoder_features[-(i+2)].size()[2], 
                                           encoder_features[-(i+2)].size()[3]), 
                                 mode='bilinear', align_corners=False)
            hx = decoder(torch.cat((hx, encoder_features[-(i+2)]), 1))

        # Final decoder
        if not self.dilated:
            hx = F.interpolate(hx, size=(encoder_features[0].size()[2], 
                                       encoder_features[0].size()[3]), 
                             mode='bilinear', align_corners=False)
        hx = self.decoder_final(torch.cat((hx, encoder_features[0]), 1))

        # Residual connection
        return hx + hxin

class U2NET(nn.Module):
    """
    U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection

    Architecture specifications:
    - Model size: 176.3 MB
    - Parameters: 44.0M
    - Target performance: 30 FPS on GTX 1080Ti
    - Input resolution: 320×320×3
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super(U2NET, self).__init__()

        # Encoder stages with increasing depth
        self.stage1 = RSU(7, in_ch, 32, 64)      # 320×320
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU(6, 64, 32, 128)        # 160×160  
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU(5, 128, 64, 256)       # 80×80
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU(4, 256, 128, 512)      # 40×40
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Bridge stages with dilated convolutions
        self.stage5 = RSU(4, 512, 256, 512, dilated=True)  # 20×20
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU(4, 512, 256, 512, dilated=True)  # 10×10

        # Decoder stages with skip connections
        self.stage5d = RSU(4, 1024, 256, 512)
        self.stage4d = RSU(4, 1024, 128, 256) 
        self.stage3d = RSU(5, 512, 64, 128)
        self.stage2d = RSU(6, 256, 32, 64)
        self.stage1d = RSU(7, 128, 16, 64)

        # Side outputs for deep supervision
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        # Final fusion layer
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        hx = x

        # Encoder path
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        # Decoder path with skip connections
        hx5d = self.stage5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=(hx4.size()[2], hx4.size()[3]), 
                               mode='bilinear', align_corners=False)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size()[2], hx3.size()[3]), 
                               mode='bilinear', align_corners=False)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size()[2], hx2.size()[3]), 
                               mode='bilinear', align_corners=False)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size()[2], hx1.size()[3]), 
                               mode='bilinear', align_corners=False)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # Side outputs for multi-stage supervision
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)

        # Resize all side outputs to input size
        d1 = F.interpolate(d1, size=x.size()[2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(d2, size=x.size()[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(d3, size=x.size()[2:], mode='bilinear', align_corners=False)
        d4 = F.interpolate(d4, size=x.size()[2:], mode='bilinear', align_corners=False)
        d5 = F.interpolate(d5, size=x.size()[2:], mode='bilinear', align_corners=False)
        d6 = F.interpolate(d6, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Final fusion
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return (torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), 
                torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6))

class SaliencyDataset(Dataset):
    """Dataset class for salient object detection training"""

    def __init__(self, image_dir: str, mask_dir: str, transform=None, image_size: int = 320):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Handle different mask extensions
        mask_name = img_name.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            mask_name = img_name.rsplit('.', 1)[0] + '.jpg'
            mask_path = os.path.join(self.mask_dir, mask_name)

        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
            mask_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            mask = mask_transform(mask)
        else:
            # Default transforms
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            mask_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            image = image_transform(image)
            mask = mask_transform(mask)

        return image, mask

class BackgroundRemover:
    """Production-ready background removal inference class"""

    def __init__(self, model_path: str, device: str = 'cuda', image_size: int = 320):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size

        # Load model
        self.model = U2NET(3, 1)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def remove_background(self, image_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Remove background from single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            mask_pred = outputs[0]  # Main output

        # Post-process mask
        mask = mask_pred.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_CUBIC)

        # Apply mask to original image
        image_np = np.array(image)

        # Create RGBA output
        result = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
        result[:, :, :3] = image_np
        result[:, :, 3] = mask

        if output_path:
            result_image = Image.fromarray(result, 'RGBA')
            result_image.save(output_path)

        return result, mask

    def batch_remove_background(self, image_paths: List[str], output_dir: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Remove background from batch of images"""
        results = []

        for i, image_path in enumerate(image_paths):
            output_path = os.path.join(output_dir, f"result_{i}.png")
            result, mask = self.remove_background(image_path, output_path)
            results.append((result, mask))

        return results

def get_model_info() -> dict:
    """Get model specifications and information"""
    model = U2NET(3, 1)

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size in MB
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2

    return {
        'model_name': 'U²-Net (U-Squared Net)',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': round(size_mb, 1),
        'target_fps': '30 FPS on GTX 1080Ti',
        'input_resolution': '320×320×3',
        'architecture': 'Two-level nested U-structure with RSU blocks',
        'application': 'Salient object detection and background removal'
    }

if __name__ == "__main__":
    # Display model information
    info = get_model_info()
    print("U²-Net Model Information:")
    print("=" * 40)
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Example usage
    print("\n" + "="*40)
    print("Example Usage:")
    print("1. Initialize model: model = U2NET(3, 1)")
    print("2. Create remover: remover = BackgroundRemover('u2net.pth')")
    print("3. Process image: result, mask = remover.remove_background('input.jpg')")

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import argparse

# Try importing MFRGN model definitions (for TimmModel and TimmModel_u classes)
try:
    from Retrieval_Models.MFRGN import mfrgn_model  # Module with MFRGN model classes
except ImportError:
    mfrgn_model = None

def feature_fusion_all(output):
    """
    Fuse multi-scale features into a single global descriptor.
    - If output is a list/tuple of features (e.g. global and local), L2-normalize each and concatenate.
    - If output is a single feature tensor, L2-normalize it directly.
    Returns a normalized global descriptor tensor.
    """
    if isinstance(output, (list, tuple)):
        # L2 normalize each component feature vector
        normed_parts = [torch.nn.functional.normalize(feat, p=2, dim=1) for feat in output]
        # Concatenate along feature dimension to form the global descriptor
        fused = torch.cat(normed_parts, dim=1)
        # Normalize the fused descriptor (final global descriptor)
        fused = torch.nn.functional.normalize(fused, p=2, dim=1)
        return fused
    else:
        # Single feature vector: L2-normalize and return
        return torch.nn.functional.normalize(output, p=2, dim=1)

def extract_features(input_data, method_dict, input_id=1):
    """
    Extract feature descriptors from the given input image(s) using the specified retrieval model.
    Parameters:
      - input_data: image path (str), PIL Image, NumPy array (BGR image), or a list of images.
      - method_dict: dictionary with keys 'retrieval_method', 'model', and optionally 'transform'.
      - input_id: branch identifier for MFRGN models (1 for UAV/ground, 2 for reference/satellite).
    Returns:
      - Tensor of shape [N, D] with L2-normalized feature descriptors (N images, D descriptor dim).
    """
    method = method_dict.get('retrieval_method', '').strip().upper()
    model = method_dict.get('model') or method_dict.get('retrieval_model')
    if model is None:
        raise ValueError("No model provided in method_dict for feature extraction")
    # Use provided image transform or define a default (256x256 resize, ImageNet normalization)
    transform = method_dict.get('transform') or method_dict.get('img_transform')
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    # If input_data is a list or tuple, process each image and stack the results
    if isinstance(input_data, (list, tuple)):
        features_list = []
        for item in input_data:
            # Recursively extract features for each image in the list
            feat = extract_features(item, method_dict, input_id=input_id)
            features_list.append(feat)
        # Concatenate all feature tensors (each is [1, D]) into a single [N, D] tensor
        return torch.cat(features_list, dim=0)
    # Handle single image input (path, PIL image, NumPy array, or torch Tensor)
    if isinstance(input_data, str):
        # Load image from path and convert to RGB
        img = Image.open(input_data).convert('RGB')
    elif isinstance(input_data, Image.Image):
        img = input_data  # PIL Image already in RGB mode
    elif isinstance(input_data, np.ndarray):
        # Convert OpenCV BGR NumPy array to RGB PIL Image
        img_rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
    elif torch.is_tensor(input_data):
        # If input_data is a torch Tensor, ensure it has shape [1,C,H,W] or [C,H,W]
        if input_data.dim() == 3:
            # Unsqueeze to add batch dimension
            if input_data.shape[0] == 3 or input_data.shape[0] == 1:
                img_tensor = input_data.unsqueeze(0)  # shape [1,C,H,W]
            elif input_data.shape[2] == 3 or input_data.shape[2] == 1:
                # Tensor is [H,W,C], permute to [C,H,W] then unsqueeze
                img_tensor = input_data.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError("Unsupported tensor shape for image input")
        elif input_data.dim() == 4:
            img_tensor = input_data  # already a batch of images
        else:
            raise ValueError("Unsupported tensor dimensions for image input")
        # Apply normalization (if transform includes Normalize) to the tensor
        if hasattr(transform, 'transforms'):
            # If transform is a Compose, apply any Normalize in it (skip Resize/ToTensor)
            for t in transform.transforms:
                if isinstance(t, transforms.Normalize):
                    img_tensor = t(img_tensor)
        else:
            # If a custom transform is provided, assume it handles tensor normalization
            img_tensor = transform(img_tensor)
    else:
        raise TypeError("Unsupported input type for extract_features")
    # If we have a PIL Image from the above cases, apply the transform to get a tensor
    if 'img' in locals():
        img_tensor = transform(img).unsqueeze(0)  # shape [1,C,H,W]
    # Move the image tensor to the same device as the model
    device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device('cpu')
    img_tensor = img_tensor.to(device)
    # Ensure model is in eval mode and perform forward pass to get features
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        if method in ['MFRGN', 'MFRGN_U1652']:
            # For MFRGN models, pass input_id to distinguish branch if supported
            try:
                raw_output = model(img_tensor, input_id=input_id)
            except TypeError:
                raw_output = model(img_tensor)  # Fallback if model.forward doesn't accept input_id
        else:
            raw_output = model(img_tensor)
    # Restore model training mode if it was originally in training mode
    if model_was_training:
        model.train()
    # Fuse multi-scale outputs and normalize to get final descriptor
    features = feature_fusion_all(raw_output)
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features using MFRGN model")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to extract features from")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to the pretrained weight file (e.g., weights_U1652-D2S.pth)")
    parser.add_argument("--output_file", type=str, default="features.pt", help="Output file to save extracted features")
    args = parser.parse_args()
    # Determine retrieval method from weight name or path
    weight_name = os.path.basename(args.weight_path)
    if 'U1652' in weight_name or 'u1652' in weight_name:
        retrieval_method = 'MFRGN_U1652'
    else:
        retrieval_method = 'MFRGN'
    # Initialize the model according to the retrieval method
    if mfrgn_model is None:
        raise ImportError("Could not import MFRGN model definitions")
    if retrieval_method == 'MFRGN':
        model = mfrgn_model.TimmModel(model_name='convnext_base', pretrained=False)
    elif retrieval_method == 'MFRGN_U1652':
        model = mfrgn_model.TimmModel_u(model_name='convnext_base', pretrained=False, num_classes=None)
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}")
    # Load the pretrained weights
    state_dict = torch.load(args.weight_path, map_location='cpu')
    if retrieval_method == 'MFRGN_U1652':
        # Remove incompatible conv1d weights from U1652 model before loading
        keys_to_remove = []
        for key, tensor in state_dict.items():
            if key.startswith('proj_gl') and tensor.ndim == 3:  # old conv1d weights (3D tensor)
                keys_to_remove.append(key)
                bias_key = key.replace('weight', 'bias')
                if bias_key in state_dict:
                    keys_to_remove.append(bias_key)
        for k in keys_to_remove:
            state_dict.pop(k, None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    # Define image transform (resize to 256 and ConvNeXt/ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Process all images in the directory and extract features
    image_names = sorted([n for n in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, n))])
    features_list = []
    with torch.no_grad():
        for name in image_names:
            img_path = os.path.join(args.image_dir, name)
            # Load and transform the image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
            # Forward pass to get feature (handles both single or multi-scale outputs)
            feat = model(img_tensor)
            feat = feature_fusion_all(feat)  # fuse multi-scale features if needed
            features_list.append(feat.squeeze(0))
    # Stack all features into a tensor [N, D] and save with image names
    features = torch.stack(features_list, dim=0)
    torch.save({'names': image_names, 'features': features}, args.output_file)
    print(f"Saved features for {features.size(0)} images to {args.output_file}")

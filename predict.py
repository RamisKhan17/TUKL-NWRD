import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

def load_checkpoint(checkpoint, model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda", threshold=0.5):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > threshold).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return dice_score / len(loader)

def compute_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold=0.5, eps=1e-6):
    pred_bin = (pred_mask > threshold).float()
    if true_mask.ndim == 2:
        true_mask = true_mask.unsqueeze(0).unsqueeze(0)
    elif true_mask.ndim == 3:
        true_mask = true_mask.unsqueeze(0)
    intersection = (pred_bin * true_mask).sum()
    union = (pred_bin + true_mask - pred_bin * true_mask).sum()
    return ((intersection + eps) / (union + eps)).item()

def pad_to_multiple_of_32(img):
    img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    w, h = img.size
    pad_w = (32 - w % 32) % 32
    pad_h = (32 - h % 32) % 32
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    img = ImageOps.expand(img, padding, fill=0)
    return np.array(img)

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

checkpoint = torch.load("models/DeepLabV3+(Res50-bin-NWRD).pth.tar", map_location="cpu", weights_only=False)
load_checkpoint(checkpoint, model=model)

image_path = "patchedData/test/images/25_1_6.JPG"
mask_path = "patchedData/test/masks/25_1_6.png"

transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

image = pad_to_multiple_of_32(np.array(Image.open(image_path).convert("RGB")))
mask = pad_to_multiple_of_32(np.array(Image.open(mask_path).convert("L"))).astype(np.float32)
mask[mask > 1.0] = 1.0

augmented = transform(image=image, mask=mask)
image_tensor = augmented["image"]

model.eval()
with torch.no_grad():
    pred = torch.sigmoid(model(image_tensor.unsqueeze(0)))
model.train()

predicted_mask = (pred > 0.5).float()
true_mask_tensor = torch.tensor(mask, dtype=torch.float32)

iou_score = compute_iou(pred, true_mask_tensor)
print(f"IoU: {iou_score:.4f}")

img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
mask_np = mask
pred_np = predicted_mask.squeeze().cpu().numpy()

model.to("cpu")
target_layer = model.encoder.layer4[-1]
input_tensor = image_tensor.unsqueeze(0)

H, W = image_tensor.shape[1:]
cam_mask = np.ones((H, W), dtype=np.float32)
target_category = [SemanticSegmentationTarget(category=0, mask=cam_mask)]

cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)[0]

def unnormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return img * std + mean

rgb_img = unnormalize(img_np)
rgb_img = np.clip(rgb_img, 0, 1)

visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.figure(figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.imshow(rgb_img)
plt.title(f"Input Image\n{W}×{H}")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mask_np, cmap='gray')
plt.title("Ground Truth Mask")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(pred_np, cmap='gray')
plt.title("Predicted Mask")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(visualization)
plt.title("Grad-CAM++")
plt.axis('off')

plt.tight_layout()
plt.show()
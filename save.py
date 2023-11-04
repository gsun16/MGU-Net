import torch
import cv2
import numpy as np
from models.net_builder import net_builder
from data.seg_transforms import Normalize  # Make sure to use the appropriate preprocessing transforms

# Load the model (you can use the code from the previous response)
model_checkpoint_path = r'C:\Users\12179\Desktop\OCT\ml\MGU-Net\result\data\dataset\train\tsmgunet_0.001_t1\model\model_latest.pth.tar'
net = net_builder('tsmgunet')  # Assuming 'tsmgunet' is the model name
model = torch.nn.DataParallel(net).cuda()
checkpoint = torch.load(model_checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Load and preprocess an image (replace with your image path)
image_path = r'C:\Users\12179\Desktop\OCT\Test\oct5.jpg'
image = cv2.imread(image_path)  # Load your image using OpenCV or another library
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure it's in RGB format

# Convert the image to a PyTorch tensor and add a batch dimension
image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().cuda()
image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

# Apply the same preprocessing used during training (e.g., normalization)
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use the correct mean and std
image_tensor = normalize(image_tensor)

# Perform inference using the model
with torch.no_grad():
    output = model(image_tensor)

# Depending on your task, you may need to post-process the output (e.g., thresholding for binary masks)
# For segmentation tasks, you can convert the output to a binary mask using a threshold
threshold = 0.5  # Adjust the threshold as needed
binary_mask = (output > threshold).cpu().numpy()

# Now 'binary_mask' contains the segmentation mask for the input image

# You can visualize or further process the segmentation mask as needed

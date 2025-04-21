# python3 -m venv .venv
# . .venv/bin/activate
# pip install Flask Pillow torch torchvision flask_cors "numpy<2" numpy
# flask --app gmm2api.py run

import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.models.segmentation as segmentation # Import segmentation models
from flask_cors import CORS
import numpy as np # Needed for creating color mask

app = Flask(__name__)
# Allow CORS from your frontend development server (adjust port if needed)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Classification Model Setup ---
CLF_MODEL_PATH = "model_weights.pth" # Path for classification model weights
model_clf = models.resnet50(weights=None, num_classes=10) # Rename to avoid conflict
try:
    model_clf.load_state_dict(torch.load(CLF_MODEL_PATH, map_location=device))
    print(f"Classification model weights loaded from {CLF_MODEL_PATH}")
except FileNotFoundError:
    print(f"Warning: {CLF_MODEL_PATH} not found. Classification endpoint will use an untrained model.")
except Exception as e:
    print(f"Error loading classification weights from {CLF_MODEL_PATH}: {e}")

model_clf.to(device)
model_clf.eval()

# Define classification preprocessing steps - REVERTED TO ORIGINAL
print(f"Defining classification transforms (Original: Resize(36)/CenterCrop(32))")
transform_clf = transforms.Compose([
    transforms.Resize(36),      # Original resize
    transforms.CenterCrop(32),  # Original crop
    transforms.ToTensor(),
    # Add normalization if your model was trained with it.
    # Example for ImageNet:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names (for CIFAR-10 - adjust if your model uses different classes)
class_names_clf = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# --- Segmentation Model Setup ---

# Define the path to YOUR saved segmentation model weights
SEG_MODEL_PATH = "deeplabv3_resnet50_voc2012.pth" # <--- CHANGE THIS TO YOUR FILENAME

# Number of classes your model was trained on (VOC has 20 classes + 1 background = 21)
NUM_SEG_CLASSES = 21

# Instantiate the DeepLabV3 ResNet50 model structure *without* pre-trained weights
model_seg = segmentation.deeplabv3_resnet50(weights=None, num_classes=NUM_SEG_CLASSES)
print(f"Segmentation model structure created (num_classes={NUM_SEG_CLASSES}).")

# Load your custom trained weights
try:
    model_seg.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=device))
    print(f"Segmentation model weights loaded successfully from {SEG_MODEL_PATH}")
except FileNotFoundError:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Error: Segmentation model weights file not found at {SEG_MODEL_PATH}")
    print(f"The segmentation endpoint will likely fail or produce random results.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
except Exception as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Error loading segmentation weights from {SEG_MODEL_PATH}: {e}")
    print(f"Ensure the file exists and the state_dict keys match the model structure.")
    print(f"The segmentation endpoint will likely fail or produce random results.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

model_seg.to(device)
model_seg.eval()

SEG_MEAN = [0.485, 0.456, 0.406]
SEG_STD = [0.229, 0.224, 0.225]
SEG_TARGET_IMG_SIZE = (256, 256)
SEG_INTERPOLATION = transforms.InterpolationMode.BILINEAR
print(f"Defining segmentation transforms with resize to {SEG_TARGET_IMG_SIZE}")
print(f"Using EXPLICIT normalization mean={SEG_MEAN}, std={SEG_STD} for segmentation.")
preprocess_seg = transforms.Compose([
    transforms.Resize(SEG_TARGET_IMG_SIZE, interpolation=SEG_INTERPOLATION), # Explicit size and interpolation
    transforms.ToTensor(),              # Convert PIL [0,255] to Tensor [0,1]
    transforms.Normalize(mean=SEG_MEAN, std=SEG_STD) # Use explicit constants
])

# --- Define Palette and Class Names for Segmentation ---

# Create a color palette for VOC classes (21 classes including background)
# Ensure this order matches the class indices your model outputs (0-20)
voc_palette = [
    (0, 0, 0),       # 0=background
    (128, 0, 0),     # 1=aeroplane
    (0, 128, 0),     # 2=bicycle
    (128, 128, 0),   # 3=bird
    (0, 0, 128),     # 4=boat
    (128, 0, 128),   # 5=bottle
    (0, 128, 128),   # 6=bus
    (128, 128, 128), # 7=car
    (64, 0, 0),      # 8=cat
    (192, 0, 0),     # 9=chair
    (64, 128, 0),    # 10=cow
    (192, 128, 0),   # 11=dining table
    (64, 0, 128),    # 12=dog
    (192, 0, 128),   # 13=horse
    (64, 128, 128),  # 14=motorbike
    (192, 128, 128), # 15=person
    (0, 64, 0),      # 16=potted plant
    (128, 64, 0),    # 17=sheep
    (0, 192, 0),     # 18=sofa
    (128, 192, 0),   # 19=train
    (0, 64, 128)     # 20=tv/monitor
]

# Define the corresponding class names IN THE SAME ORDER as the palette
class_names_seg = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "dining table", "dog",
    "horse", "motorbike", "person", "potted plant", "sheep", "sofa",
    "train", "tv/monitor"
]

# --- Sanity Checks ---
if len(voc_palette) != NUM_SEG_CLASSES:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"CRITICAL WARNING: Palette length ({len(voc_palette)}) doesn't match NUM_SEG_CLASSES ({NUM_SEG_CLASSES}).")
    print(f"Adjust palette or NUM_SEG_CLASSES. Mask colors and detected class info will be incorrect.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Attempt to pad/truncate for basic functionality, but results will be wrong
    if len(voc_palette) < NUM_SEG_CLASSES:
        voc_palette += [(0,0,0)] * (NUM_SEG_CLASSES - len(voc_palette))
    else:
        voc_palette = voc_palette[:NUM_SEG_CLASSES]

if len(class_names_seg) != NUM_SEG_CLASSES:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"CRITICAL WARNING: Class names list length ({len(class_names_seg)}) doesn't match NUM_SEG_CLASSES ({NUM_SEG_CLASSES}).")
    print(f"Adjust class names list or NUM_SEG_CLASSES. Detected class info will be incorrect.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Attempt to pad/truncate for basic functionality, but results will be wrong
    if len(class_names_seg) < NUM_SEG_CLASSES:
        class_names_seg += [f"Unknown_{i}" for i in range(len(class_names_seg), NUM_SEG_CLASSES)]
    else:
        class_names_seg = class_names_seg[:NUM_SEG_CLASSES]
# Remove the old class_label_map as it's replaced by class_names_seg
# class_label_map = [ ... ] # This is no longer needed and was incorrectly formatted


# --- Helper Function for Segmentation Output ---
# (No changes needed in this function itself, relies on correct palette)
def create_segmentation_mask(pred_mask_tensor, palette):
    """Creates a colored PIL image mask from a prediction tensor."""
    h, w = pred_mask_tensor.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Use the provided palette length for bounds checking
    palette_len = len(palette)
    max_idx = pred_mask_tensor.max().item() # Find the highest index actually present

    if max_idx >= palette_len:
        print(f"Warning in create_segmentation_mask: Max predicted index ({max_idx}) is out of bounds for palette size ({palette_len}). Some classes may render as black.")
        # Optionally pad the palette dynamically for rendering, though this indicates a mismatch upstream
        # palette = palette + [(0,0,0)] * (max_idx - palette_len + 1)
        # palette_len = len(palette) # Update length if padded

    for class_idx, color in enumerate(palette):
        # Optimization: Only check indices up to the max found index if needed,
        # but iterating through the whole defined palette is safer for consistency.
        # if class_idx > max_idx:
        #      break
        if class_idx < palette_len: # Ensure we don't go out of bounds of the original palette
            mask = pred_mask_tensor == class_idx
            rgb_mask[mask] = color
        # Pixels with indices >= palette_len will remain black (0,0,0) by default

    return Image.fromarray(rgb_mask)

# --- API Endpoints ---

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    try:
        image = Image.open(image_file).convert('RGB')
        # Apply original classification transforms (Resize(36)/CenterCrop(32))
        image_tensor = transform_clf(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_clf(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            # Check index bounds before accessing class name
            if 0 <= predicted_idx.item() < len(class_names_clf):
                predicted_class = class_names_clf[predicted_idx.item()]
            else:
                predicted_class = f"Unknown_Index_{predicted_idx.item()}"
                print(f"Warning: Classification index {predicted_idx.item()} out of bounds for class names list (len={len(class_names_clf)}).")


        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence.item()
        })
    except Exception as e:
        app.logger.error(f"Classification Error: {e}", exc_info=True)
        return jsonify({'error': f'Error during classification: {str(e)}'}), 500

@app.route('/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    try:
        input_image = Image.open(image_file).convert('RGB')

        # Apply segmentation preprocessing
        input_batch = preprocess_seg(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_seg(input_batch)['out']

        # Get the predicted class index for each pixel
        output_predictions = output.argmax(1).squeeze(0).cpu() # Shape: [H, W]

        # Create the colored mask image using the correct palette
        mask_image = create_segmentation_mask(output_predictions, voc_palette)

        # Encode the mask image to base64
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # --- Get detected class info (index, name, color) ---
        detected_indices = torch.unique(output_predictions).tolist() # Get unique class indices present in the mask
        detected_class_info = [] # List to store info for each detected class

        for index in detected_indices:
            class_info = {'index': index}

            # Get Class Name based on index
            if 0 <= index < len(class_names_seg):
                class_info['name'] = class_names_seg[index]
            else:
                # Handle cases where the model predicts an index outside the defined range
                class_info['name'] = f"Unknown_Class_{index}"
                print(f"Warning: Detected segmentation index {index} is out of bounds for class names list (len={len(class_names_seg)}).")

            # Get Color based on index
            if 0 <= index < len(voc_palette):
                # Convert tuple to list for JSON serialization
                class_info['color'] = list(voc_palette[index])
            else:
                 # Handle cases where the index is outside the palette range
                class_info['color'] = [0, 0, 0] # Default to black
                print(f"Warning: Detected segmentation index {index} is out of bounds for palette list (len={len(voc_palette)}). Using black.")

            detected_class_info.append(class_info)

        # Sort the detected classes by index for consistency (optional)
        detected_class_info.sort(key=lambda x: x['index'])

        return jsonify({
            'segmentation_mask': f'data:image/png;base64,{img_str}',
            'detected_classes': detected_class_info # Return the list of detected class details
        })

    except Exception as e:
        app.logger.error(f"Segmentation Error: {e}", exc_info=True)
        return jsonify({'error': f'Error during segmentation: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
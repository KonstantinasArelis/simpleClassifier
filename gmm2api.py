#python3 -m venv .venv
#. .venv/bin/activate
#pip install Flask
#pip install Pillow
#pip install torch
#pip install torchvision
#pip install flask_cors
#pip install "numpy<2"
#flask --app gmm2api.py run
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

# Define classification preprocessing steps
transform_clf = transforms.Compose([ # Rename to avoid conflict
    transforms.Resize(36),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    # Add normalization if your model was trained with it
    # transforms.Normalize(mean=[...], std=[...])
])

# Class names (for CIFAR-10 - adjust if your model uses different classes)
class_names_clf = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# --- Segmentation Model Setup ---

# Define the path to YOUR saved segmentation model weights
SEG_MODEL_PATH = "deeplabv3_resnet50_voc2012.pth" # <--- CHANGE THIS TO YOUR FILENAME

# Number of classes your model was trained on (VOC has 20 classes + 1 background = 21)
NUM_SEG_CLASSES = 21

# Instantiate the DeepLabV3 ResNet50 model structure *without* pre-trained weights
# IMPORTANT: Specify the number of classes your custom model outputs
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

# --- IMPORTANT: Preprocessing for Custom Model ---
# You MUST use the SAME preprocessing steps here as you used during the
# training of your custom model saved in SEG_MODEL_PATH.
# If you fine-tuned the standard torchvision model and used its default transforms,
# you can get them like this:
try:
    weights_helper = segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS
    preprocess_seg = weights_helper.transforms()
    print("Using standard torchvision preprocessing for segmentation model.")
    # If your training used different normalization, resizing, etc., define it manually:
    # preprocess_seg = transforms.Compose([
    #     transforms.Resize(...), # Use the size your model expects
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[YOUR_MEAN_R, YOUR_MEAN_G, YOUR_MEAN_B],
    #                          std=[YOUR_STD_R, YOUR_STD_G, YOUR_STD_B]),
    # ])
    # print("Using CUSTOM preprocessing steps for segmentation model.")
except Exception as e:
     print(f"Could not get standard transforms, define preprocessing manually if needed. Error: {e}")
     # Define a minimal fallback if weights can't be loaded
     preprocess_seg = transforms.Compose([transforms.ToTensor()])


# --- Class Names and Palette (Assumes Standard VOC) ---
# These should also match your training setup. If you used the standard
# 21 VOC classes in the standard order, this should be correct.
# If your classes or their order differ, you MUST update class_names_seg
# and voc_palette accordingly.
try:
    weights_helper = segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS
    class_names_seg = weights_helper.meta["categories"]
    print(f"Using standard VOC class names for segmentation: {class_names_seg}")
except Exception as e:
    print(f"Could not get standard VOC class names. Using placeholder names.")
    # Fallback if weights meta can't be read
    class_names_seg = [f"Class_{i}" for i in range(NUM_SEG_CLASSES)]


# Create a color palette for VOC classes (21 classes including background)
# Ensure this matches the order and number of your NUM_SEG_CLASSES
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
# Adjust palette size if NUM_SEG_CLASSES is different
if len(voc_palette) != NUM_SEG_CLASSES:
    print(f"Warning: Palette length ({len(voc_palette)}) doesn't match NUM_SEG_CLASSES ({NUM_SEG_CLASSES}). Adjust palette.")
    # Simple fallback: repeat black or generate random colors if needed
    voc_palette = voc_palette[:NUM_SEG_CLASSES] + [(0,0,0)] * (NUM_SEG_CLASSES - len(voc_palette))


# --- Helper Function for Segmentation Output ---
# (No changes needed in this function itself)
def create_segmentation_mask(pred_mask_tensor, palette):
    """Creates a colored PIL image mask from a prediction tensor."""
    h, w = pred_mask_tensor.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Ensure palette length matches or exceeds max class index
    max_idx = pred_mask_tensor.max().item()
    if max_idx >= len(palette):
        print(f"Warning: Max predicted index ({max_idx}) is out of bounds for palette size ({len(palette)}). Some classes may render as black.")
        # Extend palette with black if needed for safety
        palette = palette + [(0,0,0)] * (max_idx - len(palette) + 1)

    for class_idx, color in enumerate(palette):
        if class_idx > max_idx: # Optimization: stop if beyond max predicted index
             break
        mask = pred_mask_tensor == class_idx
        rgb_mask[mask] = color
        
    return Image.fromarray(rgb_mask)

# --- API Endpoints ---

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    try:
        image = Image.open(image_file).convert('RGB')
        image_tensor = transform_clf(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_clf(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = class_names_clf[predicted_idx.item()]

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
        
        # Apply segmentation preprocessing (CRITICAL: must match training)
        input_batch = preprocess_seg(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_seg(input_batch)['out']
        
        output_predictions = output.argmax(1).squeeze(0).cpu()

        # Create colored mask (CRITICAL: palette must match classes)
        mask_image = create_segmentation_mask(output_predictions, voc_palette)

        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Get detected classes (CRITICAL: class_names_seg must match classes)
        detected_indices = torch.unique(output_predictions).tolist()
        detected_classes = []
        for i in detected_indices:
             if i < len(class_names_seg):
                 detected_classes.append(class_names_seg[i])
             else:
                 detected_classes.append(f"Unknown_Class_{i}") # Handle unexpected index


        return jsonify({
            'segmentation_mask': f'data:image/png;base64,{img_str}',
            'detected_classes': detected_classes
        })
    
    except Exception as e:
        app.logger.error(f"Segmentation Error: {e}", exc_info=True)
        return jsonify({'error': f'Error during segmentation: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
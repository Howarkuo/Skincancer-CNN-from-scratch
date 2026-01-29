import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Silence TensorFlow warnings

import visualkeras
from tensorflow.keras import layers, models
from collections import defaultdict
from PIL import ImageFont

# ---------------------------------------------------------
# 1. Define Model Architecture
# ---------------------------------------------------------
model = models.Sequential([
    # 1st Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), name='Conv1'),
    layers.MaxPooling2D((2, 2), name='Pool1'),

    # 2nd Block
    layers.Conv2D(64, (3, 3), activation='relu', name='Conv2'),
    layers.MaxPooling2D((2, 2), name='Pool2'),

    # 3rd Block
    layers.Conv2D(128, (3, 3), activation='relu', name='Conv3'),
    layers.MaxPooling2D((2, 2), name='Pool3'),

    # Classifier
    layers.Flatten(name='Flatten'),
    layers.Dense(512, activation='relu', name='Dense_512'),
    layers.Dropout(0.5, name='Dropout'),
    layers.Dense(1, activation='sigmoid', name='Output')
])

# ---------------------------------------------------------
# 2. Fix for Keras 3 (Missing output_shape)
# ---------------------------------------------------------
for layer in model.layers:
    if not hasattr(layer, 'output_shape'):
        # Manually attach the shape tuple so visualkeras can find it
        layer.output_shape = tuple(layer.output.shape)

# ---------------------------------------------------------
# 3. Define the Text Callable (Logic for labels)
# ---------------------------------------------------------
def get_layer_text(layer_index, layer):
    # Toggle text above/below for cleaner look
    above = bool(layer_index % 2)

    # Get the shape (handling the None batch dimension)
    # We use the manually attached output_shape from step 2
    raw_shape = layer.output_shape
    
    # Filter out 'None' (batch size)
    clean_shape = [str(x) for x in raw_shape if x is not None]

    # Format text: 
    # If shape is [224, 224, 32] -> "224x224 \n 32"
    if len(clean_shape) == 3:
        text = f"{clean_shape[0]}x{clean_shape[1]}\n{clean_shape[2]}"
    elif len(clean_shape) == 1:
        text = f"{clean_shape[0]}"
    else:
        text = "x".join(clean_shape)

    return text, above

# ---------------------------------------------------------
# 4. Custom Colors (Matching your reference)
# ---------------------------------------------------------
color_map = defaultdict(dict)
color_map[layers.Conv2D]['fill'] = '#e34a6f'       # Red/Pink
color_map[layers.MaxPooling2D]['fill'] = '#00ce9a' # Teal
color_map[layers.Dense]['fill'] = '#f7b538'        # Yellow
color_map[layers.Flatten]['fill'] = '#808080'      # Gray
color_map[layers.Dropout]['fill'] = '#ffc0cb'      # Light Pink

# ---------------------------------------------------------
# 5. Generate and Save
# ---------------------------------------------------------
print("Generating diagram with text labels...")

# You might need a larger font size for high-res images
# If you have a specific font file, load it here:
# font = ImageFont.truetype("arial.ttf", 32) 
# Otherwise, visualkeras uses a default small font.

visualkeras.layered_view(
    model, 
    to_file='cnn_with_text.png', 
    legend=True, 
    color_map=color_map, 
    scale_xy=1, 
    scale_z=0.5, 
    spacing=50,                  # Increased spacing so text fits
    text_callable=get_layer_text # <--- THIS ADDS THE TEXT
)

print(" Saved as 'cnn_with_text.png'")
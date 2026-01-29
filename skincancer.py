import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- 1. GPU SETUP ---
# Use the first Quadro RTX 8000 (GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check if GPU is detected
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" Running on GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("  No GPU found! Running on CPU (will be slow).")

# --- 2. DATA PREPARATION ---
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescaling for testing
test_datagen = ImageDataGenerator(rescale=1./255)

print("\n--- Loading Data ---")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False # Important for correct classification report
)

# --- 3. BUILD THE CNN MODEL (Your Custom Architecture) ---
print("\n--- Building Custom CNN Model ---")
model = models.Sequential([
    # 1st Convolution Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    # 2nd Convolution Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 3rd Convolution Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flattening Layer
    layers.Flatten(),

    # Fully Connected Layers
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout to reduce overfitting
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. CALLBACKS ---
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
    # Saves the best version of your custom model
    ModelCheckpoint("my_custom_skin_model.keras", save_best_only=True, verbose=1)
]

# --- 5. TRAIN ---
print("\n--- Starting Training ---" )
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=30,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)

# --- 6. EVALUATION ---
print("\n--- generating Report ---")
# Predict on the test set
test_pred = model.predict(test_generator, verbose=1)
test_pred_labels = (test_pred > 0.5).astype("int32")
test_true_labels = test_generator.classes

# Print Report
report = classification_report(test_true_labels, test_pred_labels, target_names=list(test_generator.class_indices.keys()))
print(report)

# Save report to file
with open("classification_report.txt", "w") as f:
    f.write(report)
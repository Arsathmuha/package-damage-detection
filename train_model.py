import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────
DATASET_DIR   = "dataset"          # Path to the dataset folder
IMG_SIZE      = (224, 224)          # Resize all images to 224×224
BATCH_SIZE    = 16                  # Number of images per training batch
EPOCHS        = 10                  # How many times the model sees the full dataset
LEARNING_RATE = 0.0001              # Learning rate for the Adam optimiser
MODEL_SAVE    = "package_quality_model.h5"   # Where to save the trained model

# ──────────────────────────────────────────────
# 2. Data Loading & Augmentation
# ──────────────────────────────────────────────

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,             # Normalise pixel values to [0, 1]
    validation_split=0.2,          # 80 % train, 20 % validation
    rotation_range=20,             # Randomly rotate images up to 20°
    width_shift_range=0.2,         # Randomly shift horizontally
    height_shift_range=0.2,        # Randomly shift vertically
    horizontal_flip=True,          # Randomly flip left ↔ right
    zoom_range=0.2,                # Randomly zoom in/out
)

print("Loading training images …")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",           # Two classes → binary classification
    subset="training",             # Use the 80 % training split
    shuffle=True,
)

print("\nLoading validation images …")
val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",           # Use the 20 % validation split
    shuffle=False,
)

# Print detected classes so you know the label mapping
print(f"\nClass indices: {train_generator.class_indices}")
# Example output: {'damaged': 0, 'good': 1}

# ──────────────────────────────────────────────
# 3. Build the Model (Transfer Learning)
# ──────────────────────────────────────────────
# We load ResNet50 pretrained on ImageNet but WITHOUT the top classification
# layers. Then we add our own layers on top for binary classification.

base_model = ResNet50(
    weights="imagenet",            # Use pretrained ImageNet weights
    include_top=False,             # Remove the original classifier head
    input_shape=(224, 224, 3),     # Expect 224×224 RGB images
)

# Freeze the base model so its weights are NOT updated during training.
# This preserves the powerful features ResNet already learned.
base_model.trainable = False

# Add custom classification layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)   # Condense feature maps to a 1-D vector
x = Dense(128, activation="relu")(x)  # Fully-connected hidden layer
x = Dropout(0.3)(x)               # Drop 30 % of neurons to reduce overfitting
output = Dense(1, activation="sigmoid")(x)  # Single output: 0 = damaged, 1 = good

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",    # Standard loss for binary classification
    metrics=["accuracy"],
)

# Show a summary of the model architecture
model.summary()

with tf.device('/GPU:0'):
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
    )

# ──────────────────────────────────────────────
# 4. Train the Model
# ──────────────────────────────────────────────
print("\n🚀 Starting training …\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
)

# ──────────────────────────────────────────────
# 5. Save the Trained Model
# ──────────────────────────────────────────────
model.save(MODEL_SAVE)
print(f"\n✅ Model saved to: {MODEL_SAVE}")

# ──────────────────────────────────────────────
# 6. Plot Training & Validation Accuracy / Loss
# ──────────────────────────────────────────────
# Two side-by-side charts: accuracy on the left, loss on the right.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# — Accuracy plot —
ax1.plot(history.history["accuracy"],     label="Training Accuracy")
ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# — Loss plot —
ax2.plot(history.history["loss"],     label="Training Loss")
ax2.plot(history.history["val_loss"], label="Validation Loss")
ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("📊 Training plot saved to: training_history.png")
plt.show()

print("\n🎉 Training complete!")

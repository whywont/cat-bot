import os
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import collections

BASE_DIR = "../cat_profiles"
CROP_DIR = "../cropped_profiles"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 15
YOLO_MODEL_PATH = "../yolov8n.pt"

# Step 1: Auto-crop all training images using YOLO
os.makedirs(CROP_DIR, exist_ok=True)
model_yolo = YOLO(YOLO_MODEL_PATH)

for folder in os.listdir(BASE_DIR):
    src_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(src_path):
        continue

    dst_path = os.path.join(CROP_DIR, folder)
    os.makedirs(dst_path, exist_ok=True)

    for file in os.listdir(src_path):
        img_path = os.path.join(src_path, file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        cat_crops_saved = 0
        results = model_yolo(image, verbose=False)[0]
        for i, (box, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.cls)):
            if int(cls) != 15:
                continue
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, IMG_SIZE)
            out_name = f"{os.path.splitext(file)[0]}_crop{i}.jpg"
            cv2.imwrite(os.path.join(dst_path, out_name), crop)
            cat_crops_saved+=1

        if cat_crops_saved == 0:
            print(f"??  No cats detected in {file}, saving full image")
            full_image = cv2.resize(image, IMG_SIZE)
            out_name = f"{os.path.splitext(file)[0]}_full.jpg"
            cv2.imwrite(os.path.join(dst_path, out_name), full_image)
print("\u2705 Cropping complete. Now training classifier...")

# Step 2: Train on cropped, augmented images
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    CROP_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)

val_data = datagen.flow_from_directory(
    CROP_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb',
    shuffle=False
)


model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # Smaller than 128x128
    layers.Conv2D(16, 3, activation='relu'),  # Fewer filters
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),  # Instead of Flatten
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Compute class weights
y_classes = train_data.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_classes),
    y=y_classes
)
class_weights = dict(enumerate(class_weights))

early_stop = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[early_stop],
    class_weight=class_weights
)

model.save("cat_classifier.h5")
print("\u2705 Model saved as cat_classifier.h5")
print("Class mapping:", train_data.class_indices)

# Evaluate with precision/recall report
val_data.reset()
y_pred = np.argmax(model.predict(val_data), axis=1)
y_true = val_data.classes
print(classification_report(y_true, y_pred, target_names=list(val_data.class_indices.keys())))

# Manual sanity check of a few predictions
batch_x, batch_y = next(val_data)
preds = model.predict(batch_x)
for i in range(len(preds)):
    print("Predicted:", np.argmax(preds[i]), "True:", np.argmax(batch_y[i]))

counts = collections.Counter(train_data.classes)
print("Training class distribution:", counts)

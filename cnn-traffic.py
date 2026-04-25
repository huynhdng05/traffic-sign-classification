#!/usr/bin/env python
# coding: utf-8

# In[1]:


!pip install tensorflow

# In[2]:


import os
import pandas as pd
import numpy as np
from skimage.io import imread
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam,AdamW
# from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def load_data(input_size=(224, 224), data_path='/kaggle/input/datasets/hunhduyng/datasets-traffic/GTSRB/Final_Training/Images'):
    pixels = []
    labels = []
    for dir in os.listdir(data_path):
        if dir == '.DS_Store':
            continue
        class_dir = os.path.join(data_path, dir)
        info_file = pd.read_csv(os.path.join(class_dir, "GT-" + dir + '.csv'), sep=';')
        for _, row in info_file.iterrows():
            pixel = imread(os.path.join(class_dir, row['Filename']))
            pixel = pixel[row['Roi.Y1']:row['Roi.Y2'], row['Roi.X1']:row['Roi.X2'], :]
            img = cv2.resize(pixel, input_size)
            pixels.append(img)
            labels.append(row['ClassId'])
    return np.array(pixels), np.array(labels)

def split_train_val_test_data(pixels, labels):
    pixels = np.array(pixels)
    labels = to_categorical(labels)
    randomize = np.arange(len(pixels))
    np.random.shuffle(randomize)
    X = pixels[randomize]
    y = labels[randomize]
    train_size = int(X.shape[0] * 0.6)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    val_size = int(X_val.shape[0] * 0.5)
    X_val, X_test = X_val[:val_size], X_val[val_size:]
    y_val, y_test = y_val[:val_size], y_val[val_size:]
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_model(input_shape=(224, 224, 3), filter_size=(3, 3), pool_size=(2, 2), output_size=43):
    model = Sequential([
        Conv2D(16, filter_size, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(16, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Conv2D(32, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Conv2D(64, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Flatten(),
        Dense(2048, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=AdamW(learning_rate=1e-4), metrics=['accuracy'])
    return model

# Load and preprocess data
pixels, labels = load_data()
X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test_data(pixels, labels)


# Build and train the model
model = build_model(input_shape=(224, 224, 3), output_size=43)
epochs = 100
batch_size =64
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# # Define K-Fold Cross Validation
# num_folds = 5
# kfold = KFold(n_splits=num_folds, shuffle=True)

# # K-fold Cross Validation model evaluation
# accuracy_list = []
# loss_list = []
# fold_idx = 1

# for train_ids, val_ids in kfold.split(pixels, labels):
#     model = build_model(input_shape=(64, 64, 3), output_size=43)
#     print("Training fold ", fold_idx)
#     model.fit(pixels[train_ids], to_categorical(labels[train_ids]), batch_size=batch_size, epochs=epochs, verbose=1)
#     scores = model.evaluate(pixels[val_ids], to_categorical(labels[val_ids]), verbose=0)
#     print("Finished fold ", fold_idx)
#     accuracy_list.append(scores[1] * 100)
#     loss_list.append(scores[0])
#     fold_idx += 1


# # Evaluate model
# print("Evaluating the model on the test set...")
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")

# Make predictions
y_hat = model.predict(X_test)
y_pred = np.argmax(y_hat, axis=1)
y_test_label = np.argmax(y_test, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test_label, y_pred)
precision = precision_score(y_test_label, y_pred, average='macro')
recall = recall_score(y_test_label, y_pred, average='macro')
f1 = f1_score(y_test_label, y_pred, average='macro')
auc = roc_auc_score(y_test, y_hat, multi_class='ovr')
matrix = confusion_matrix(y_test_label, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 score: {f1}')
print(f'ROC AUC: {auc}')
print("Confusion Matrix:")
print(matrix)
# print(f'K-Fold CV Accuracy: {np.mean(accuracy_list)} (+/- {np.std(accuracy_list)})')
# print(f'K-Fold CV Loss: {np.mean(loss_list)}')

# Save the model
model.save("traffic_sign_model.keras")


# In[7]:


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. CẤU HÌNH BAN ĐẦU & MIXED PRECISION
# ==========================================
# THAY ĐỔI ĐƯỜNG DẪN NÀY NẾU CẦN
data_path = '/kaggle/input/datasets/hunhduyng/datasets-traffic/GTSRB/Final_Training/Images'

# Bật Mixed Precision để tối đa hóa tốc độ trên GPU Kaggle
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Siêu tham số (Hyperparameters) cho ViT
input_shape = (72, 72, 3) # Resize ảnh biển báo về 72x72
patch_size = 6
num_patches = (input_shape[0] // patch_size) ** 2
num_classes = 43 # Bắt buộc là 43 cho tập GTSRB
projection_dim = 64
num_heads = 4
transformer_layers = 4
transformer_units = [projection_dim * 2, projection_dim]
mlp_head_units = [1024, 512]
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 64 # Có thể tăng lên 128 nếu GPU còn dư RAM
num_epochs = 50

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (Hỗ trợ file .ppm của GTSRB)
# ==========================================
print("⏳ Đang chuẩn bị luồng dữ liệu...")

# Data Augmentation cho tập Train (Rất quan trọng với ViT)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # Lấy 20% làm Validation
    rotation_range=15,    # Xoay ảnh tối đa 15 độ
    zoom_range=0.15,      # Zoom in/out 15%
    width_shift_range=0.1,
    height_shift_range=0.1,
    # Tuyệt đối KHÔNG dùng horizontal_flip với biển báo giao thông!
)

# Chỉ Rescale cho tập Validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    data_path,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False, # Không xáo trộn tập Val để lúc đánh giá không bị sai lệch
    seed=42
)

# ==========================================
# 3. KIẾN TRÚC VISION TRANSFORMER
# ==========================================
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    
    # Do Data Generator đã làm rescale/augmentation, truyền thẳng vào Patches
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Các khối Transformer Block
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    # Classification Head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    # LƯU Ý QUAN TRỌNG: Ép dtype='float32' để tránh lỗi Mixed Precision
    logits = layers.Dense(num_classes, dtype='float32')(features)
    
    return keras.Model(inputs=inputs, outputs=logits)

# ==========================================
# 4. HUẤN LUYỆN MÔ HÌNH
# ==========================================
keras.backend.clear_session()
vit_classifier = create_vit_classifier()

optimizer = keras.optimizers.AdamW(
    learning_rate=learning_rate, 
    weight_decay=weight_decay
)

vit_classifier.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Callbacks
checkpoint_filepath = "vit_gtsrb_best.keras"
callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor="val_accuracy", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
]

print("\n🚀 Bắt đầu huấn luyện...")
history = vit_classifier.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    callbacks=callbacks
)

# ==========================================
# 5. ĐÁNH GIÁ VÀ VẼ BIỂU ĐỒ
# ==========================================
# Vẽ biểu đồ Accuracy và Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Predict trên tập Validation để xem Metrics
print("\n📊 Đang đánh giá chi tiết trên tập Validation...")
# Cần reset generator trước khi predict để mảng kết quả khớp thứ tự
val_generator.reset() 
y_pred_prob = vit_classifier.predict(val_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_generator.classes

print("\n--- BÁO CÁO PHÂN LOẠI (Classification Report) ---")
print(classification_report(y_true, y_pred))

vit_classifier.save('vit_gtsrb_final.keras')
print("\n✅ Đã lưu mô hình thành công!")

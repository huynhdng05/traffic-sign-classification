import cv2
import tensorflow as tf
from keras import layers, Model
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from PIL import Image, ImageFont, ImageDraw
import os

# --- 1. ĐỊNH NGHĨA CẤU TRÚC (Khớp chính xác với Model Summary) ---

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
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

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection.units})
        return config

def create_vit_model():
    image_size, patch_size = 72, 6
    num_patches = (image_size // patch_size) ** 2 # 144
    projection_dim = 64
    
    # Theo Summary của bạn: Có 4 khối Transformer chính (add_7 là layer cuối trước flatten)
    # Mỗi khối gồm: LayerNorm -> MultiHeadAtt -> Add -> LayerNorm -> MLP (Dense->Dropout->Dense->Dropout) -> Add
    transformer_layers = 4 
    transformer_units = [projection_dim * 2, projection_dim] # [128, 64]

    input_layer = layers.Input(shape=(image_size, image_size, 3), name="input_layer")
    
    # Khối tiền xử lý
    p = Patches(patch_size, name="patches")(input_layer)
    encoded = PatchEncoder(num_patches, projection_dim, name="patch_encoder")(p)

    # Các lớp Transformer Blocks
    for i in range(transformer_layers):
        # Part 1: Self-attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        # Summary của bạn Key_dim=64, num_heads thường là 4 để khớp param 66,368
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attn, encoded])
        
        # Part 2: MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        y = layers.Dense(transformer_units[0], activation=tf.nn.gelu)(x3)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(transformer_units[1], activation=tf.nn.gelu)(y)
        y = layers.Dropout(0.1)(y)
        encoded = layers.Add()([y, x2])

    # Khối Output (Khớp với Flatten -> Dense 1024 -> Dense 512 -> Dense 43)
    rep = layers.LayerNormalization(epsilon=1e-6)(encoded)
    rep = layers.Flatten()(rep)
    rep = layers.Dropout(0.5)(rep)
    
    rep = layers.Dense(1024, activation=tf.nn.gelu)(rep)
    rep = layers.Dropout(0.5)(rep)
    
    rep = layers.Dense(512, activation=tf.nn.gelu)(rep)
    rep = layers.Dropout(0.5)(rep)
    
    output = layers.Dense(43, activation="softmax")(rep)
    
    return Model(inputs=input_layer, outputs=output)

# --- 2. KHỞI TẠO APP VÀ MÔ HÌNH ---
app = FastAPI()
model = create_vit_model()

# Build model với dummy input
dummy_input = np.zeros((1, 72, 72, 3), dtype=np.float32)
_ = model(dummy_input)

WEIGHTS_PATH = 'vit_gtsrb_final.keras'

if os.path.exists(WEIGHTS_PATH):
    try:
        # Sử dụng load_model nếu file là .keras đầy đủ, 
        # hoặc load_weights nếu chỉ có trọng số. Ở đây ta dùng load_weights cho an toàn với create_vit_model.
        model.load_weights(WEIGHTS_PATH)
        print("✅ Đã khớp cấu trúc và nạp trọng số thành công!")
    except Exception as e:
        print(f"❌ Lỗi nạp trọng số: {e}")
else:
    print(f"⚠️ Không tìm thấy file {WEIGHTS_PATH}")

CLASSES = [
    'Giới hạn tốc độ (20km/h)', 'Giới hạn tốc độ (30km/h)', 'Giới hạn tốc độ (50km/h)', 
    'Giới hạn tốc độ (60km/h)', 'Giới hạn tốc độ (70km/h)', 'Giới hạn tốc độ (80km/h)', 
    'Hết giới hạn 80km/h', 'Tốc độ (100km/h)', 'Tốc độ (120km/h)', 'Cấm vượt', 
    'Cấm tải vượt', 'Đường ưu tiên', 'Nhường đường', 'STOP', 'Cấm xe', 'Cấm tải', 
    'Cấm vào', 'Nguy hiểm', 'Cua trái', 'Cua phải', 'Cua đôi', 'Đường gồ ghề', 
    'Trơn trượt', 'Đường hẹp', 'Công trường', 'Tín hiệu đèn', 'Người đi bộ', 
    'Trẻ em', 'Xe đạp', 'Băng tuyết', 'Động vật', 'Hết mọi lệnh cấm', 'Rẽ phải', 
    'Rẽ trái', 'Đi thẳng', 'Thẳng/Phải', 'Thẳng/Trái', 'Bên phải', 'Bên trái', 
    'Vòng xuyến', 'Hết đường ưu tiên', 'Cấm đỗ xe'
]

# --- 3. TIỆN ÍCH VÀ STREAMING ---

def draw_text_vietnamese(image, text, position):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: 
        # Đảm bảo bạn có file font này trong thư mục hoặc đổi đường dẫn
        font = ImageFont.truetype("arial.ttf", 24) 
    except: 
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        
        # Tiền xử lý
        img_in = cv2.resize(frame, (72, 72))
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB) / 255.0
        img_in = np.expand_dims(img_in, axis=0).astype(np.float32)

        # Dự đoán
        preds = model.predict(img_in, verbose=0)
        idx = np.argmax(preds[0])
        conf = preds[0][idx]

        if conf > 0.5:
            label = CLASSES[idx] if idx < len(CLASSES) else "Unknown"
            text = f"{label} ({conf*100:.1f}%)"
            # Vẽ nền đen cho chữ dễ đọc
            cv2.rectangle(frame, (0, 0), (500, 40), (0, 0, 0), -1)
            frame = draw_text_vietnamese(frame, text, (10, 5))

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get('/')
def index():
    return {"status": "Running", "endpoint": "/video_feed"}

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
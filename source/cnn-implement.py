import cv2
import keras
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from PIL import Image, ImageFont, ImageDraw
import os

app = FastAPI()

# --- CẤU HÌNH ---
# 1. Đường dẫn mô hình
MODEL_PATH = "cnn_traffic_sign_model.keras"

# 2. Danh sách 43 biển báo giao thông (Thứ tự chuẩn bộ dữ liệu GTSRB)
CLASSES = [
    "Tốc độ tối đa 20km/h", "Tốc độ tối đa 30km/h", "Tốc độ tối đa 50km/h", 
    "Tốc độ tối đa 60km/h", "Tốc độ tối đa 70km/h", "Tốc độ tối đa 80km/h",
    "Hết hạn chế tốc độ 80km/h", "Tốc độ tối đa 100km/h", "Tốc độ tối đa 120km/h",
    "Cấm vượt", "Cấm xe tải vượt", "Giao lộ ưu tiên", "Đường ưu tiên",
    "Nhường đường", "Dừng lại (STOP)", "Cấm tất cả phương tiện", "Cấm xe tải",
    "Cấm đi ngược chiều", "Nguy hiểm khác", "Cua ngoát trái", "Cua ngoát phải",
    "Cua ngoát kép", "Đường gồ ghề", "Trơn trượt", "Đường hẹp bên phải",
    "Công trường", "Tín hiệu đèn", "Người đi bộ", "Trẻ em", "Người đi xe đạp",
    "Băng tuyết", "Động vật hoang dã", "Hết hạn chế", "Rẽ phải phía trước",
    "Rẽ trái phía trước", "Đi thẳng", "Đi thẳng hoặc rẽ phải", "Đi thẳng hoặc rẽ trái",
    "Tránh bên phải", "Tránh bên trái", "Vòng xuyến", "Hết cấm vượt", "Hết cấm xe tải vượt"
]

# 3. Load mô hình khi khởi động
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print("--- Đã tải mô hình thành công! ---")
else:
    print(f"--- LỖI: Không tìm thấy file {MODEL_PATH} ---")

def draw_text_vietnamese(image, text, position):
    """Vẽ tiếng Việt có dấu lên ảnh dùng Pillow"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        # Sử dụng font Arial có sẵn trên Windows
        font = ImageFont.truetype("arial.ttf", 26)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    # Giảm độ phân giải camera để xử lý nhanh hơn (tùy chọn)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # --- BƯỚC 1: TIỀN XỬ LÝ (SỬA LỖI SHAPE: 224x224x3) ---
        # 1. Resize về đúng kích thước mô hình yêu cầu
        input_img = cv2.resize(frame, (224, 224))
        
        # 2. Chuyển BGR sang RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # 3. Chuẩn hóa pixel về [0, 1]
        input_img = input_img.astype('float32') / 255.0
        
        # 4. Thêm chiều batch để thành shape (1, 224, 224, 3)
        input_img = np.expand_dims(input_img, axis=0)

        # --- BƯỚC 2: DỰ ĐOÁN (INFERENCE) ---
        try:
            preds = model.predict(input_img, verbose=0)
            class_id = np.argmax(preds[0])
            confidence = np.max(preds[0])

            # --- BƯỚC 3: HIỂN THỊ KẾT QUẢ ---
            # Chỉ hiển thị nếu độ tin cậy > 60% để tránh nhiễu
            if confidence > 0.6: 
                label = CLASSES[class_id] if class_id < len(CLASSES) else f"ID: {class_id}"
                display_text = f"{label} ({confidence*100:.1f}%)"
                
                # Vẽ nền đen phía sau chữ
                cv2.rectangle(frame, (0, 0), (550, 60), (0, 0, 0), -1)
                frame = draw_text_vietnamese(frame, display_text, (15, 15))
                
        except Exception as e:
            print(f"Lỗi dự đoán: {e}")

        # --- BƯỚC 4: STREAMING ---
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get('/')
def index():
    return {"status": "Running", "visit": "/video_feed"}

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
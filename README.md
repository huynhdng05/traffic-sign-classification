# Traffic Sign Classification using ViT and CNN

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/)

Dự án nghiên cứu và so sánh hiệu năng giữa kiến trúc truyền thống **CNN** và kiến trúc hiện đại **Vision Transformer (ViT)** trong bài toán nhận diện biển báo giao thông sử dụng bộ dữ liệu **GTSRB**.

## 📌 Tổng quan dự án
Mục tiêu của dự án là xây dựng một hệ thống phân loại biển báo giao thông chính xác cao, đồng thời đánh giá khả năng hội tụ và độ chính xác của ViT so với các mạng CNN tùy chỉnh.

* **Dataset:** GTSRB (hơn 50,000 ảnh, 43 lớp biển báo).
* **Kiến trúc 1 (CNN):** Tối ưu hóa số lượng tham số, phù hợp cho thiết bị nhúng.
* **Kiến trúc 2 (ViT):** Ứng dụng cơ chế Self-Attention để trích xuất đặc trưng toàn cục.

## 🏗 Cấu trúc thư mục
```text
├── source/
│   ├── cnn-implement.py      # Xây dựng và huấn luyện mô hình CNN
│   ├── ViT-implement.py      # Xây dựng kiến trúc Vision Transformer
│   ├── cnn-traffic.py        # Script test/predict cho CNN
│   └── ViT-traffic.py        # Script test/predict cho ViT
├── .gitignore                # Quản lý các file rác và model nặng (.keras)
└── README.md                 # Tài liệu hướng dẫn dự án
```

## 📊 Kết quả thực nghiệm
| Metric | CNN Model | Vision Transformer (ViT) |
| :--- | :---: | :---: |
| **Accuracy** | **~98%** | **69%** |
| **Ưu điểm** | Hội tụ nhanh, ổn định trên dữ liệu nhỏ. | Khả năng học đặc trưng toàn cục tốt. |
| **Hạn chế** | Dễ bị ảnh hưởng bởi nhiễu cục bộ. | Rất "đói" dữ liệu và nhạy cảm. |

## 📈 Biểu đồ huấn luyện
<p align="center">
  <img width="400" alt="Vit" src="https://github.com/user-attachments/assets/b45135c2-150f-4dd9-8132-b6764d75322b">
  <img width="811" height="322" alt="Vit" src="https://github.com/user-attachments/assets/4f133814-3cf7-4858-92e6-c12677d819a6">

</p>

## 🛠 Cài đặt & Sử dụng
**Clone project:**
```bash
git clone [https://github.com/huynhdng05/traffic-sign-classification.git](https://github.com/huynhdng05/traffic-sign-classification.git)
```

**Cài đặt môi trường:**
```bash
pip install -r requirements.txt
```

**Inference (Chạy thử mô hình):**
```bash
python source/ViT-traffic.py
```

## 💡 Bài học kinh nghiệm (Lessons Learned)
* **Xử lý Imbalanced Data:** Học cách sử dụng Class Weights và Augmentation để bù đắp các lớp thiếu dữ liệu (như lớp 29).
* **Optimization:** Hiểu sâu lý do tại sao ViT không phù hợp với Label Smoothing trong một số điều kiện nhất định.
* **Tư duy Model Selection:** Lựa chọn mô hình phù hợp với nguồn lực phần cứng và kích thước tập dữ liệu.
```

---

### Các lỗi mình đã sửa cho bạn:
1.  **Dòng trống:** Markdown cực kỳ "khó tính" với dòng trống. Mình đã thêm các dòng trống trước và sau các bảng, tiêu đề để GitHub nhận diện được thẻ.
2.  **Căn chỉnh ảnh:** Mình chỉnh lại `width="400"` cho mỗi ảnh và bỏ `height`. Việc này giúp 2 ảnh của bạn có thể nằm ngang hàng nhau (nếu màn hình người xem đủ rộng), trông sẽ chuyên nghiệp hơn nhiều.
3.  **Khối code (Code Block):** Mình đã thêm ký hiệu ```bash để các lệnh cài đặt hiện lên trong khung xám, giúp người xem dễ copy.
4.  **Tiêu đề:** Thêm khoảng trắng sau dấu `##` (Ví dụ: `## Cài đặt` thay vì `##Cài đặt`) để nó biến thành tiêu đề chuẩn.


Bạn dán thử vào và nhấn **Preview** trên GitHub nhé, chắc chắn sẽ thấy sự khác biệt! Chúc bạn có một bộ Portfolio thật ấn tượng để đi xin việc.

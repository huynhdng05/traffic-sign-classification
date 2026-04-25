# Traffic Sign Classification using ViT and CNN

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dự án nghiên cứu và so sánh hiệu năng giữa kiến trúc truyền thống **CNN** và kiến trúc hiện đại **Vision Transformer (ViT)** trong bài toán nhận diện biển báo giao thông sử dụng bộ dữ liệu **GTSRB** (German Traffic Sign Recognition Benchmark).

## 📌 Tổng quan dự án
Mục tiêu của dự án là xây dựng một hệ thống phân loại biển báo giao thông chính xác cao, đồng thời đánh giá khả năng hội tụ và độ chính xác của ViT so với các mạng CNN tùy chỉnh.

* **Dataset:** GTSRB (hơn 50,000 ảnh, 43 lớp biển báo).
* **Kiến trúc 1 (CNN):** Tối ưu hóa số lượng tham số, phù hợp cho thiết bị nhúng.
* **Kiến trúc 2 (ViT):** Ứng dụng cơ chế Self-Attention để trích xuất đặc trưng toàn cục.

## 🏗 Cấu trúc thư mục
```text
├── source/
│   ├── cnn-implement.py      # Script test/predict cho CNN
│   ├── ViT-implement.py      # Script test/predict cho ViT 
│   ├── cnn-traffic.py        # Xây dựng và huấn luyện mô hình CNN
│   └── ViT-traffic.py        # Xây dựng kiến trúc Vision Transformer
├── .gitignore                # Quản lý các file rác và model nặng (.keras)
└── README.md                 # Tài liệu hướng dẫn dự án

Metric	   CNN Model	                                    Vision Transformer (ViT)
Accuracy	       98%                                     	        69%
Ưu điểm	   Hội tụ nhanh, ổn định trên dữ liệu nhỏ.    	  Khả năng học đặc trưng toàn cục (Global context) tốt.
Hạn chế	   Dễ bị ảnh hưởng bởi nhiễu cục bộ.	            Rất "đói" dữ liệu và nhạy cảm với Hyperparameters.

<img width="811" height="322" alt="Vit" src="https://github.com/user-attachments/assets/b45135c2-150f-4dd9-8132-b6764d75a4d8" />
<img width="814" height="328" alt="cnn" src="https://github.com/user-attachments/assets/77ce01c2-6db6-441d-964e-e913bef13e6e" />

Cài đặt & Sử dụng
Clone project:
git clone https://github.com/huynhdng05/traffic-sign-classification.git

Cài đặt môi trường:
pip install -r requirements.txt

Inference (Chạy thử mô hình):
python source/ViT-traffic.py

Bài học kinh nghiệm (Lessons Learned)
Xử lý Imbalanced Data: Học cách sử dụng Class Weights và Augmentation để bù đắp các lớp thiếu dữ liệu (như lớp 29).

Optimization: Hiểu sâu lý do tại sao ViT không phù hợp với Label Smoothing trong một số điều kiện nhất định.

Tư duy Model Selection: Biết cách lựa chọn mô hình phù hợp với nguồn lực phần cứng (Kaggle/Colab) và kích thước tập dữ liệu.

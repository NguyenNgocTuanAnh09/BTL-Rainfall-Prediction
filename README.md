# Rainfall Prediction using Machine Learning

# Giới thiệu
Dự án xây dựng mô hình Machine Learning để dự đoán khả năng có mưa dựa trên dữ liệu thời tiết.  
Mục tiêu là so sánh hiệu quả giữa hai thuật toán: Decision Tree (ID3) và Logistic Regression.

Dataset gồm các đặc trưng khí tượng:  
Temperature, Humidity, Wind Speed, Wind Bearing, Visibility, Pressure.

---

# Cấu trúc thư mục

PROJECT-RAINFALL-PREDICTION/
│
├── data/raw/weather.csv
├── src/
│ ├── id3luongmua.py
│ └── LogisticRegression.py
├── README.md

---

# Thuật toán sử dụng
- Decision Tree (ID3)
- Logistic Regression

---

# Quy trình thực hiện
1. Đọc dữ liệu từ file CSV  
2. Tiền xử lý dữ liệu  
3. Chia dữ liệu train/test (80% / 20%)  
4. Huấn luyện mô hình  
5. Dự đoán và đánh giá bằng Accuracy, Precision, Recall, F1-score  

---

# Cách chạy chương trình

Cài thư viện:
```bash
Chạy Decision Tree (ID3):

bash
python src/id3luongmua.py
Chạy Logistic Regression:

bash
python src/LogisticRegression.py
Kết quả
Mô hình	Accuracy
Decision Tree (ID3)	98.15%
Logistic Regression	99.3%

Logistic Regression cho kết quả tổng thể tốt hơn, đặc biệt ở lớp "Không mưa".

Mục tiêu học tập
Hiểu quy trình xây dựng mô hình Machine Learning cơ bản

So sánh hiệu quả giữa các thuật toán phân loại

Rèn luyện kỹ năng xử lý dữ liệu và đánh giá mô hình
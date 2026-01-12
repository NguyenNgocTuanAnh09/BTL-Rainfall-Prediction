import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Thông tin dữ liệu
data = pd.read_csv(r'D:\Project-Rainfall-Prediction\data\weather.csv')

# Hiển thị thông tin dữ liệu
print("Thông tin dữ liệu:")
print(data.info())
print("\n")

# Hiển thị thống kê cơ bản của dữ liệu
print("Thống kê cơ bản của dữ liệu:")
print(data.describe())
print("\n")



#B2: Tiền xử lí dữ liệu
#Kiểm tra giá trị thiếu 
print("Số lượng giá trị thiếu trong mỗi cột:")
print(data.isnull().sum())

# Xóa các hàng có giá trị thiếu
data.dropna(inplace=True)


# Bước 3: Xác định biến đầu vào (features) và nhãn (labels)
X = data[['Temperature_c', 'Humidity', 'Wind_Speed_kmh', 'Wind_Bearing_degrees', 'Visibility_km', 'Pressure_millibars']]
y = data['Rain'] 

# Chuyển đổi biến mục tiêu thành nhị phân (0: không mưa, 1: có mưa)
y = (y > 0).astype(int)


# B4: Biểu đồ này được dùng để hiển thị phân phối của các thuộc tính đầu vào
plt.figure(figsize=(12, 8))
for i, column in enumerate(X.columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[column], bins=30, kde=True)
    plt.title(f'Phân phối {column}')
    plt.xlabel(column)
    plt.ylabel('Tần suất')
plt.tight_layout()
plt.show()


# Định nghĩa hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Định nghĩa hàm tính toán hàm mất mát
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

# Hàm gradient descent
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    for i in range(iterations):
        # Tính toán dự đoán
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        # Tính gradient
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Cập nhật trọng số
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Tính toán hàm mất mát
        if i % 100 == 0:  # In ra mỗi 100 lần lặp
            loss = compute_loss(y, y_pred)
            print(f'Iteration {i}, Loss: {loss}')

    return weights, bias


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Khởi tạo và huấn luyện mô hình Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra cho Logistic Regression
y_pred_logistic = logistic_model.predict(X_test)

# Chạy gradient descent
weights, bias = gradient_descent(X_train, y_train, learning_rate=0.01, iterations=1000)

# Dự đoán trên tập kiểm tra
def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return (y_pred >= 0.5).astype(int)

y_pred = predict(X_test, weights, bias)


# Bước 6: Dự đoán với dữ liệu mới
new_data = pd.DataFrame({
    'Temperature_c': [-7.105555556],          # Nhiệt độ cao
    'Humidity': [0.88],              # Độ ẩm thấp
    'Wind_Speed_kmh': [15.1662],          # Tốc độ gió nhẹ
    'Wind_Bearing_degrees': [30],   # Hướng gió
    'Visibility_km': [9.9015],          # Tầm nhìn xa
    'Pressure_millibars': [1012.05]    # Áp suất cao
})

# Chuẩn hóa dữ liệu mới
new_data_scaled = scaler.transform(new_data)

# Dự đoán với giá trị mới
new_prediction = logistic_model.predict(new_data_scaled) 

# In ra kết quả dự đoán
if new_prediction[0] == 1:
    print("Dự đoán: Có mưa")
else:
    print("Dự đoán: Không mưa")


# Bước 7: Đánh giá mô hình Logistic Regression
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
report_logistic = classification_report(y_test, y_pred_logistic)

# In ra kết quả Logistic Regression
print(f"Logistic Regression - Accuracy: {accuracy_logistic}")
print("Logistic Regression - Classification Report:")
print(report_logistic)


# Giá trị từ báo cáo phân loại
precision = [0.98, 0.99]  # precision cho lớp 0 và 1
recall = [0.95, 1.00]     # recall cho lớp 0 và 1
f1_score = [0.97, 1.00]   # f1-score cho lớp 0 và 1
accuracy = [0.993, 0.993] # Độ chính xác cho cả hai lớp (giá trị cố định)
labels = ['Không mưa (0)', 'Có mưa (1)']

# Bước 8: Vẽ biểu đồ đường cho các chỉ số
plt.figure(figsize=(10, 6))

# Vẽ đường cho precision
plt.plot(labels, precision, marker='o', color='blue', linestyle='-', linewidth=2, label='Precision')
# Vẽ đường cho recall
plt.plot(labels, recall, marker='o', color='orange', linestyle='-', linewidth=2, label='Recall')
# Vẽ đường cho f1-score
plt.plot(labels, f1_score, marker='o', color='green', linestyle='-', linewidth=2, label='F1 Score')
# Vẽ đường cho accuracy
plt.plot(['Không mưa (0)', 'Có mưa (1)'], accuracy, marker='o', color='red', linestyle='-', linewidth=2, label='Accuracy')

plt.ylim(0, 1)  # Đặt giới hạn cho trục y
plt.title('Các chỉ số phân loại cho mô hình Logistic Regression')
plt.ylabel('Giá trị')
plt.grid()
plt.legend()
plt.show()
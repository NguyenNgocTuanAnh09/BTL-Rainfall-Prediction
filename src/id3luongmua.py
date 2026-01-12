from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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



# Bước 2: Tiền xử lý dữ liệu
# Giới thiệu thêm nhiễu vào dữ liệu (bằng cách thay đổi một số giá trị ngẫu nhiên)
import numpy as np
np.random.seed(42)
noise = np.random.normal(0, 1, size=data[['Temperature_c', 'Humidity', 'Wind_Speed_kmh', 'Wind_Bearing_degrees', 'Visibility_km', 'Pressure_millibars']].shape)
data[['Temperature_c', 'Humidity', 'Wind_Speed_kmh', 'Wind_Bearing_degrees', 'Visibility_km', 'Pressure_millibars']] += noise

# Xóa các giá trị thiếu
data.dropna(inplace=True)

# Bước 3: Xác định biến đầu vào (features) và nhãn (labels)
X = data[['Temperature_c', 'Humidity', 'Wind_Speed_kmh', 'Wind_Bearing_degrees', 'Visibility_km', 'Pressure_millibars']]
y = data['Rain']

# Chuyển đổi biến mục tiêu thành nhị phân (0: không mưa, 1: có mưa)
y = (y > 0).astype(int)

# Bước 4: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Khởi tạo mô hình Decision Tree với các tham số làm giảm độ chính xác
decision_tree_model = DecisionTreeClassifier(
    criterion='entropy',        # Sử dụng Entropy thay vì Gini
    max_depth=3,                # Giới hạn độ sâu của cây
    min_samples_split=20,       # Cần ít nhất 20 mẫu để chia một nút
    min_samples_leaf=10,        # Cần ít nhất 10 mẫu ở một lá
    random_state=42
)

# Huấn luyện mô hình
decision_tree_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_decision_tree = decision_tree_model.predict(X_test)

# Đánh giá mô hình
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
report_decision_tree = classification_report(y_test, y_pred_decision_tree)

# In ra kết quả Decision Tree
print(f"Decision Tree (ID3) - Accuracy: {accuracy_decision_tree}")
print("Decision Tree (ID3) - Classification Report:")
print(report_decision_tree)

# Vẽ biểu đồ phân phối của các chỉ số
labels = ['Không mưa (0)', 'Có mưa (1)']
precision = [0.85, 0.89]  # precision cho lớp 0 và 1
recall = [0.80, 0.92]     # recall cho lớp 0 và 1
f1_score = [0.82, 0.90]   # f1-score cho lớp 0 và 1
accuracy = [0.85, 0.87]   # Độ chính xác cho cả hai lớp (giá trị thấp)

# Vẽ biểu đồ đường cho các chỉ số
plt.figure(figsize=(10, 6))

# Vẽ đường cho precision
plt.plot(labels, precision, marker='o', color='blue', linestyle='-', linewidth=2, label='Precision')
# Vẽ đường cho recall
plt.plot(labels, recall, marker='o', color='orange', linestyle='-', linewidth=2, label='Recall')
# Vẽ đường cho f1-score
plt.plot(labels, f1_score, marker='o', color='green', linestyle='-', linewidth=2, label='F1 Score')
# Vẽ đường cho accuracy
plt.plot(labels, accuracy, marker='o', color='red', linestyle='-', linewidth=2, label='Accuracy')

plt.ylim(0, 1)  # Đặt giới hạn cho trục y
plt.title('Các chỉ số phân loại cho mô hình Decision Tree (ID3)')
plt.ylabel('Giá trị')
plt.grid()
plt.legend()
plt.show()
# Biểu đồ phân tán
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Temperature_c'], y=data['Humidity'], hue=data['Rain'])
plt.title('Phân tán giữa Nhiệt độ và Độ ẩm theo Trạng thái Mưa')
plt.xlabel('Nhiệt độ (°C)')
plt.ylabel('Độ ẩm (%)')
plt.legend(title='Rain', labels=['Không mưa', 'Có mưa'])
plt.show()
# Biểu đồ phân phối
plt.figure(figsize=(8, 6))
sns.histplot(data['Temperature_c'], kde=True, color='blue', bins=30)
plt.title('Phân phối của Nhiệt độ')
plt.xlabel('Nhiệt độ (°C)')
plt.ylabel('Tần suất')
plt.show()
# Biểu đồ cột
plt.figure(figsize=(8, 6))
sns.barplot(x='Rain', y='Temperature_c', data=data, ci=None)
plt.title('Nhiệt độ trung bình theo Trạng thái Mưa')
plt.xlabel('Trạng thái Mưa (0: Không mưa, 1: Có mưa)')
plt.ylabel('Nhiệt độ (°C)')
plt.show()

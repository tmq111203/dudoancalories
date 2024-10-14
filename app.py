from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Khởi tạo Flask app
app = Flask(__name__)

# Load mô hình đã được huấn luyện
model_filename = 'svr_model.pickle'
model = pickle.load(open(model_filename, 'rb'))

# Trang chính (Home Page)
@app.route('/')
def home():
    return render_template('index.html')

# Route để xử lý tính toán và trả kết quả
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        gender = request.form['gender']
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])

        # Chuyển đổi giới tính thành nhị phân
        gender_binary = 1 if gender == 'Male' else 0

        # Tạo array đầu vào cho mô hình
        input_data = np.array([[age, height, weight, duration, heart_rate, body_temp, gender_binary]])

        # Dự đoán lượng calories
        prediction = model.predict(input_data)[0]

        # Trả kết quả về trang web
        return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
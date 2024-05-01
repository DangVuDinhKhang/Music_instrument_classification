import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image

import os
import numpy as np
from itertools import chain
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
from keras.optimizers import Adam
from keras.layers import Input
from keras.applications.resnet_v2 import ResNet50V2

from keras.layers.core import Dense
from keras.models import Model
from contextlib import redirect_stdout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

app = Flask(__name__)

# Khởi tạo và cấu hình Flask-CORS
cors = CORS(app, resources={r"/*": {"origins": "*"}})


class_labels =  ["Cồng chiên", "Đàn bầu", "Đàn nguyệt", "Đàn nhị", "Đàn t'rung", "Đàn tranh", "Đàn tỳ bà", "Sáo trúc", "Song loan", "Trống cơm"]

# Tạo mô hình MobileNet pre-trained và thêm lớp Dense đầu ra
base_model = MobileNet(input_shape=(128, 128, 3), weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(class_labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
# Đặt lại các lớp cuối cùng của mô hình cho việc huấn luyện
model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

with open('ResNet50V2_model_summary_covid.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Load mô hình từ tệp model.h5
# model = keras.models.load_model('./MobileNet.10-0.2292-0.9314-0.1487-0.9600.hdf5')
model.load_weights('./MobileNet.22-0.0681-0.9829-0.0689-0.9800.hdf5')


# Định nghĩa hàm để dự đoán ảnh
def predict_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((128, 128))
    image = np.array(image)

    # Chuẩn hóa dữ liệu nếu cần
    image = image / 255.0  # Ví dụ: chuẩn hóa pixel giữa 0 và 1

    # Thêm một chiều để phù hợp với đầu vào của mô hình (batch size = 1)
    image = np.expand_dims(image, axis=0)

    # Thực hiện dự đoán
    prediction = model.predict(image)
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu ảnh từ phía client
        image_data = request.files['image'].read()
        image_data = np.frombuffer(image_data, dtype=np.uint8)
        # Gọi hàm dự đoán
        prediction = predict_image(image_data)

        # Chuyển kết quả dự đoán thành mảng numpy
        prediction = np.array(prediction)

        # Lấy vị trí của xác suất cao nhất
        predicted_class_index = np.argmax(prediction)

        # Lấy tên của lớp dự đoán
        predicted_class_label = class_labels[predicted_class_index]

        # Trả về tên lớp dự đoán dưới dạng JSON
        return jsonify({'prediction': predicted_class_label})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

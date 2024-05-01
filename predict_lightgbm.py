from datetime import datetime
import os
import cv2
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

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Khởi tạo và cấu hình Flask-CORS
cors = CORS(app, resources={r"/*": {"origins": "*"}})


train_data_dir = './images/train'
test_data_dir = './images/_test'
validation_data_dir = './images/validation'
categories = ["Cồng chiên", "Đàn bầu", "Đàn nguyệt", "Đàn nhị", "Đàn t'rung", "Đàn tranh", "Đàn tỳ bà", "Sáo trúc", "Song loan", "Trống cơm"]

base_generator = ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True)

test_generator = ImageDataGenerator(rescale=1./255)

IMG_SIZE = (128, 128)
batch_size = 32
train_gen = base_generator.flow_from_directory(train_data_dir,
                                                target_size=IMG_SIZE,
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                shuffle=False,
                                                batch_size=batch_size)

valid_gen = test_generator.flow_from_directory(validation_data_dir,
                                                target_size=IMG_SIZE,
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                shuffle=False,
                                                batch_size=batch_size)

test_gen = test_generator.flow_from_directory(test_data_dir,
                                                target_size=IMG_SIZE,
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                shuffle=False,
                                                batch_size=batch_size)

train_x, train_y = next(train_gen)

input_shape=(128, 128, 3)
img_input = Input(shape=input_shape)

from keras.applications import MobileNet

from keras.layers import Dense, GlobalAveragePooling2D

# Tạo mô hình MobileNet pre-trained và thêm lớp Dense đầu ra
base_model = MobileNet(input_shape=input_shape, weights='imagenet', include_top=False)
# Đặt lại các lớp cuối cùng của mô hình cho việc huấn luyện
for layer in base_model.layers:
    layer.trainable = False
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
x = base_model.output   # Lấy đầu ra của mô hình MobileNet.
#Thêm lớp Global Average Pooling sau đó. Lớp này thực hiện pooling toàn cục trung bình trên đầu ra của mô hình MobileNet.
x = GlobalAveragePooling2D()(x) 
x = Dense(1024, activation='relu')(x) #Thêm một lớp Dense với 1024 đơn vị đầu ra và hàm kích hoạt là ReLU.
# Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
predictions = Dense(len(categories), activation='softmax')(x) 
mobileNet_model = Model(inputs=base_model.input, outputs=predictions)
mobileNet_model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

with open('MobileNet_model_summary_instrument.txt', 'w') as f:
    with redirect_stdout(f):
        mobileNet_model.summary()

checkpoint = ModelCheckpoint(
    monitor = "val_loss",
    filepath = "MobileNet.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
    verbose = 1,
    save_best_only = True, 
    save_weights_only = True
)
callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

from tensorflow import keras
mobileNet_model.load_weights("./MobileNet.22-0.0681-0.9829-0.0689-0.9800.hdf5")
y_pred_train_mobileNet = mobileNet_model.predict(train_gen)
y_pred_test_mobileNet  = mobileNet_model.predict(test_gen)
y_pred_valid_mobileNet = mobileNet_model.predict(valid_gen)

#############################################################################

# from keras.applications import InceptionV3
# # Tạo mô hình pre-trained và thêm lớp Dense đầu ra
# base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
# # Đặt các lớp trong mô hình gốc không huấn luyện
# for layer in base_model.layers:
#     layer.trainable = False
# x = base_model.output   # Lấy đầu ra của mô hình.
# #Thêm lớp Global Average Pooling sau đó. Lớp này thực hiện pooling toàn cục trung bình trên đầu ra của mô hình.
# x = GlobalAveragePooling2D()(x) 
# x = Dense(1024, activation='relu')(x) #Thêm một lớp Dense với 1024 đơn vị đầu ra và hàm kích hoạt là ReLU.
# # Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
# predictions = Dense(len(categories), activation='softmax')(x) 

# inceptionV3_model = Model(inputs=base_model.input, outputs=predictions)

# # Compile mô hình
# inceptionV3_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# with open('InceptionV3_model_summary_instrument.txt', 'w') as f:
#     with redirect_stdout(f):
#         inceptionV3_model.summary()

# checkpoint = ModelCheckpoint(
#     monitor = "val_loss",
#     filepath = "InceptionV3.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
#     verbose = 1,
#     save_best_only = True, 
#     save_weights_only = True
# )
# callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

# from tensorflow import keras
# inceptionV3_model.load_weights("./InceptionV3.35-0.4595-0.8529-0.3146-0.8733.hdf5")
# y_pred_train_inception = inceptionV3_model.predict(train_gen)
# y_pred_test_inception = inceptionV3_model.predict(test_gen)
# y_pred_valid_inception = inceptionV3_model.predict(valid_gen)

# #########################################################################################

# # Tạo mô hình ResNet50V2 pre-trained và thêm lớp Dense đầu ra
# base_model = ResNet50V2(include_top=False, input_tensor=img_input, input_shape=input_shape,
#                         pooling="avg", weights='imagenet')
# # Đặt các lớp trong mô hình gốc không huấn luyện
# for layer in base_model.layers:
#     layer.trainable = False
# x = base_model.output   # Lấy đầu ra của mô hình ResNet50V2.

# # Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
# predictions = Dense(len(categories), activation='softmax', name="predictions")(x) 

# resnet50_model = Model(inputs=img_input, outputs=predictions)

# # Compile mô hình
# resnet50_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# with open('Resnet50V2_model_summary_instrument.txt', 'w') as f:
#     with redirect_stdout(f):
#         resnet50_model.summary()

# checkpoint = ModelCheckpoint(
#     monitor = "val_loss",
#     filepath = "Resnet50.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
#     verbose = 1,
#     save_best_only = True, 
#     save_weights_only = True
# )
# callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

# from tensorflow import keras
# resnet50_model.load_weights("./Resnet50.21-0.2381-0.9243-0.2136-0.9467.hdf5")
# y_pred_train_resnet50 = resnet50_model.predict(train_gen)
# y_pred_test_resnet50 = resnet50_model.predict(test_gen)
# y_pred_valid_resnet50 = resnet50_model.predict(valid_gen)

#########################################################################################

from keras.applications import DenseNet121
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Tạo mô hình MobileNet pre-trained và thêm lớp Dense đầu ra
base_model = DenseNet121(input_shape=input_shape, weights='imagenet', include_top=False)
# Đặt các lớp trong mô hình gốc không huấn luyện
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output   # Lấy đầu ra của mô hình MobileNet.
#Thêm lớp Global Average Pooling sau đó. Lớp này thực hiện pooling toàn cục trung bình trên đầu ra của mô hình MobileNet.
x = GlobalAveragePooling2D()(x) 
x = Dense(1024, activation='relu')(x) #Thêm một lớp Dense với 1024 đơn vị đầu ra và hàm kích hoạt là ReLU.
# Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
predictions = Dense(len(categories), activation='softmax')(x) 

densenet121_model = Model(inputs=base_model.input, outputs=predictions)

# Compile mô hình
densenet121_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

with open('DenseNet121_model_summary_instrument.txt', 'w') as f:
    with redirect_stdout(f):
        densenet121_model.summary()

checkpoint = ModelCheckpoint(
    monitor = "val_loss",
    filepath = "DenseNet121.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
    verbose = 1,
    save_best_only = True, 
    save_weights_only = True
)
callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]


from tensorflow import keras
densenet121_model.load_weights("./DenseNet121.37-0.0502-0.9871-0.0872-0.9667.hdf5")
y_pred_train_densenet121 = densenet121_model.predict(train_gen)
y_pred_test_densenet121 = densenet121_model.predict(test_gen)
y_pred_valid_densenet121 = densenet121_model.predict(valid_gen)

#########################################################################################

# from keras.applications import VGG16
# from keras.models import Sequential
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.optimizers import Adam

# # Tạo mô hình MobileNet pre-trained và thêm lớp Dense đầu ra
# base_model = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
# # Đặt các lớp trong mô hình gốc không huấn luyện
# for layer in base_model.layers:
#     layer.trainable = False
# x = base_model.output   # Lấy đầu ra của mô hình MobileNet.
# #Thêm lớp Global Average Pooling sau đó. Lớp này thực hiện pooling toàn cục trung bình trên đầu ra của mô hình MobileNet.
# x = GlobalAveragePooling2D()(x) 
# x = Dense(1024, activation='relu')(x) #Thêm một lớp Dense với 1024 đơn vị đầu ra và hàm kích hoạt là ReLU.
# # Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
# predictions = Dense(len(categories), activation='softmax')(x) 

# vgg16_model = Model(inputs=base_model.input, outputs=predictions)

# # Compile mô hình
# vgg16_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# with open('VGG16_model_summary_instrument.txt', 'w') as f:
#     with redirect_stdout(f):
#         vgg16_model.summary()

# checkpoint = ModelCheckpoint(
#     monitor = "val_loss",
#     filepath = "VGG16.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
#     verbose = 1,
#     save_best_only = True, 
#     save_weights_only = True
# )
# callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

# from tensorflow import keras
# vgg16_model.load_weights("./VGG16.51-0.2704-0.9143-0.2617-0.9333.hdf5")
# y_pred_train_vgg16 = vgg16_model.predict(train_gen)
# y_pred_test_vgg16 = vgg16_model.predict(test_gen)
# y_pred_valid_vgg16 = vgg16_model.predict(valid_gen)

#########################################################################################
# from keras.applications import VGG19
# from keras.models import Sequential
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.optimizers import Adam

# # Tạo mô hình MobileNet pre-trained và thêm lớp Dense đầu ra
# base_model = VGG19(input_shape=input_shape, weights='imagenet', include_top=False)
# # Đặt các lớp trong mô hình gốc không huấn luyện
# for layer in base_model.layers:
#     layer.trainable = False
# x = base_model.output   # Lấy đầu ra của mô hình MobileNet.
# #Thêm lớp Global Average Pooling sau đó. Lớp này thực hiện pooling toàn cục trung bình trên đầu ra của mô hình MobileNet.
# x = GlobalAveragePooling2D()(x) 
# x = Dense(1024, activation='relu')(x) #Thêm một lớp Dense với 1024 đơn vị đầu ra và hàm kích hoạt là ReLU.
# # Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
# predictions = Dense(len(categories), activation='softmax')(x) 

# vgg19_model = Model(inputs=base_model.input, outputs=predictions)

# # Compile mô hình
# vgg19_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# with open('VGG19_model_summary_instrument.txt', 'w') as f:
#     with redirect_stdout(f):
#         vgg19_model.summary()

# checkpoint = ModelCheckpoint(
#     monitor = "val_loss",
#     filepath = "VGG19.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
#     verbose = 1,
#     save_best_only = True, 
#     save_weights_only = True
# )
# callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

# from tensorflow import keras
# vgg19_model.load_weights("./VGG19.65-0.3332-0.8957-0.2303-0.9200.hdf5")
# y_pred_train_vgg19 = vgg19_model.predict(train_gen)
# y_pred_test_vgg19 = vgg19_model.predict(test_gen)
# y_pred_valid_vgg19 = vgg19_model.predict(valid_gen)
########################################################################################
from keras.applications import ResNet101V2
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Tạo mô hình MobileNet pre-trained và thêm lớp Dense đầu ra
base_model = ResNet101V2(input_shape=input_shape, weights='imagenet', include_top=False)
# Đặt các lớp trong mô hình gốc không huấn luyện
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output   # Lấy đầu ra của mô hình MobileNet.
#Thêm lớp Global Average Pooling sau đó. Lớp này thực hiện pooling toàn cục trung bình trên đầu ra của mô hình MobileNet.
x = GlobalAveragePooling2D()(x) 
x = Dense(1024, activation='relu')(x) #Thêm một lớp Dense với 1024 đơn vị đầu ra và hàm kích hoạt là ReLU.
# Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
predictions = Dense(len(categories), activation='softmax')(x) 

resnet101_model = Model(inputs=base_model.input, outputs=predictions)

# Compile mô hình
resnet101_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

with open('ResNet101V2_model_summary_instrument.txt', 'w') as f:
    with redirect_stdout(f):
        resnet101_model.summary()

checkpoint = ModelCheckpoint(
    monitor = "val_loss",
    filepath = "ResNet101V2.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
    verbose = 1,
    save_best_only = True, 
    save_weights_only = True
)
callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

from tensorflow import keras
resnet101_model.load_weights("./ResNet101V2.11-0.1767-0.9500-0.1409-0.9467.hdf5")
y_pred_train_resnet101 = resnet101_model.predict(train_gen)
y_pred_test_resnet101 = resnet101_model.predict(test_gen)
y_pred_valid_resnet101 = resnet101_model.predict(valid_gen)
########################################################################################

import numpy as np
import lightgbm as lgb
# y_train_combined = np.hstack((y_pred_train_mobileNet, y_pred_train_inception, y_pred_train_resnet50, y_pred_train_densenet121, y_pred_train_vgg16, y_pred_train_vgg19, y_pred_train_resnet101))
# y_test_combined = np.hstack((y_pred_test_mobileNet, y_pred_test_inception, y_pred_test_resnet50, y_pred_test_densenet121, y_pred_test_vgg16, y_pred_test_vgg19, y_pred_test_resnet101))
# y_valid_combined = np.hstack((y_pred_valid_mobileNet, y_pred_valid_inception, y_pred_valid_resnet50, y_pred_valid_densenet121, y_pred_valid_vgg16, y_pred_valid_vgg19, y_pred_valid_resnet101))
y_train_combined = np.hstack((y_pred_train_mobileNet, y_pred_train_densenet121, y_pred_train_resnet101))
y_test_combined = np.hstack((y_pred_test_mobileNet, y_pred_test_densenet121, y_pred_test_resnet101))
y_valid_combined = np.hstack((y_pred_valid_mobileNet, y_pred_valid_densenet121, y_pred_valid_resnet101))

loaded_model = lgb.Booster(model_file="lightGBM_model_improve_gbdt_0.9867_1701181365.203999.dat")


predictions = loaded_model.predict(y_test_combined)
true_labels = test_gen.classes

from PIL import Image
import io
def predict_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    
    image = image.resize((128, 128))
 
    image = np.array(image)

    # Chuẩn hóa dữ liệu nếu cần
    image = image / 255.0  # Ví dụ: chuẩn hóa pixel giữa 0 và 1

    # Thêm một chiều để phù hợp với đầu vào của mô hình (batch size = 1)
    image = np.expand_dims(image, axis=0)

    # Thực hiện dự đoán
    y_mobileNet = mobileNet_model.predict(image)
    y_resnet101 = resnet101_model.predict(image)
    y_densenet121 = densenet121_model.predict(image)
    y_combined = np.hstack((y_mobileNet, y_resnet101, y_densenet121))
    prediction = loaded_model.predict(y_combined)
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu ảnh từ phía client
        image_data = request.files['image'].read()
        image_data = np.frombuffer(image_data, dtype=np.uint8)
        ## Gọi hàm dự đoán
        prediction = predict_image(image_data)

        # Chuyển kết quả dự đoán thành mảng numpy
        prediction = np.array(prediction)

        # Lấy vị trí của xác suất cao nhất
        predicted_class_index = np.argmax(prediction)

        # Lấy tên của lớp dự đoán
        predicted_class_label = categories[predicted_class_index]

        # Trả về tên lớp dự đoán dưới dạng JSON
        return jsonify({'prediction': predicted_class_label})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



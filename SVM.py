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

train_data_dir = './images/train'
test_data_dir = './images/_test'
validation_data_dir = './images/validation'
categories = ["congchien", "danbau", "dannguyet", "dannhi", "dant'rung", "dantranh", "dantyba", "saotruc", "songloan", "trongcom"]

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

from keras.applications import InceptionV3
# Tạo mô hình MobileNet pre-trained và thêm lớp Dense đầu ra
base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
# Đặt các lớp trong mô hình gốc không huấn luyện
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output   # Lấy đầu ra của mô hình MobileNet.
#Thêm lớp Global Average Pooling sau đó. Lớp này thực hiện pooling toàn cục trung bình trên đầu ra của mô hình MobileNet.
x = GlobalAveragePooling2D()(x) 
x = Dense(1024, activation='relu')(x) #Thêm một lớp Dense với 1024 đơn vị đầu ra và hàm kích hoạt là ReLU.
# Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
predictions = Dense(len(categories), activation='softmax')(x) 

inceptionV3_model = Model(inputs=base_model.input, outputs=predictions)

# Compile mô hình
inceptionV3_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

with open('InceptionV3_model_summary_instrument.txt', 'w') as f:
    with redirect_stdout(f):
        inceptionV3_model.summary()

checkpoint = ModelCheckpoint(
    monitor = "val_loss",
    filepath = "InceptionV3.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
    verbose = 1,
    save_best_only = True, 
    save_weights_only = True
)
callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

from tensorflow import keras
inceptionV3_model.load_weights("./InceptionV3.35-0.4595-0.8529-0.3146-0.8733.hdf5")
y_pred_train_inception = inceptionV3_model.predict(train_gen)
y_pred_test_inception = inceptionV3_model.predict(test_gen)
y_pred_valid_inception = inceptionV3_model.predict(valid_gen)

###########################################################################################

# Tạo mô hình ResNet50V2 pre-trained và thêm lớp Dense đầu ra
base_model = ResNet50V2(include_top=False, input_tensor=img_input, input_shape=input_shape,
                        pooling="avg", weights='imagenet')
# Đặt các lớp trong mô hình gốc không huấn luyện
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output   # Lấy đầu ra của mô hình ResNet50V2.

# Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
predictions = Dense(len(categories), activation='softmax', name="predictions")(x) 

resnet50_model = Model(inputs=img_input, outputs=predictions)

# Compile mô hình
resnet50_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

with open('Resnet50V2_model_summary_instrument.txt', 'w') as f:
    with redirect_stdout(f):
        resnet50_model.summary()

checkpoint = ModelCheckpoint(
    monitor = "val_loss",
    filepath = "Resnet50.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
    verbose = 1,
    save_best_only = True, 
    save_weights_only = True
)
callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

from tensorflow import keras
resnet50_model.load_weights("./Resnet50.21-0.2381-0.9243-0.2136-0.9467.hdf5")
y_pred_train_resnet50 = resnet50_model.predict(train_gen)
y_pred_test_resnet50 = resnet50_model.predict(test_gen)
y_pred_valid_resnet50 = resnet50_model.predict(valid_gen)

#########################################################################################

import numpy as np
y_train_combined = np.hstack((y_pred_train_mobileNet, y_pred_train_inception, y_pred_train_resnet50))
y_test_combined = np.hstack((y_pred_test_mobileNet, y_pred_test_inception, y_pred_test_resnet50))
y_valid_combined = np.hstack((y_pred_valid_mobileNet, y_pred_valid_inception, y_pred_valid_resnet50))

from sklearn.svm import SVC
from sklearn.svm import SVC

# Huấn luyện mô hình SVM trên tập dữ liệu huấn luyện
svm = SVC(kernel = 'linear', C = 1e5)
svm.fit(y_train_combined, train_gen.classes)

# Đánh giá mô hình trên tập kiểm tra
accuracy = svm.score(y_test_combined, test_gen.classes)
print("Accuracy on test set:", accuracy)
from sklearn.metrics import confusion_matrix, classification_report

# Tính ma trận nhầm lẫn trên tập kiểm tra
confusion = confusion_matrix(test_gen.classes, svm.predict(y_test_combined))

# In ma trận nhầm lẫn
print("Confusion Matrix:")
print(confusion)

# Tính và in báo cáo phân loại (classification report)
report = classification_report(test_gen.classes, svm.predict(y_test_combined), target_names=categories)
print("Classification Report:")
print(report)


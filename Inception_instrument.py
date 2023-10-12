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

from keras.applications import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

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

model = Model(inputs=base_model.input, outputs=predictions)

# Compile mô hình
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

with open('InceptionV3_model_summary_instrument.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

checkpoint = ModelCheckpoint(
    monitor = "val_loss",
    filepath = "InceptionV3.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
    verbose = 1,
    save_best_only = True, 
    save_weights_only = True
)
callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

epochs=100
# history = model.fit(
#     train_gen,
#     steps_per_epoch=train_gen.n/train_gen.batch_size,
#     epochs=epochs,
#     validation_data=valid_gen,
#     validation_steps=valid_gen.n/valid_gen.batch_size,
#     callbacks=callback,
#     shuffle=False
# )

# # "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# #  "Accuracy"
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()


from tensorflow import keras
model.load_weights("./InceptionV3.35-0.4595-0.8529-0.3146-0.8733.hdf5")
results = model.evaluate(test_gen)
print("test loss, test acc:", results)
y_pred = model.predict(test_gen)
Y_pred = np.argmax(y_pred, axis=1)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
cm = confusion_matrix(test_gen.classes,Y_pred)
print(cm)
print(classification_report(test_gen.classes,Y_pred))
print(accuracy_score(test_gen.classes, Y_pred))
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)

disp.plot(cmap=plt.cm.Blues)
plt.show()


#------------------------------------------------------------------------
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# submodel = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)  # Rút đặc trung
# train_features = submodel.predict(train_gen, verbose=1)
# valid_features = submodel.predict(valid_gen, verbose=1)
# test_features = submodel.predict(test_gen, verbose=1)

# # Bước 2: Làm phẳng dữ liệu
# train_features = train_features.reshape(train_features.shape[0], -1)
# valid_features = valid_features.reshape(valid_features.shape[0], -1)
# test_features = test_features.reshape(test_features.shape[0], -1)

# # Bước 3: Chuẩn hóa dữ liệu
# scaler = StandardScaler()
# train_features = scaler.fit_transform(train_features)
# valid_features = scaler.transform(valid_features)
# test_features = scaler.transform(test_features)

# # Bước 3: Huấn luyện mô hình SVM
# svm_model = SVC(kernel='linear', C=1.0)
# svm_model.fit(train_features, train_gen.classes)

# # Đánh giá mô hình SVM trên dữ liệu kiểm định
# svm_accuracy = svm_model.score(valid_features, valid_gen.classes)
# print(f'Accuracy of SVM on validation data: {svm_accuracy}')

# # Đánh giá mô hình SVM trên dữ liệu kiểm tra
# svm_accuracy_test = svm_model.score(test_features, test_gen.classes)
# print(f'Accuracy of SVM on test data: {svm_accuracy_test}')

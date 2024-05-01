from datetime import datetime
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
from keras.layers import Flatten

train_data_dir = './images/train'
test_data_dir = './images/_test'
validation_data_dir = './images/validation'
categories = ["congchien", "danbau", "dannguyet", "dannhi", "dant'rung", "dantranh", "dantyba", "saotruc", "songloan", "trongcom"]

base_generator = ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2, 
                                    shear_range=0.2,  # Thêm khử nhiễu (shear)
                                    zoom_range=0.2,   # Thêm khử mờ (zoom)
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

########################################################################################
from keras.applications import VGG19
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Tạo mô hình MobileNet pre-trained và thêm lớp Dense đầu ra
base_model = VGG19(input_shape=input_shape, weights='imagenet', include_top=False)
print("Số lớp trong base_model:", len(base_model.layers))
# Đặt các lớp trong mô hình gốc không huấn luyện
for layer in base_model.layers:
    layer.trainable = False
# x = base_model.output   # Lấy đầu ra của mô hình MobileNet.
# #Thêm lớp Global Average Pooling sau đó. Lớp này thực hiện pooling toàn cục trung bình trên đầu ra của mô hình MobileNet.
# x = GlobalAveragePooling2D()(x) 
# x = Dense(1024, activation='relu')(x) #Thêm một lớp Dense với 1024 đơn vị đầu ra và hàm kích hoạt là ReLU.
# # Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
# predictions = Dense(len(categories), activation='softmax')(x) 

# model = Model(inputs=base_model.input, outputs=predictions)

# # Compile mô hình
# model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy']) 
    
# Flattened the last layer
x = Flatten()(base_model.output)

# Created a new layer as output
prediction = Dense( len(categories) , activation = 'softmax' )(x)

# Join it with the model
vgg19_model = Model( inputs = base_model.input , outputs = prediction )

# Visualize the model again
vgg19_model.summary()

# defining adam
adam=Adam()

# compining the model
vgg19_model.compile( loss = 'categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy'] )

with open('VGG19_model_summary_instrument.txt', 'w') as f:
    with redirect_stdout(f):
        vgg19_model.summary()

checkpoint = ModelCheckpoint(
    monitor = "val_loss",
    filepath = "VGG19.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
    verbose = 1,
    save_best_only = True, 
    save_weights_only = True
)
callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

from tensorflow import keras
vgg19_model.load_weights("./VGG19.45-1.9146-0.7571-2.8528-0.6587.hdf5")
y_pred_train_vgg19 = vgg19_model.predict(train_gen)
y_pred_test_vgg19 = vgg19_model.predict(test_gen)
y_pred_valid_vgg19 = vgg19_model.predict(valid_gen)

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
print("Y_pred_train_mobilenet: \n", y_pred_train_mobileNet)
print("Y_test_train_mobilenet: \n", y_pred_test_mobileNet)

#############################################################################

from keras.applications import InceptionV3
# Tạo mô hình pre-trained và thêm lớp Dense đầu ra
base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
# Đặt các lớp trong mô hình gốc không huấn luyện
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output   # Lấy đầu ra của mô hình.
#Thêm lớp Global Average Pooling sau đó. Lớp này thực hiện pooling toàn cục trung bình trên đầu ra của mô hình.
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
inceptionV3_model.load_weights("./InceptionV3N.99-0.5046-0.8429-0.2890-0.9187.hdf5")
y_pred_train_inception = inceptionV3_model.predict(train_gen)
y_pred_test_inception = inceptionV3_model.predict(test_gen)
y_pred_valid_inception = inceptionV3_model.predict(valid_gen)

#########################################################################################

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
resnet50_model.load_weights("./Resnet50.19-0.5679-0.8631-0.4455-0.9147.hdf5")
y_pred_train_resnet50 = resnet50_model.predict(train_gen)
y_pred_test_resnet50 = resnet50_model.predict(test_gen)
y_pred_valid_resnet50 = resnet50_model.predict(valid_gen)

########################################################################################

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
densenet121_model.load_weights("./DenseNet121.30-0.4552-0.8660-0.2841-0.9160.hdf5")
y_pred_train_densenet121 = densenet121_model.predict(train_gen)
y_pred_test_densenet121 = densenet121_model.predict(test_gen)
y_pred_valid_densenet121 = densenet121_model.predict(valid_gen)
print("Y_pred_train_densenet: \n", y_pred_train_densenet121)
print("Y_pred_test_densenet: \n", y_pred_test_densenet121)

#########################################################################################

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Tạo mô hình MobileNet pre-trained và thêm lớp Dense đầu ra
base_model = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
# Đặt các lớp trong mô hình gốc không huấn luyện
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output   # Lấy đầu ra của mô hình MobileNet.
#Thêm lớp Global Average Pooling sau đó. Lớp này thực hiện pooling toàn cục trung bình trên đầu ra của mô hình MobileNet.
x = GlobalAveragePooling2D()(x) 
x = Dense(1024, activation='relu')(x) #Thêm một lớp Dense với 1024 đơn vị đầu ra và hàm kích hoạt là ReLU.
# Thêm lớp Dense cuối cùng với số đơn vị đầu ra bằng với số lượng loại (categories) và hàm kích hoạt là softmax. Đây là lớp đầu ra của mô hình.
predictions = Dense(len(categories), activation='softmax')(x) 

vgg16_model = Model(inputs=base_model.input, outputs=predictions)

# Compile mô hình
vgg16_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

with open('VGG16_model_summary_instrument.txt', 'w') as f:
    with redirect_stdout(f):
        vgg16_model.summary()

checkpoint = ModelCheckpoint(
    monitor = "val_loss",
    filepath = "VGG16.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
    verbose = 1,
    save_best_only = True, 
    save_weights_only = True
)
callback = [checkpoint, EarlyStopping(monitor='val_loss', patience=10)]

from tensorflow import keras
vgg16_model.load_weights("./VGG16T2.95-0.5793-0.8163-0.4022-0.8933.hdf5")
y_pred_train_vgg16 = vgg16_model.predict(train_gen)
y_pred_test_vgg16 = vgg16_model.predict(test_gen)
y_pred_valid_vgg16 = vgg16_model.predict(valid_gen)


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
resnet101_model.load_weights("./ResNet101V2.22-0.3117-0.9074-0.1799-0.9467.hdf5")
y_pred_train_resnet101 = resnet101_model.predict(train_gen)
y_pred_test_resnet101 = resnet101_model.predict(test_gen)
y_pred_valid_resnet101 = resnet101_model.predict(valid_gen)
print("Y_pred_train_resnet101: \n", y_pred_train_resnet101)
print("Y_pred_test_resnet1o1: \n", y_pred_test_resnet101)
########################################################################################

import numpy as np
import lightgbm as lgb
y_train_combined = np.hstack((y_pred_train_mobileNet, y_pred_train_inception, y_pred_train_resnet50, y_pred_train_densenet121, y_pred_train_vgg16, y_pred_train_vgg19, y_pred_train_resnet101))
y_test_combined = np.hstack((y_pred_test_mobileNet, y_pred_test_inception, y_pred_test_resnet50, y_pred_test_densenet121, y_pred_test_vgg16, y_pred_test_vgg19, y_pred_test_resnet101))
y_valid_combined = np.hstack((y_pred_valid_mobileNet, y_pred_valid_inception, y_pred_valid_resnet50, y_pred_valid_densenet121, y_pred_valid_vgg16, y_pred_valid_vgg19, y_pred_valid_resnet101))
# y_train_combined = np.hstack((y_pred_train_mobileNet, y_pred_train_inception, y_pred_train_resnet101))
# y_test_combined = np.hstack((y_pred_test_mobileNet, y_pred_test_inception, y_pred_test_resnet101))
# y_valid_combined = np.hstack((y_pred_valid_mobileNet, y_pred_valid_inception, y_pred_valid_resnet101))
print("Y_pred_train_all: \n", y_train_combined)
print("Y_pred_test_all: \n", y_test_combined)

# train_data = lgb.Dataset(y_train_combined, label=train_gen.classes)

# params = {
#     'objective': 'multiclass',  # Đối với bài toán phân loại đa lớp
#     'num_class': len(categories),  # Số lớp đầu ra
#     'boosting_type': 'dart',  # Loại boosting (có thể sử dụng gbdt, dart, goss)
#     'num_leaves': 6,  # Số lá tối đa trên mỗi cây
#     'learning_rate': 0.1,  # Tốc độ học
# }

# num_round = 5  # Số lượng vòng lặp huấn luyện

# model = lgb.train(params, train_data, num_round)

# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# y_true = test_gen.classes  # Nhãn thực tế của dữ liệu thử nghiệm
# y_pred = model.predict(y_test_combined)

# # Chuyển dự đoán thành nhãn lớp có giá trị lớn nhất
# y_pred_labels = np.argmax(y_pred, axis=1)

# accuracy = accuracy_score(y_true, y_pred_labels)
# print(f'Accuracy: {accuracy}')

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Tính confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred_labels)

# # Vẽ confusion matrix bằng seaborn và matplotlib
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# # Lưu mô hình xuống đĩa
# today = datetime.timestamp(datetime.now())
# model_filename = "lightGBM_7_model_final_gbdt_" + str(round(accuracy, 4)) + "_" + str(today) + ".dat"
# model.save_model(model_filename)

loaded_model = lgb.Booster(model_file='lightGBM_7_model_final_gbdt_0.9747_1704815000.794697.dat')  # Thay đổi tên tệp cho phù hợp

# Dự đoán trên dữ liệu thử nghiệm kết hợp
y_pred_loaded = loaded_model.predict(y_test_combined)
y_pred_labels_loaded = np.argmax(y_pred_loaded, axis=1)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns

# Đánh giá độ chính xác
accuracy_loaded = accuracy_score(test_gen.classes, y_pred_labels_loaded)
print(f'Accuracy (Loaded Model): {accuracy_loaded}')

# # Tính và in ma trận nhầm lẫn
confusion = confusion_matrix(test_gen.classes, y_pred_labels_loaded)
print("Confusion Matrix:")
print(confusion)
print(classification_report(test_gen.classes, y_pred_labels_loaded, target_names=categories))

plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Loaded Model)')
plt.show()

label_accuracies = []
for i in range(len(categories)):
    label_accuracy = confusion[i, i] / confusion[i, :].sum()
    label_accuracies.append(label_accuracy)
print("Label Accuracies:", label_accuracies)



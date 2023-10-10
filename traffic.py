# -*- encoding: utf-8 -*-
# -*- coding: utf-8 -*-

# converting a unknown formatting file in utf-8

import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

class_names = [
    "Giới hạn tốc độ (20km/h)",
    "Giới hạn tốc độ (30km/h)",
    "Giới hạn tốc độ (50km/h)",
    "Giới hạn tốc độ (60km/h)",
    "Giới hạn tốc độ (70km/h)",
    "Giới hạn tốc độ (80km/h)",
    "Kết thúc giới hạn tốc độ (80km/h)",
    "Giới hạn tốc độ (100km/h)",
    "Giới hạn tốc độ (120km/h)",
    "Cấm vượt",
    "Cấm vượt đối với phương tiện trên 3.5 tấn",
    "Quyền ưu tiên ở giao lộ tiếp theo",
    "Đường ưu tiên",
    "Nhường đường",
    "Dừng lại",
    "Cấm xe",
    "Cấm xe trên 3.5 tấn",
    "Cấm vào",
    "Cảnh báo chung",
    "Cua nguy hiểm về bên trái",
    "Cua nguy hiểm về bên phải",
    "Đường cong kép",
    "Đường xấu",
    "Đường trơn",
    "Đường hẹp về bên phải",
    "Công trường",
    "Đèn giao thông",
    "Người đi bộ",
    "Băng qua đường",
    "Xe đạp đi qua",
    "Cảnh báo đường trơn do băng/tuyết",
    "Cảnh báo có động vật hoang dã băng qua",
    "Kết thúc tất cả giới hạn tốc độ và cấm vượt",
    "Rẽ phải",
    "Rẽ trái",
    "Chỉ hướng thẳng",
    "Đi thẳng hoặc rẽ phải",
    "Đi thẳng hoặc rẽ trái",
    "Luôn đi bên phải",
    "Luôn đi bên trái",
    "Buồng lái tròn bắt buộc",
    "Kết thúc cấm vượt",
]
class_names_en = [
        "Speed limit (20km/h)",
        "Speed limit (30km/h)",
        "Speed limit (50km/h)",
        "Speed limit (60km/h)",
        "Speed limit (70km/h)",
        "Speed limit (80km/h)",
        "End of speed limit (80km/h)",
        "Speed limit (100km/h)",
        "Speed limit (120km/h)",
        "No passing",
        "No passing for vehicles over 3.5 metric tons",
        "Right-of-way at the next intersection",
        "Priority road",
        "Yield",
        "Stop",
        "No vehicles",
        "Vehicles over 3.5 metric tons prohibited",
        "No entry",
        "General caution",
        "Dangerous curve to the left",
        "Dangerous curve to the right",
        "Double curve",
        "Bumpy road",
        "Slippery road",
        "Road narrows on the right",
        "Road work",
        "Traffic signals",
        "Pedestrians",
        "Children crossing",
        "Bicycles crossing",
        "Beware of ice/snow",
        "Wild animals crossing",
        "End of all speed and passing limits",
        "Turn right ahead",
        "Turn left ahead",
        "Ahead only",
        "Go straight or right",
        "Go straight or left",
        "Keep right",
        "Keep left",
        "Roundabout mandatory",
        "End of no passing",
        "End of no passing by vehicles over 3.5 metric tons",
        "No stopping",
        "Crossroad",
        "No entry for vehicles",
        "General danger"
    ]


def main():

    
    # python traffic.py loading => xem ảnh trong dự doán
    if sys.argv[1] == 'loading':
        print("[+] Loading model and testing")
        # Đường dẫn đến file mô hình đã được lưu
        model_path = "./model"

        # Get image arrays and labels for all image files
        images, labels = load_data('gtsrb')

        # Split data into training and testing sets
        labels = tf.keras.utils.to_categorical(labels)
        
       
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=TEST_SIZE
        )
        
        # Tải mô hình
        model = tf.keras.models.load_model(model_path)
        
        # Dự đoán nhãn của tập dữ liệu kiểm tra
        predictions = model.predict(x_test)
        
        # TODO Ddang xem anh 0->3 | -012
        for x in range(len(predictions[50:55])):
            max_index = np.argmax(predictions[x])
            zoomed_image = cv2.resize(x_test[x], (x_test[x].shape[1]*10, x_test[x].shape[0]*10))
            print("Ảnh này là : " + class_names[max_index])
            cv2.imshow(str(max_index), zoomed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return
    
    # python traffic.py image (ten anh) => dự đoán ảnh truyền vào
    if sys.argv[1] == 'image':
        print("[+] Loading image and predictions")
        
        # Đường dẫn đến file mô hình đã được lưu
        model_path = "./model"

        # Tải mô hình
        model = tf.keras.models.load_model(model_path)
        
        image = cv2.imread(sys.argv[2])
        res = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation= cv2.INTER_AREA)
        
        expanded_image = np.expand_dims(res, axis=0)
        
        # Dự đoán nhãn của tập dữ liệu kiểm tra
        predictions = model.predict(expanded_image)
        
        max_index = np.argmax(predictions)
        zoomed_image = cv2.resize(res, (res.shape[1]*10, res.shape[0]*10))
        
        # Tính toán tọa độ và kích thước khung dự đoán
        x, y, w, h = 0, 0, 300, 300

        # Vẽ khung dự đoán lên hình ảnh
        cv2.rectangle(zoomed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Viết chữ lên hình ảnh
        text = class_names_en[max_index]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(zoomed_image, text, (5, 30), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)


        print("Ảnh này là : " + class_names[max_index])
        cv2.imshow(str(max_index), zoomed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    
    
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")
    
    
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network => mạng thần kinh 
    model = get_model()
    
    # Fit model on training data
    history = model.fit(x_train, y_train, epochs=EPOCHS)

    import matplotlib.pyplot as plt

    # Lấy các thông tin lịch sử
    accuracy = history.history['accuracy']
    loss = history.history['loss']

    epochs = range(1, len(accuracy) + 1)

    
    # Biểu đồ độ chính xác và sai số
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.title('Training Accuracy and Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)
        
    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    # Path to data folder
    data_path = os.path.join(data_dir)

    # Number of subdirectories/labels
    number_of_labels = 0
    
    for i in os.listdir(data_path):
        number_of_labels += 1

    # Loop through the subdirectories
    for sub in range(number_of_labels):
        sub_folder = os.path.join(data_path, str(sub))

        images_in_subfolder = []

        for image in os.listdir(sub_folder):
            images_in_subfolder.append(image)

        # Open each image 
        for image in images_in_subfolder:

            image_path = os.path.join(data_path, str(sub), image)

            # Add Label
            labels.append(sub)

            # Resize and Add Image
            img = cv2.imread(image_path)
            res = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation= cv2.INTER_AREA)
            images.append(res)

    return (images, labels)


def get_model():
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    main()

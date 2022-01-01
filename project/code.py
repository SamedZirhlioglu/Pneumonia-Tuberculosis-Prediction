import os
import cv2 as cv2
import numpy as np
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt

from time import time
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def start_time(name):
    print(name + " islemi basladi.")
    return time()

def stop_time(start_time, name):
    print(name + " islemi bitti. ({:.2f} saniye)".format(round((time() - start_time), 2)))


# Veri setinde bulunan train, test, validation klasörlerinin yolu
TRAIN = r"E:\finished_projects\machine_learning\mehtap_banu\preprocessed_data\train"
VALID = r"E:\finished_projects\machine_learning\mehtap_banu\preprocessed_data\val"
TEST  = r"E:\finished_projects\machine_learning\mehtap_banu\preprocessed_data\test"

# Görsellerin boyutları ve eğitim değişkenleri
batch_size = 64
img_height = 512
img_widht = 512
noepochs = 1
epochs_range=range(noepochs)

# Train Dataset
base_time = start_time("Training dataset import")
train_ds = image_dataset_from_directory(
    TRAIN,
    validation_split = 0.3,
    subset = 'training',
    seed = 123,
    image_size = (img_height, img_widht),
    batch_size = batch_size
)
stop_time(base_time, "Training dataset import")

# Valid Dataset
base_time = start_time("Validation dataset import")
valid_ds = image_dataset_from_directory(
    VALID,
    validation_split = 0.3,
    subset = 'validation',
    seed = 123,
    image_size = (img_height, img_widht),
    batch_size = batch_size
)
stop_time(base_time, "Validation dataset import")

# Test Dataset
# Test konumundaki tüm fotoğrafların konumlarının listeye kaydedilmesi
test_dir = []
base_time = start_time("Test dataset import")
for img in os.listdir(TEST):
    test_dir.append(os.path.join(TEST, img))
stop_time(base_time, "Test dataset import")

# Sınıflar hakkında bilgi verildi
class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)

# Bir veri üzerinde çalışırken, aynı anda sıradaki verinin bellekte hazır hale getirilmesini sağlayan kod parçacığı
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Modelin oluşturulması
# [CNN Algoritması]
base_time = start_time("Model olusturma")
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_widht, 3)),
    layers.Conv2D(16,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes)
])
stop_time(base_time, "Model olusturma")

# Modeli compile ederek eğitime hazır hale getirme 
base_time = start_time("Model derleme(compile)")
model.compile(
    optimizer = 'adam',
    metrics=['accuracy'],
    loss=SparseCategoricalCrossentropy(from_logits=True)
)
stop_time(base_time, "Model derleme(compile)")

# Modelin fit edilmesi(eğitim/train)
base_time = start_time("Model egitme(fit)")
my_model=model.fit(
    train_ds,
    epochs = noepochs,
    validation_data = valid_ds
)
stop_time(base_time, "Model egitme(fit)")

# Eğitim sonrasında çıkan verinin değişkenlere atanması
acc = my_model.history['accuracy']
val_acc = my_model.history['val_accuracy']
loss = my_model.history['loss']
val_loss = my_model.history['val_loss']

# Son haline getirilen verinin, işleme sırasında oluşan değerlerinin tablolaştırılması
plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range,val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#write_console() fonksiyonunu konsola yazarak verdiğimiz klasördeki resimlerin sonuçlarını hem yazılı hem görsel olarak görebiliriz.
def write_console():
    root = tk.Tk()
    root.withdraw()
    for img_path in test_dir:
        img = load_img(img_path, target_size=(img_height, img_widht))
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(img_path)
        print("Bu fotograf: {}. OLASILIK:  {:.2f}"
            .format(class_names[np.argmax(score)], 100 * np.max(score)))

        """font = cv2.FONT_HERSHEY_SIMPLEX
        org = (00, 40)
        fontScale = 0.5
        color = (0, 0, 255)
        thickness = 1
        image = cv2.putText(
            cv2.imread(img_path),
            class_names[np.argmax(score)],
            org, font, fontScale, color, thickness, cv2.LINE_AA, False
        )
        cv2.imshow(img_path, image)
        cv2.waitKey()"""

write_console()
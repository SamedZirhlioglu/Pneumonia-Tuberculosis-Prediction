import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Prepocess yapılacak olan verinin dosya yolunu kaydettik.
PATH = r"E:\finished_projects\machine_learning\mehtap_banu\preprocessed_data\test"

# Preprocess sonucunda kaydedilecek olan fotoğrafların boyutunu kaydettik.
HEIGHT = 512
WIDTH = 512
IMG_SIZE = (HEIGHT, WIDTH)

# Fonksiyonu çağırırken gönderdiğimiz parametrelere göre, konumdaki belli uzantıya sahip görselleri import eden fonksiyon
def load_images(folder_name, extension):
    image_files = []
    
    # Process yapılacak olan konumda, belirtilen klasöre(folder_name) girerek içindeki görselleri tek tek dönen döngü
    for file in os.listdir(PATH + "\\" + folder_name):
        # Belli uzantılardaki(extension) dosyaları seçen if bloğu
        if file.endswith("." + extension):
            # Seçilmiş görselleri list'e kaydeden kod
            image_files.append(os.path.join(PATH, folder_name, file))
    
    # Seçilmiş verilerin return edilmesi
    return image_files

# Kendisine parametre olarak gönderilen görseli ekrana çıkartan fonksiyon
def display_one(img, title1 = "Orjinal"):
    # 'img'yi ekrana çıkartılacak görsel olarak belirler ve başlık olarak 'title1' ayarlar
    plt.imshow(img), plt.title(title1)
    # Görseli x-y eksenine yerleştirir
    plt.xticks([]), plt.yticks([])
    # Görseli ekrana çıkartır
    plt.show()

# Kendisine parametre olarak gönderilen 2 görseli ekrana çıkartan fonksiyon
def display_two(img1, img2, title1 = "Orjinal", title2 = "Düzenlenmiş"):
    # 'img1'i ekrana çıkartılacak görsel olarak belirler ve başlık olarak 'title1' ayarlar
    plt.subplot(121), plt.imshow(img1), plt.title(title1)
    # Görseli x-y eksenine yerleştirir
    plt.xticks([]), plt.yticks([])
    # 'img2'yi ekrana çıkartılacak görsel olarak belirler ve başlık olarak 'title2' ayarlar
    plt.subplot(122), plt.imshow(img2), plt.title(title2)
    # Görseli x-y eksenine yerleştirir
    plt.xticks([]), plt.yticks([])
    # Görseli ekrana çıkartır
    plt.show()

# Preprocess işlemini yöneten fonksiyon
def preprocessing(data):
    images = []
    resized_images = []
    no_noise = []
    segmented = []
    unknown = []
    
    #images
    # Görselleri, 'data'daki konum listesinden alır ve 'images'in içine kaydeder
    for img in data:
        images.append(cv2.imread(img, cv2.IMREAD_UNCHANGED))
    
    #resized_images
    # 'images' list'inin içindeki görselleri istediğimiz boyuta göre kırpar ve 'resized_images' içine kaydeder
    for i in range(len(images)):
        # [interpolation] = Fotoğrafları her yönünden eşit olarak boyutlandırmak için kullanılır
        res_img = cv2.resize(images[i], IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        resized_images.append(res_img)
    
    #no_noise
    # 'resized_images' içindeki görsellerin gürültüsünü silerek 'no_noise' içine kaydeder
    for i in range(len(resized_images)):
        blur = cv2.GaussianBlur(resized_images[i], (5, 5), 0)
        no_noise.append(blur)
    
    #grayscale
    # 'no_noise' içindeki grayscale olmayan görselleri seçip grayscale moduna çevirir
    for i in range(len(images)):
        if images[i].shape != IMG_SIZE:
            try:
                no_noise[i] = cv2.cvtColor(no_noise[i], cv2.COLOR_RGB2GRAY)
            except:
                continue

    #segmented
    # 'no_noise' içindeki verileri segmente edip 'segmented' içine kaydeder
    for i in range(len(no_noise)):
        ret, thresh = cv2.threshold(no_noise[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        segmented.append(thresh)

    #segment_bg & sure_fg
    # 'segmented' içindeki verileri; background ve foreground olarak ikiye ayırır
    # Segmente veriden background'u kaldırarak foreground kısmını 'unknown' içine kaydeder
    for i in range(len(segmented)):
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(segmented[i], cv2.MORPH_OPEN, kernel, iterations=2)
        # [bg] = Background => Ciğerin tamamını görsel olarak kaydeder, arkaplanı siler. Sadece ciğer kalır.
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # [fg] = Foreground => Ciğerin gereksiz kısımlarını görsel olarak kaydeder, arkaplanı siler. Sadece ciğerin gereksiz kısımları kalır.
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Ciğer kısmından, ciğerin gereksiz kısımlarını çıkartır. Geriye sadece ciğerin gerekli kısmı kalmış olur.
        unkn = cv2.subtract(sure_bg, sure_fg)
        unknown.append(unkn)
    
    # Verilerin düzenlenmiş halini, üstüne yazarak kaydeder
    for i in range(len(images)):
        cv2.imwrite(data[i], unknown[i])
    
    """for i in range(len(images)):
        display_two(images[i], resized_images[i], "Orijinal", "Yeniden Boyutlandırılmış")
        display_two(resized_images[i], no_noise[i], "Yeniden Boyutlandırılmış", "Gürültüsü Azaltılmış")
        display_two(no_noise[i], segmented[i], "Gürültüsü Azaltılmış", "Segmente Edilmiş")
        display_two(segmented[i], unknown[i], "Segmente Edilmiş", "Sonuç")"""


preprocessing(load_images("PNEUMONIA", "png"))
preprocessing(load_images("PNEUMONIA", "jpg"))
preprocessing(load_images("PNEUMONIA", "jpeg"))

preprocessing(load_images("TURBERCULOSIS", "png"))
preprocessing(load_images("TURBERCULOSIS", "jpg"))
preprocessing(load_images("TURBERCULOSIS", "jpeg"))

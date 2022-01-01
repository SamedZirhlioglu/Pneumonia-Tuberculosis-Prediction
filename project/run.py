import os
from datetime import datetime

# Mevcut zamanı string olarak return eden fonksiyon
def get_time_string():
    # Mevcut zamanı 'now' içine kaydettik
    now = datetime.now()
    # Kaydettiğimiz zamanı; gün_ay_yıl__saat_dakika_saniye olarak return ettik
    return now.strftime("%d_%m_%Y__%H_%M_%S")

# Python derleyicisinin konumunu kaydettik
python_dir = "C:/Users/SAMED/AppData/Local/Programs/Python/Python39/python.exe"
# Mevcut konumumuzu kaydettik
# [realpath] = Derlenen bu dosyanın tam konumunu return eden sistem fonksiyonu
current_dir = os.path.dirname(os.path.realpath(__file__))
# Çalıştırılacak dosyanın yolunu kaydettik
code_dir = os.path.join(current_dir, "code.py")
# Çalıştırma çıktılarının kaydedileceği dosyanın yolunu kaydettik
log_dir = os.path.join(current_dir, get_time_string() + ".txt")

# Kaydettiğimiz verileri kullanarak konsolu çalıştırdık
os.system(python_dir + " \"" + code_dir + "\" > \"" + log_dir)


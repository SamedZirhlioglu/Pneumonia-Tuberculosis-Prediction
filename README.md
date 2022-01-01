# Pneumonia Tuberculosis Prediction
## Göğüs filmi (X-Ray) taramasından yola çıkarak ***Zatürre/Tüberküloz*** tahmini yapılması

Hastalardan çekilen, Zatürre/Tüberküloz etiketli göğüs filmleri ile eğitilen projemiz; girdisi yapılan görüntüyü test eder. Test sonucunda kişinin zatürre mi tüberküloz mu olduğu belirlenir.

1. [Veri seti](https://www.kaggle.com/jtiptj/chest-xray-pneumoniacovid19tuberculosis) temin edildi.
2. [Önişleme aşamaları](./project/preprocess.py) gerçekleştirildi.
3. CNN modeli [oluşturuldu](./project/code.py) ve [eğitildi](./project/code.py).
4. Proje [run.py](./project/run.py) üzerinden çalıştırılırsa, sonuçlar [Outputs](./project/Outputs/) konumuna kaydedilir.
> Bu projede makine öğrenmesi kullanılmıştır.
# Hazır model ile drone video entegrasyonu

## Kullanıcının netleştirdiği hedef — 24 Kasım 2026

Hazır, güçlü bir modeli mevcut bilgisayarda drone görüntüsüne uygulamayı ve daha sonra canlı kamera akışına bağlamayı öğrenmek. Şu anda drone/kamera yok. İlk aşama kayıtlı örnek videodur. Önceki küçük UAVid modeli öğrenme ve deney kaydı olarak korunur.

Zorunlu koşul: model ve çalıştırma yazılımında ücretli lisans, sertifika, API veya bulut aboneliği gerektirmeyen bir yerel yol seçmek. Açık kaynak lisansları yine kullanım koşulları içerir; kod, hazır ağırlık ve örnek veri ayrı ayrı kontrol edilir. Kamera/donanım gideri bu yazılım koşulundan ayrıdır.

## İki ayrı görev

- **Nesne tespiti:** insan ve araç gibi nesnelerin konumu, kutusu, sınıfı ve güven puanı.
- **Çevrenin anlamsal bölütlenmesi:** yol, bina, ağaç gibi sınıfların piksel haritası. Hazır nesne tespit modelinin sınıf listesi bütün bu çevre sınıflarını kapsamayabilir.

İlk entegrasyon için nesne tespiti adayları araştırıldı. [YOLOX kaynak kodu Apache-2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE); kullanılacak ağırlığın koşulları ayrıca doğrulanmalı. [RT-DETR R18 model kartı](https://huggingface.co/PekingU/rtdetr_r18vd) ağırlık lisansını Apache-2.0 olarak listeliyor. [Ultralytics](https://www.ultralytics.com/license) ücretsiz AGPL ve ücretli Enterprise seçeneklerine sahip; bütün YOLO uygulamaları aynı lisansa sahip sayılmaz. **Henüz model seçilmedi, model ağırlığı indirilmedi veya video üzerinde çalıştırılmadı.**

## Örnek video kaynakları

- [Pexels drone trafik videoları](https://www.pexels.com/search/videos/drone%20traffic/): kısa bir yerel demo için. [Pexels lisansı](https://www.pexels.com/license/) ücretsiz indirme/kullanımı koşullarıyla açıklar. Videolar nesne doğruluk etiketleri sağlamaz.
- [VisDrone resmî video veri seti](https://github.com/VisDrone/VisDrone-Dataset#task-2-object-detection-in-videos): video nesne tespiti ve etiketleri. VisDrone-VID valset resmî sayfada yaklaşık 1,49 GB olarak listelenir. Kullanım koşulları indirme kaynağında ayrıca kontrol edilir; model lisansı veri lisansı yerine geçmez.

Arama ifadeleri: `drone traffic footage`, `aerial city traffic video`, `drone footage cars pedestrians`, `VisDrone VID dataset download`. İlk örnekte araçları belirgin gösteren, yavaş kamera hareketli, 20–30 saniyelik 1080p video yeterli olabilir. YouTube'dan bulunan bir videonun indirme/kullanım izni kendiliğinden varsayılmaz.

## Sonraki oturum

1. Lisansı doğrulanmış bir örnek video ve hazır ağırlık seçip kaynak/sürümünü kaydetmek. Henüz video indirilmedi.
2. Önce kısa bir kesitte yerel tahmin çalıştırmak; kutularla işaretlenmiş çıktı ve işlem süresi üretmek. Eğitim gerekmez.
3. Gecikme ölçümünde ısınma, video çözme ve model işlem sürelerini ayırmak; cihaz ve çözünürlüğü kaydetmek. Hız sayısı doğruluk kanıtı değildir.
4. Etiketli örneklerle hataları incelemek ve uygun adayları aynı koşullarda karşılaştırmak; bu ölçümlerden önce “en iyi model” ilan etmemek.
5. Kamera edinildiğinde aynı görüntü hattına desteklenen canlı kaynak eklemek; drone üzerinde çalıştırma ayrıca donanıma göre değerlendirilir.

Bu oturumda çalışma kullanıcı isteğiyle burada bırakıldı. Kaggle notebook'u veya paylaşım ayarı değiştirilmedi; yeni eğitim veya ücretli servis başlatılmadı.

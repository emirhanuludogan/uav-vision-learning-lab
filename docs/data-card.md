# Veri kartı — UAVid

## Kaynak ve görev

[UAVid](https://uavid.nl/), şehir sahnelerinde yüksek çözünürlüklü UAV görüntüleri üzerinde sekiz sınıflı anlamsal bölütleme verisidir. Her pikselin sınıfını tahmin etmek hedeflenir; nesne kutusu bulma veya instance segmentation bu ilk çalışmanın görevi değildir.

Resmî edinme yolu: sitede **Semantic Labelling with Images Only → University of Twente EOStore**. Kaynak sayfalar 23 Kasım 2026 tarihinde kontrol edildi. [Resmî renk tanımları](https://github.com/YeLyuUT/UAVidToolKit/blob/master/colorTransformer.py): clutter `(0,0,0)`, building `(128,0,0)`, road `(128,64,128)`, tree `(0,128,0)`, low vegetation `(128,128,0)`, moving car `(64,0,128)`, static car `(192,0,192)`, human `(64,64,0)`.

Resmî site UAVid 2020 için toplam 42 sekans (20 train, 7 validation, 15 test) bildiriyor. Kullanıcının README aktarımı da ilk 30 sekansa 12 ek sekans eklendiğini belirtiyor. Train/val paketleri için kullanıcı Kaggle'da 20/7 sekans ve 200/70 çift doğruladı; test paketi indirilmedi. Makaledeki eski sürüm sayıları ile 2020 paketini karıştırmamak gerekir.

## Bu projedeki durum

Topluluk kopyası yalnız tarihsel ön incelemedir: [günlük](journal/2026-11-23.md). Resmî train/val paketleri [Kaggle resmî kaynak kopyasına](https://www.kaggle.com/datasets/emirhanuludogan/paaaaa) yüklendi; veri başlığı **UAVid Train and Validation — Official Source Copy**, slug `paaaaa` olarak kaldı. Kullanıcı veriyi public yaptı; lisans seçimi `Specified in description`, açıklamadaki kaynak koşulu CC BY-NC-SA 4.0'dır. Orijinal ayrımlar korunur; yakın video kareleri rastgele train/val olarak yeniden bölünmez.

Kaggle kökü `/kaggle/input/datasets/emirhanuludogan/paaaaa`; image/label dizinleri iç içe `uavid_train/uavid_train/seq*` ve val benzeridir. Yerel hazırlama aracı tek katlı orijinal ZIP klasörlerini korur. Notebook ikisini de açıkça çözebilir. [24 Kasım kaydı](journal/2026-11-24.md) hangi sonuçların kullanıcı aktarımı, hangilerinin yerel yürütme olduğunu ayırır.

24 Kasım'da yerel arşivler açıldı ve notebook'un tam denetimi CPU'da çalıştırıldı: 270 çiftin tamamı okundu; sekans/sayı/boyut ve sınıf yüzdeleri Kaggle aktarımıyla eşleşti. Bilinmeyen RGB, boyut farkı ve okuma hatası 0. Tam denetim kaynak kökenini veya görüntü içerik sızıntısının yokluğunu kanıtlamaz.

Paket denetiminde indirme kaynağı, paket adı/sürümü, SHA-256 özeti, sekans ve dosya sayıları, eksik eşleşmeler, boyutlar, renkler ve olası içerik örtüşmeleri kaydedilmeli. `scripts.audit_uavid` eşleşme, boyut, okunabilirlik ve renk kontrollerini yapar; arşiv özeti veya içerik kopyası/sızıntı analizi yapmaz. Aynı sekans adının farklı klasörlerde görünmesi ayrıca incelenmelidir.

## Kullanım koşulları ve atıf

Veri [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) altında yayımlanmıştır; atıf, ticari olmayan kullanım ve aynı lisansla paylaşım koşulları için [resmî açıklamayı](https://uavid.nl/) ve lisans metnini okuyun. Veri lisansı ile bu repoda yazılan kodun lisansı ayrı konulardır. Repo veri arşivi, görüntü veya etiket içermez.

Ye Lyu, George Vosselman, Gui-Song Xia, Alper Yilmaz, Michael Ying Yang (2020). *UAVid: A Semantic Segmentation Dataset for UAV Imagery*. ISPRS Journal of Photogrammetry and Remote Sensing 165, 108–119. [DOI: 10.1016/j.isprsjprs.2020.05.009](https://doi.org/10.1016/j.isprsjprs.2020.05.009), [arXiv](https://arxiv.org/abs/1810.10438).

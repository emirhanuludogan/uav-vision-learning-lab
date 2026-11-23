# UAV Vision Learning Lab

Drone görüntülerinde nesne tespiti, çevrenin **anlamsal bölütlenmesi (semantic segmentation)** ve hazır modellerin video/kamera akışına entegrasyonunu öğrenmek için Türkçe çalışma günlüğü ve araçlar. Anlamsal bölütleme her piksele bina, yol veya insan gibi bir sınıf; nesne tespiti ise tekil nesnelere konum kutuları atar. UAV, *Unmanned Aerial Vehicle* (insansız hava aracı) demektir.

Bu repo üçüncü sınıf düzeyinde bir görüntü işleme öğrenme portföyüdür. Amaç; problem tanımı, veri güvenilirliği, deney tasarımı, değerlendirme ve hata analizini öğrenerek ileride staj ve TEKNOFEST başvurularında açıklanabilir bir çalışma sunmaktır. Kod geliştirmede yapay zekâ desteği kullanılır; deney kararlarının ve sonuçların anlaşılması öğrenme sürecinin merkezindedir.

**Güncel yön:** hazır modelleri ücretsiz yerel kullanım koşullarıyla kayıtlı drone videosunda değerlendirmek, ardından canlı kamera akışına bağlamak. Kamera henüz yok; model ve örnek video henüz seçilip çalıştırılmadı. [Video entegrasyonu ve sonraki oturum planı](docs/video-integration-plan.md).

**[Tarih sıralı günlük: 23 Kasım, 24 Kasım ve sonraki çalışmalar](docs/journal/README.md)**

## Notebook'u aç — yerel Jupyter veya Kaggle

**[01_UAVid_Yerel_ve_Kaggle.ipynb](notebooks/01_UAVid_Yerel_ve_Kaggle.ipynb)** bütün kodu içerir: veri inceleme, letterbox dönüşümü, veri yükleyici, aynı başlangıç modeli, açık eğitim başlatma, checkpoint ve hata inceleme. Hücreleri sohbetten kopyalamak gerekmez.

**[Kaggle'da gerçekten çalıştırılmış notebook arşivi](notebooks/archive/2026-11-24_kaggle_uavid_original.ipynb):** kullanıcıdan alınan kod hücreleri ve kayıtlı metin çıktıları korunur; bir gömülü veri görseli çıkarılmıştır. [Arşiv notları](notebooks/archive/README.md), [3 epoch sonuç tablosu](docs/experiments/2026-11-24_kaggle-baseline.csv), [eğitim logu](docs/experiments/2026-11-24_kaggle-training-log.txt). Arşivde Run All doğrudan eğitim başlatır; kontrollü kullanım için yukarıdaki taşınabilir notebook'u kullanın. Bu manuel bir arşivdir; Kaggle ve GitHub arasında otomatik eşitleme kurulmadı.

Bu bilgisayarda masaüstündeki `UAV-Vision-Lab/Jupyter-Baslat.cmd` dosyasını çift tıklayın. `Proje` bağlantısı bu depodaki aynı dosyaları açar. Kaggle'da aynı `.ipynb` dosyasını **Import Notebook** ile yükleyip [resmî kaynak kopyasını](https://www.kaggle.com/datasets/emirhanuludogan/paaaaa) ekleyin. **Eğitim varsayılan olarak kapalıdır.** [Açılış, ortam, veri ve reset/resume rehberi](docs/notebook-kullanim.md).

## Mevcut durum — 24 Kasım 2026

- İlk ön incelemede UAVid topluluk kopyası kullanıldı; bu tarihsel gözlemler 23 Kasım kaydında tutulur. Kullanıcının sonradan indirdiği notebook ise resmî veri üzerindeki çalışmayı ve 3 epoch eğitim çıktısını içerir; şimdi ayrı arşiv dosyası olarak bu repodadır.
- Kaynak sekansları belirsiz olan topluluk kopyası bırakıldı. Resmî UAVid image-only train/val ve README indirildi, Kaggle'a yüklendi. Veri lisansı ve yazar atıfları veri açıklamasında belirtiliyor.
- Kullanıcının Kaggle çıktıları: train 20 sekans/200 çift, val 7 sekans/70 çift; 270 çiftte eksik eşleşme, okuma hatası, boyut farkı veya bilinmeyen RGB yok. İnceleme kapsamı ve sınıf dağılımları günlükte.
- Kullanıcının Kaggle'da tamamladığı sıfırdan küçük modelin 3 epoch sonuçları aktarıldı: en iyi validation mIoU **0,046915 (%4,6915), epoch 1**. Son epoch'ta yalnız bina sınıfının IoU'su sıfırdan farklı. Bu ilk deney güçlü bir model değildir; kayıp azalırken mIoU artmamıştır.
- Önceki Kaggle checkpoint'i yerelde bulunmuyor; bu repo hazırlığında tam model eğitimi yeniden başlatılmadı. Yeni notebook eğitim ve çıktı dosyalarını benzersiz çalışma klasörlerine yazar.

Gözlemler ve sınırları: [23 Kasım tarihsel ön inceleme](docs/journal/2026-11-23.md), [24 Kasım resmî veri ve baseline kaydı](docs/journal/2026-11-24.md). Veri kaynağı ve kullanım koşulları: [veri kartı](docs/data-card.md).

## Yerel aracı çalıştırma

Python 3.10+ gerekir. Repo veri dosyası içermez; veriyi [resmî UAVid sitesindeki](https://uavid.nl/) **Semantic Labelling with Images Only → EOStore** bağlantısından ayrıca edinmek gerekir. Arşivleri ayrı bir veri klasörüne açın; `seq*/Images` ve `seq*/Labels` yapısını ve resmî train/val ayrımını koruyun.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m scripts.audit_uavid --root "D:\datasets\uavid\train" --output reports/train-audit.json
.\.venv\Scripts\python -m scripts.audit_uavid --root "D:\datasets\uavid\val" --output reports/val-audit.json
```

Yollar örnektir; açtığınız paketteki gerçek dizinleri kullanın. Kaggle'da aynı modül `--root /kaggle/input/<ozel-veri>/<train-klasoru>` ile çalıştırılabilir; repo çalışma dizininde veya Python yolunda olmalıdır. Önce `python -m pip install -r requirements.txt` ile bağımlılığı kurun.

Araç tüm PNG çiftlerini okumaya çalışır; sekans başına sayıları, eksik eşleşmeleri, boyut farklarını, okuma hatalarını ve RGB etiket histogramını JSON'a yazar. Hata bulursa çıkış kodu 1, temiz kontrolde 0 döner. Veri seti büyük olduğunda renk sayımı zaman alabilir. Bir klasördeki dosya ve renk kontrolü; veri kaynağını, train/val bağımsızlığını veya sızıntı bulunmadığını kanıtlamaz. Raporlar ve veri dosyaları Git dışında tutulur.

## Sonraki adımlar

1. Örnek video ve hazır model ağırlığı için ücretsiz kullanım koşullarını doğrulamak; mevcut bilgisayarda kısa video tahmini üretmek.
2. Aynı koşullarda adayların hızını ve etiketli örneklerde doğruluğunu karşılaştırmak; canlı kamera entegrasyonuna temel hazırlamak.
3. Önceki deneyin en iyi Kaggle checkpoint'i ve özgün CSV'sini ayrıca arşivlemek; henüz bu dosyalara erişilmedi. Bina ağırlıklı tahmin şüphesi checkpoint geldiğinde histogram/overlay ile incelenebilir.
4. Her çalışma gününü tarihli günlükte belgelemek; sonuçları ve sınırlamaları açıklayan demo/rapor hazırlamak. Simülasyon sonraki fazdır.

## Kaynaklar ve lisans

UAVid verisi [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) koşullarındadır; veri bu repoda dağıtılmaz. Bu koşullar özgün repo koduna kendiliğinden uygulanmış bir kod lisansı değildir. Bu aşamada repo için ayrıca bir kod lisansı seçilmedi.

Veri ve makale atfı: Ye Lyu, George Vosselman, Gui-Song Xia, Alper Yilmaz ve Michael Ying Yang, *UAVid: A Semantic Segmentation Dataset for UAV Imagery*, ISPRS Journal of Photogrammetry and Remote Sensing, 165, 108–119, 2020. [Makale](https://arxiv.org/abs/1810.10438), [DOI](https://doi.org/10.1016/j.isprsjprs.2020.05.009), [resmî site](https://uavid.nl/), [renk tablosu](https://github.com/YeLyuUT/UAVidToolKit/blob/master/colorTransformer.py).

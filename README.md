# UAV Vision Learning Lab

Drone görüntülerinde **anlamsal bölütlemeyi (semantic segmentation)** adım adım öğrenmek için Türkçe çalışma günlüğü ve küçük, tekrar kullanılabilir araçlar. Anlamsal bölütleme, görüntüdeki her piksele bina, yol veya insan gibi bir sınıf atar. UAV, *Unmanned Aerial Vehicle* (insansız hava aracı) demektir.

Bu repo üçüncü sınıf düzeyinde bir görüntü işleme öğrenme portföyüdür. Amaç; problem tanımı, veri güvenilirliği, deney tasarımı, değerlendirme ve hata analizini öğrenerek ileride staj ve TEKNOFEST başvurularında açıklanabilir bir çalışma sunmaktır. Kod geliştirmede yapay zekâ desteği kullanılır; deney kararlarının ve sonuçların anlaşılması öğrenme sürecinin merkezindedir.

## Mevcut durum — 23 Kasım 2026

- Kaggle'daki `Drone Görüntüleri — 01 Veri İnceleme` çalışmasında bir UAVid topluluk kopyası incelendi. Bulgular kullanıcı tarafından paylaşılan çıktılara dayanır; özgün notebook bu repoda yoktur.
- Bir görüntü ve referans etiketi, sınıf dağılımı ve dosya adları incelendi. **Henüz model eğitilmedi veya model başarısı ölçülmedi.**
- Kaynak sekansları belirsiz olan topluluk kopyası yerine resmî UAVid verisiyle devam etme kararı alındı.
- Kullanıcının son bildirimine göre resmî doğrulama ZIP'i ve README indirildi, eğitim ZIP'i indiriliyordu. Paketlerin içeriği henüz doğrulanmadı ve resmî veri henüz Kaggle'a yüklenmedi.

Gözlemler ve sınırları: [23 Kasım çalışma günlüğü](docs/journal/2026-11-23.md). Veri kaynağı ve kullanım koşulları: [veri kartı](docs/data-card.md).

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

1. Resmî train/val paketleri ve README'yi kontrol edip sürüm, dosya sayıları ve özgün sekans ayrımını kaydetmek; özel Kaggle verisi olarak hazırlamak. Test paketi şimdilik kapsam dışı.
2. Sekans kaynağını ve olası örtüşmeleri değerlendirmek; ardından basit bir başlangıç modeli eğitmek.
3. Sınıf başına IoU ve mIoU raporlamak, küçük nesnelerde hataları görselleştirmek.
4. Her seferinde bir değişkeni değiştirerek kontrollü deney yapmak; sonuçları günlükte belgelemek.
5. Rapor ve demo hazırlamak. Simülasyon daha sonraki fazdır.

## Kaynaklar ve lisans

UAVid verisi [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) koşullarındadır; veri bu repoda dağıtılmaz. Bu koşullar özgün repo koduna kendiliğinden uygulanmış bir kod lisansı değildir. Bu aşamada repo için ayrıca bir kod lisansı seçilmedi.

Veri ve makale atfı: Ye Lyu, George Vosselman, Gui-Song Xia, Alper Yilmaz ve Michael Ying Yang, *UAVid: A Semantic Segmentation Dataset for UAV Imagery*, ISPRS Journal of Photogrammetry and Remote Sensing, 165, 108–119, 2020. [Makale](https://arxiv.org/abs/1810.10438), [DOI](https://doi.org/10.1016/j.isprsjprs.2020.05.009), [resmî site](https://uavid.nl/), [renk tablosu](https://github.com/YeLyuUT/UAVidToolKit/blob/master/colorTransformer.py).

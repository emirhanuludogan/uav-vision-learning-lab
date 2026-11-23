# Kaggle'da çalıştırılan notebook arşivi

[24 Kasım UAVid notebook'u](2026-11-24_kaggle_uavid_original.ipynb), kullanıcının Kaggle'dan indirip sağladığı **Drone Görüntüleri — 01 Veri İnceleme** dosyasından alınmıştır. Kod hücreleri değiştirilmedi; mevcut metin çıktıları ve execution count değerleri korundu. Veri görsellerini repoda yeniden dağıtmamak için bir gömülü `image/png` çıktısı çıkarıldı. Kaynak dosya İndirilenler klasöründe değiştirilmeden korundu.

Bu arşiv **yerelde yeniden çalıştırılmadı**. Özellikle son dolu hücre doğrudan 3 epoch eğitim başlatır ve aynı Kaggle çıktı dosyalarına yazar. Arşivi öncelikle geçmiş kod ve sonuçları okumak için kullanın. Kontrollü reset/resume ve yerel/Kaggle kullanım için [taşınabilir notebook'u](../01_UAVid_Yerel_ve_Kaggle.ipynb) açın; o dosyada eğitim varsayılan olarak kapalıdır.

Kaggle veri yolu kodda korunur; yerelde arşivi açmak bu yolu otomatik uyarlamaz. İhracın `kaggle` metadata alanı GPU kapalı/veri kaynağı boş derken kayıtlı çıktı iki Tesla T4 ve `cuda:0` kullanımını gösterir. Metadata, gerçek eğitim ortamının kesin kaydı olarak yorumlanmaz. Çıktıda Python 3.13.16, Torch 2.11.0+cu130 ve torchvision 0.26.0+cu130 bilgileri bulunur; bunları Windows'a kurulum sabitlemesi olarak kullanmayın. Kod tek `cuda:0` cihazını kullanır; iki GPU listelenmesi iki GPU ile eğitim yapıldığı anlamına gelmez.

Kaynak/arşiv SHA-256 özetleri ve yapılan çıkarma [köken kaydında](../../docs/experiments/2026-11-24_export-provenance.json). Eğitim çıktısı [metin logunda](../../docs/experiments/2026-11-24_kaggle-training-log.txt), özet tablo [CSV'de](../../docs/experiments/2026-11-24_kaggle-baseline.csv). CSV, notebook'un altı ondalıklı metin tablosundan çıkarıldı; Kaggle'daki özgün `ilk_egitim_gecmisi.csv` indirilmiş değildir. `.pt` ağırlık dosyası da henüz alınmadı.

Eski checkpoint'i daha sonra inceleyecek kişi için **koddan doğrulanmış biçim**: `model_state_dict`, `optimizer_state_dict`, `epoch`, `validation_miou`, `siniflar`, `boyut`, `ignore_index`, `model_adi`. Eski ağ ağırlık anahtarları `ag.` öneki taşır; taşınabilir notebook'un doğrudan Sequential ağırlık biçimiyle aynı değildir. Dosya alınmadan ağırlık dönüşümü veya yükleme yapılmadı.

Veri ve etiketler UAVid'in [CC BY-NC-SA 4.0](https://uavid.nl/) koşullarına tabidir; kod lisansıyla karıştırılmamalıdır.

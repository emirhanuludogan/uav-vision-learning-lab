# Aynı notebook ile yerel Jupyter ve Kaggle

## Masaüstünden aç

Masaüstündeki `UAV-Vision-Lab` klasöründe **Jupyter-Baslat.cmd** dosyasını çift tıklayın. JupyterLab tarayıcıda açılır; `notebooks/01_UAVid_Yerel_ve_Kaggle.ipynb` dosyasını açın. Terminal penceresi oturum boyunca açık kalmalı. Notebook'un çekirdeği **Python 3 (ipykernel)** olmalı. Hücreleri sırayla Shift+Enter ile çalıştırabilir veya Run All kullanabilirsiniz. Eğitim varsayılan olarak kapalıdır.

Masaüstü klasöründeki `Proje`, kayıtlı `Documents/ChatGPT/kaggle` Git deposuna bir Windows junction bağlantısıdır. İçinde gördüğünüz dosyalar aynı fiziksel dosyalardır; ikinci bir checkout veya senkronize edilecek kopya yoktur. Kaynak depoyu taşımayın; bağlantı hedefini veya `Proje` içeriğini topluca silmeyin. Klasörü temizlemek gerekirse yalnız bağlantının kendisini kaldırmak ile hedef dosyaları silmek farklı işlemlerdir.

PowerShell alternatifi (proje klasöründeyken):

```powershell
.\start-jupyter.cmd
```

Bu makinede mevcut Python 3.10.11 ortamı, Torch 2.7.1+cu118 ve RTX 4050 ile CUDA'yı tanıyor. Bu gözlem tüm makineler için kurulum reçetesi değildir. Büyük paket kurulumu veya tam eğitim bu hazırlık sırasında başlatılmadı. Notebook `auto` ile CUDA varsa `cuda:0`, yoksa CPU seçer; `CIHAZ_SECIMI="cpu"` ile CPU açıkça seçilebilir.

## Yeni makinede ortam

Mevcut çalışan ortam varsa koruyun. Ayrı ortam gerekirse:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-notebook.txt
```

Torch ayrı kurulur: [resmî seçicide](https://pytorch.org/get-started/locally/) Windows/Pip/Python ve CPU veya sürücünüzle uyumlu CUDA'yı seçin; gösterilen `pip` komutunu `.venv\Scripts\python -m pip ...` biçiminde çalıştırın. Kaggle'ın CUDA sürümünü Windows'a sabitlemeyin. Repo yalnız `torch` kullanır; torchvision veya torchaudio gerekmez. Yeni Torch sürümü Python 3.11+ isteyebilir; seçtiğiniz sürümün Python uyumunu kontrol edin. Hazırlıkta kullanılan sürümler tekrar üretim kaydıdır, platformlar arasında evrensel kilit dosyası değildir.

Başlatıcı `.venv` varsa onu, yoksa PATH'teki Python'u kullanır. Terminalden Jupyter açarken de aynı Python'u kullanın. Notebook'un ilk hücresi etkin Python/Torch/CUDA sürümlerini gösterir.

## Yerel veriyi hazırlama

Bu bilgisayarda kaynak arşivler masaüstündeki `uavid` klasöründe korunur. Hazırlama komutu veriyi Git dışında tutulan `data/official` altına açar. Mevcut hedefi birleştirmez veya ezmez; yol taşmasını, symlink ve çakışan ZIP üyelerini reddeder. Eksik açma başarısızsa benzersiz geçici klasör bırakılır; notebook bunu otomatik seçmez.

```powershell
python -m scripts.prepare_local_data --source "$env:USERPROFILE\Desktop\uavid" --destination ".\data\official"
```

Bu görev sırasında veri zaten hazırlanmışsa komutu tekrar çalıştırmanız gerekmez. Başlatıcı `UAVID_DATA_ROOT` değişkenini bu yerel hedefe ayarlar. Notebook tek başına açılırsa mevcut dizinin üstlerinde `data/official` arar. İsterseniz ilk ayar hücresinde `VERI_KOKU` ile train ve val içeren kökü açıkça seçin. Birden fazla geçerli split kopyası varsa seçim yapılmaz, hata açıklanır. Eski topluluk veri kaynağı aranmaz.

## Kaggle'da aynı dosyayı aç

1. Masaüstünde `Proje/notebooks/01_UAVid_Yerel_ve_Kaggle.ipynb` dosyasını seçin. Kaggle notebook arayüzünde **Import Notebook / Upload** ile bu `.ipynb` dosyasını yükleyin. Hücre kopyalamak gerekmez.
2. Input/Add Data ile [UAVid Train and Validation — Official Source Copy](https://www.kaggle.com/datasets/emirhanuludogan/paaaaa) verisini bağlayın. Başlık değişmiş olsa da slug `paaaaa` olarak kalır.
3. İsterseniz Accelerator ayarından GPU açın. İlk hücre gerçek cihazı gösterir. Repo modülü veya ek kod paketi yüklemeye gerek yoktur; notebook kendi kodunu içerir.
4. Hücreleri yukarıdan aşağı çalıştırın. `TAM_DENETIM=True` dosya ve RGB kontrolünü tekrarlar. Eğitim için `EGITIMI_BASLAT=True` ayarlayıp eğitim hücresini çalıştırın; varsayılan 3 epoch'tur.
5. Save Version ile notebook ve `uavid-runs` çıktılarını saklayın/indirin. Yerel ve Kaggle çıktıları otomatik senkronize olmaz; her biri farklı, benzersiz çalışma klasörüne yazar. Ayarları ve manifesti `config.json` üzerinden karşılaştırın.

Yerelde notebook kaydedilince masaüstü bağlantısında ve repoda aynı dosya değişir. Kaggle kopyası bağımsızdır; Kaggle'da yaptığınız kod değişikliklerini notebook indirerek bilinçli biçimde geri alın, eşzamanlı değişmiş yerel dosyanın üzerine düşünmeden yazmayın. Git'e hazırlarken çıktıları temizleyin: Jupyter menüsünden **Clear Outputs of All Cells**, sonra kaydedin.

## Eğitim ve sürdürme kuralları

`reset` her çağrıda seed 42 ile yeni ağ, Adam ve yeni çalışma klasörü kurar. `resume` yalnız sizin ürettiğiniz checkpoint'ten devam eder; `DEVAM_CHECKPOINT` dolu olmalı, sınıf sırası, veri manifesti, boyut, seed ve batch eşleşmeli. `EPOCH_SAYISI` ek epoch değil toplam hedeftir. Örneğin 3 epoch sonrası 5 yazmak iki epoch daha çalıştırır. Sürdürme de yeni çıktı klasörüne yazar.

Kayıtlar model, optimizer, epoch, sınıf haritası, hedef boyut, ignore index, RNG durumları ve geçmişi içerir. Validation mIoU ile seçilen `best.pt` bağımsız test başarısı değildir. Kontrolsüz eğitim tekrarlarıyla validation'a fazla uyum sağlayabileceğinizi unutmayın. Checkpoint yalnız güvendiğiniz kendi dosyanız olmalı; yükleme `weights_only=True` kullanır.

Notebook'taki veri/başlangıç modeli kontrolleri eğitim sonucu değildir. İlk ileri geçişte CE yaklaşık 2 civarında olabilir; daha önce paylaşılan 2.0729 sayısını her ortamda aynen beklemeyin. Aynı seed GPU ve CPU'da aynı sonucu garanti etmez.

## Kaynak dosyayı güncelleme ve doğrulama

Okunabilir kaynak `notebooks/uavid_notebook.py`, dağıtılan dosya `.ipynb`'dir. Kaynak değiştiğinde `python -m scripts.build_notebook` ile çıktısız notebook'u yeniden oluşturun. `.ipynb` bağımsız çalışır; bu `.py` dosyasına ihtiyaç duymaz.

```powershell
python -m scripts.build_notebook
python -m unittest discover -s tests -v
python -m scripts.smoke_notebook
```

Son komut, resmî yerel veri hazırsa notebook'u temiz bir CPU kernel'inde çalıştırır; **eğitim kapalıdır**. Sentetik eğitim/optimizer/checkpoint testleri gerçek model başarısını ölçmez.

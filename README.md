# Image Processing with Yolov8 - Fighting UAV

## 📌 Proje Açıklaması

Bu proje, 2023-2024 Teknofest Savaşan İHA yarışması için geliştirilmiştir. Amaç, rakip İHA'ları tanıyıp işaretlemek ve kamikaze görevindeki QR kod şifresini okumak.

- Yolov8 algoritması ile 80 sınıf arasından **sadece uçakları algılayacak şekilde filtreleme** yapılmıştır.
- Algılanan uçaklar, görüntü üzerinde işaretlenmektedir.
- Ayrıca bir **QR kod okuyucu** sistemi entegre edilmiştir.
- Görevler için hazır hale getirilmiştir.

## 🚀 Kullanılan Teknolojiler

- Python
- OpenCV
- Ultralytics Yolov8
- QR Code Reader (pyzbar, opencv vb.)

## 🛠️ Kurulum ve Kullanım

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt

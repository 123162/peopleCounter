# Kamera ile İçeri ve Dışarı Giren Kişi Sayısını Sayan Uygulama

Bu uygulama, kamera üzerinden video akışını izleyerek içeri ve dışarı giren kişi sayısını takip eder. Aynı zamanda YOLO modeli ile nesne tespiti yapar ve takip algoritmalarını kullanarak her kişiyi izler.

## Nasıl Çalışır?

1. **Gereksinimler:**
   - Python 3.x
   - Gerekli kütüphaneleri yüklemek için: `pip install -r requirements.txt`

2. **Video Dosyasını Belirleme:**
   - `cap = cv2.VideoCapture('a.mp4')` satırında kullanmak istediğiniz video dosyasını belirleyin.

3. **Yapılandırma:**
   - İlgili bölgeleri (`cy1`, `cy2`) ve diğer parametreleri ihtiyacınıza göre güncelleyebilirsiniz.

4. **Çalıştırma:**
   - `python your_script.py` komutunu kullanarak uygulamayı başlatın.

5. **Sonuçları İzleme:**
   - İçeri ve dışarı giren kişi sayısını ekran üzerinde izleyebilirsiniz.



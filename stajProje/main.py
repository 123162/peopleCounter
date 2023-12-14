import cv2
import sqlite3
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone


# SQLite veritabanı bağlantısı oluştur
conn = sqlite3.connect('personCount.db')
cursor = conn.cursor()

# Tabloyu oluştur (eğer zaten varsa bu satırı atlayacak)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS personCount (
        frame_number INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        inside_count INTEGER,
        outside_count INTEGER,
        total_count INTEGER,
        event_label TEXT
    );
''')
conn.commit()

model = YOLO('best.pt')  # best.pt ağırlıklı dosyamızı kullanıyoruz.
cap = cv2.VideoCapture('a.mp4')  # video üzerinden görüntüleri alıyoruz
#rtsp_url = 'rtsp://admin:bts112233@192.168.2.133/live'
#cap = cv2.VideoCapture(rtsp_url)

etiket = open("coco.txt", "r")  # coco.txt dosyasının yolunu veriyoruz. modelin algılayabileceği nesnelerin isimlerini içeriyor
data = etiket.read()  # veri olarak coco.txt dosyasındaki etiketleri okuyoruz
classList = data.split("\n")  # etiketlerin olduğu veriyi satırlara göre böler ve her birini bir listeye atar

count = 0  # nesne algılama sayacı
persondown = {}
tracker = Tracker()  # nesne takip algoritmasından bir nesne oluşturdum
counter1 = []  # içeri giren kişilerin sayısını takip ediyor
personup = {}
counter2 = []  # dışarı çıkan kişilerin sayısını takip ediyor.
cy1 = 194
cy2 = 220
offset = 6

while True:  # videodaki kareleri sürekli okumak için kullanılır.
    ret, frame = cap.read()  # başarı durumu,okunan kare=cap nesnesinden(video) bir kare okur.
    if not ret:  # eğer kare başarı ile okunmazsa videodan çıkılır.
        break
    count += 1  # kaç kare okunduğunu tespit eder ve her kare okunmasında sayaç bir artar
    if count % 3 != 0:  # işlemi optimize etmeye yarayan her üç karede bir nesne algılama işlemi yapmadır.
        continue
    frame = cv2.resize(frame, (1020, 500))  # modelin daha hızlı çalışması için belirli boyutlara sokuyoruz.
    results = model.predict(frame)  # tespit ettiği her nesneyi result değişkenine atama
    a = results[0].boxes.data  # ilk sonuç setinden kutu verileri çekilir ve a değişkenine atanır.
    px = pd.DataFrame(a).astype("float")  # pandas kütüphanesi ile a değişkeninden kutu bilgileri çekilip bir dataFrame oluşturulur ve kutu bilgileri daha kolay işlenir
    list = []  # boş bir liste oluşturulur.
    for index, row in px.iterrows():  # her bir satır,index değeri almak için px üzerinde döngü başlatılır.
        x1 = int(row[0])  # x1-y1 sol üst köşesinden kordinat bilgilerini alır.
        y1 = int(row[1])
        x2 = int(row[2])  # x2-y2 sağ alt köşesinden bilgiler alır.
        y2 = int(row[3])
        d = int(row[5])
        c = classList[d]  # classlisteyi kullanarak sınıf adlarını alır
        if 'person' in c:  # eğer içinde person varsa listeye ekler.
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)  # takip algoritmam ile listedeki değerleri güncelliyorum.
    for bbox in bbox_id:  # her bir takip edilen nesnenin bilgilerini içeren listeyi döngüye aldık
        x3, y3, x4, y4, id = bbox  # sol üst ,id ve sağ alt köşe kordinat bilgilerini içeren bir alt liste oluşturdum.
        cx = int(x3 + x4) // 2  # nesnenin kordinatlarını alarak ikiye bölüp merkez noktalarını hesaplar.
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)  # merkez noktası,yarıçap,kırmızı renk,dairenin içi dolu olsun.circle ile daire çizdik

        if cy1 < (cy + offset) and cy1 > (cy - offset):  # belirlediğim cy1(194) değerinin alt ve üst limitlerini belirliyorum.
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # eğer nesne bu bölgeden geçiyorsa nesnenin etrafına çizgi içine çiziyorum 2 piksel kalınlığında amvi renk
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)  # çizilen dikdörtgenin içine id yazan metin ekleme
            persondown[id] = (cx, cy)  # bu nesnenin merkez noktalarını persondown sözlüğüne ekler.

        if id in persondown:  # id sözlüğün içinde ise yani bu nesne belirli bir bölgeden geçmişse
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)  # algılanan nesneyi kırmızı çizgi ile çiziyorum
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)  # çizilen  dikdörtgenin içine nesnenin id sini yazıyorum
                if counter1.count(id) == 0:  # eğer bu nesnenin kimliği counter1 listesine daha önce eklenmemişse
                    counter1.append(id)  # id yi counter1 sayacına ekliyorum
        ##----------------------------
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            personup[id] = (cx, cy)

        if id in personup:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if counter2.count(id) == 0:
                    counter2.append(id)
    cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)

    down = (len(counter1))  # içeri giren kişilerin sayısını sayıyor
    up = (len(counter2))  # dışarı çıkan kişileri sayıyor

    cvzone.putTextRect(frame, f'Iceride={down}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'Disarida={up}', (50, 160), 2, 2)

    # Güncellenmiş SQLLite veritabanına güncel içeri ve dışarı kişi sayılarını yaz
    total_count = down + up  # İçeri giren ve dışarı çıkan toplam kişi sayısı
    event_label = "Some Event"  # İsteğe bağlı: Olay etiketi
    # SQLLite veritabanına güncel içeri ve dışarı kişi sayılarını yaz
    cursor.execute('''
        INSERT INTO personCount (timestamp, inside_count, outside_count, total_count, event_label)
        VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?)
    ''', (down, up, total_count, event_label))
    conn.commit()

    cv2.imshow("Cam1", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from datetime import datetime




classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 460)

# cap = cv2.VideoCapture("hareketli_ucak2.mp4")

# video kayıt için fourcc ve VideoWriter tanımlama
cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
success, img = cap.read()
print(img.shape)
yukseklik = img.shape[0]
genislik = img.shape[1]

cv2.imwrite("ornek_resim.jpg", img)
size = list(img.shape)
del size[2]
size.reverse()
video = cv2.VideoWriter("kaydedilen_video.mp4", cv2_fourcc, 24, size)  # output video name, fourcc, fps, size

model = YOLO("yolov8n.pt")

qr_detector = cv2.QRCodeDetector()

vurus_suresi = 4  # Vuruş süresi 4 saniye
start_time = datetime.now()  # Başlangıç zamanını al

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)

    results = model(img, stream=True)
    aeroplane_detected = False  # Aeroplane algılandı mı?
    vurus_basladi = False  # Vuruş başladı mı?
    vurus_baslama_zamani = None  # Vuruşun başlama zamanı
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            currentClass = classNames[cls]
            cv2.rectangle(img, (280, 72), (720 + 280, 720 - 72), (255, 0, 0), 2)

            if currentClass == "aeroplane" and conf > 0.3:
                aeroplane_detected = True
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                cx2, cy2 = 1280 // 2, 720 // 2
                cv2.circle(img, (cx2, cy2), 5, (255, 0, 255), cv2.FILLED)

                cv2.line(img, (cx2, cy2), (cx, cy), (255, 0, 0), 1)

    # QR kodu tespiti
    data, bbox, _ = qr_detector.detectAndDecode(img)
    if bbox is not None:
        for i in range(len(bbox)):
            pt1 = tuple(bbox[i][0].astype(int))
            pt2 = tuple(bbox[(i + 1) % len(bbox)][0].astype(int))
            cv2.line(img, pt1, pt2, color=(255, 0, 255), thickness=2)
            cv2.putText(img, data, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Saati ekrana yazdır
    current_time = datetime.now()
    saat = current_time.strftime("%H:%M:%S")
    salise = current_time.microsecond // 100000
    cv2.putText(img, f"{saat}:{salise:02}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Vuruş süresi kontrolü
    if aeroplane_detected:
        if not vurus_basladi:  # Vuruş başlamadıysa başlat
            vurus_basladi = True
            vurus_baslama_zamani = time.time()
        else:
            gecen_sure = time.time() - vurus_baslama_zamani
            if gecen_sure >= vurus_suresi:  # Vuruş süresi dolduğunda
                cv2.putText(img, "Vurus basarili", (640, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        vurus_basladi = False  # Aeroplane algılanmadıysa vuruş başlamadı kabul et

    # FPS değerini ekrana yazdır
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f"FPS: {int(fps)}", (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # video kayıt
    video.write(img)

    print("fps: ", int(fps))

    cv2.imshow("Image", img)

    # q tuşuna basıldığında video kaydını bitir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import screeninfo

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Goruntu - 4 Koseyi Sec", image_display)

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]        # top-left
    ordered[2] = pts[np.argmax(s)]        # bottom-right
    ordered[1] = pts[np.argmin(diff)]     # top-right
    ordered[3] = pts[np.argmax(diff)]     # bottom-left
    return ordered

def resize_to_screen(image, screen_w, screen_h):
    h, w = image.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def get_screen_resolution():
    monitor = screeninfo.get_monitors()[0]
    return monitor.width, monitor.height

# --- Başlangıç ---

image = cv2.imread("sayfa.jpg")
if image is None:
    print("HATA: 'sayfa.jpg' dosyası bulunamadı.")
    exit()

image_display = image.copy()
cv2.namedWindow("Goruntu - 4 Koseyi Sec", cv2.WINDOW_NORMAL)
cv2.imshow("Goruntu - 4 Koseyi Sec", image_display)
cv2.setMouseCallback("Goruntu - 4 Koseyi Sec", click_event)

while True:
    if len(points) == 4:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        print("İşlem iptal edildi.")
        exit()

cv2.destroyAllWindows()

src_pts = order_points(points)

width = int(max(
    np.linalg.norm(src_pts[0] - src_pts[1]),
    np.linalg.norm(src_pts[2] - src_pts[3])
))
height = int(max(
    np.linalg.norm(src_pts[0] - src_pts[3]),
    np.linalg.norm(src_pts[1] - src_pts[2])
))

# --- DİK GÖRÜNÜM İÇİN YÜKSEKLİĞİ ARTIR ---
height = int(height * 1.3)  # %30 artırdık, burayı istediğin gibi değiştir

dst_pts = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(image, M, (width, height))

screen_w, screen_h = get_screen_resolution()
resized_output = resize_to_screen(warped, screen_w, screen_h)

cv2.namedWindow("Kusbakisi - Ekran Boyutlu", cv2.WINDOW_NORMAL)
cv2.namedWindow("Kusbakisi - Tam Netlik", cv2.WINDOW_NORMAL)
cv2.imshow("Kusbakisi - Ekran Boyutlu", resized_output)
cv2.imshow("Kusbakisi - Tam Netlik", warped)

cv2.imwrite("kusbakisi_sonuc.jpg", warped)
print("Kuşbakışı görünüm kaydedildi: kusbakisi_sonuc.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()

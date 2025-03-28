import os
import re
import requests
from bs4 import BeautifulSoup
import random
import string
from ultralytics import YOLO
import cv2

collected_ims = './collected_ims'
os.makedirs(collected_ims, exist_ok=True)
cropped_ims = './cropped_im.png'

def face_detect(image_path):
    print(f"Đang phát hiện mặt trong ảnh: {image_path}")
    # Load the YOLOv8 model
    model = YOLO('./yolov11n-face.pt')  # Use the appropriate model for your task
    im = cv2.imread(cropped_ims)

    results = model.predict(im, conf=0.5)  # Adjust confidence threshold as needed
    if len(results[0].boxes) == 1:
        boxes = results[0].boxes 
        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Expand the cropping area a bit to get the entire face
            h, w = im.shape[:2]
            margin = int(min((x2 - x1), (y2 - y1)) * 0.2)  # Add 20% margin

            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            face_img = im[y1:y2, x1:x2]
            cv2.imwrite(image_path, face_img)  # Save the cropped image
        return cropped_ims
    else:
        return None

def get_image(url, index, gender):
    kol_path = os.path.join(collected_ims, f'{index}')

    # Lấy nội dung HTML của trang
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Không lấy được trang, status code: {response.status_code}")

    soup = BeautifulSoup(response.text, 'html.parser')

    # Tìm tất cả các thẻ img có thuộc tính alt bắt đầu bằng "thumbnail-"
    img_tags = soup.find_all('img', alt=re.compile(r'^thumbnail-\d+'))
    if len(img_tags) < 2:
        return
    os.makedirs(kol_path, exist_ok=True)
    print(f"Đã tìm thấy {len(img_tags)} ảnh với alt là 'thumbnail-...'")

    # Tải xuống 3 ảnh đầu tiên (nếu có đủ)
    for idx, img in enumerate(img_tags[:5]):
        src = img.get('src')
        if src:
            try:
                print(f"Đang tải ảnh {idx}: {src}")
                img_response = requests.get(src)
                if img_response.status_code == 200:
                    filename = os.path.join(kol_path, f'{idx}-{gender}.jpg')
                    if filename is None:
                        print(f"Không phát hiện thấy mặt trong ảnh {src}. Bỏ qua ảnh này.")
                        continue
                    with open(cropped_ims, 'wb') as f:
                        f.write(img_response.content)
                    print(f"Tải xuống thành công: {filename}")
                else:
                    print(f"Lỗi tải ảnh {src} (status code: {img_response.status_code})")
            except Exception as e:
                print(f"Có lỗi khi tải ảnh {src}: {e}")
        else:
            print("Không tìm thấy URL ảnh trong thẻ.")
        face_detect(filename)
            


if __name__ == '__main__':
    urls = [
        {"url": "https://bookingkols.com.vn/kol/phuong-%C4%91oan-ri-viu-vo1P", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/%C4%91oan-kim-anh-qkzF", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/khoa-pug--6858", "gender": "man"},
        {"url": "https://bookingkols.com.vn/kol/ngoc-eng-ozSV", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/-quan-khong-go-5mTL", "gender": "man"},
        {"url": "https://bookingkols.com.vn/kol/minh-thuy-j6cy", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/-blackii_d-2340", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/trinh-pham-p13d", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/chang-dory-4903", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/tasee_tasee-9810", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/bao-ngoc-oQVb", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/linh-gau-0837", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/nguyen-tuyet-chinh-3582", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/ping-le-LwZv", "gender": "man"},
        {"url": "https://bookingkols.com.vn/kol/pham-ai-thuong-iFe3", "gender": "woman"},
        {"url": "https://bookingkols.com.vn/kol/mc-khanh-bang-review-3122", "gender": "woman"}
    ]

i = 0
for index, item in enumerate(urls):
    i += 1
    get_image(item["url"], i, item["gender"])
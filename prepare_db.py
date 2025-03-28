import os, cv2
from ultralytics import YOLO
import gdown
import zipfile

class PrepareDb:
    def __init__(self, is_get_data=False):
        if is_get_data:
            self.get_data()
        self.filtered_images = "./ims_68"
        try:
            self.face_detection_model = YOLO("yolov11n-face.pt")
            print("Face detection model loaded successfully")
        except Exception as e:
            print(f"Error loading face detection model: {e}")
            print("Loading model from hub...")
            self.face_detection_model = YOLO("yolov8n-face.pt")

    def get_data(self):
        file_id = "16F5Kk4ruyLYTMK20U3UQG8scnCFDyqtA"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "ims_68.zip"
        gdown.download(url, output, quiet=False)
        
        # Tạo folder "ims_68" nếu chưa tồn tại
        folder_name = "ims_68"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(folder_name)
        os.remove(output)

        file_id = "1sdnehh6xciMWlSlvASp-9Y8IA8FuIFEr"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "yolov11n-face.pt"
        gdown.download(url, output, quiet=False)
    
    def face_detection(self, im):
        results = self.face_detection_model(im, conf=0.8)        
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                print("No face detected")
                return None
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

        return face_img

    def split_db_test(self):
        for i in range(5):
            db_dir = f"{self.filtered_images}_ver_{i}/db"
            test_dir = f"{self.filtered_images}_ver_{i}/test"
            os.makedirs(db_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            train_ims_path, test_ims_path = {}, {}
            
            for attendant_name in os.listdir(self.filtered_images):
                train_ims_path[attendant_name] = []
                test_ims_path[attendant_name] = []
                attendant_path = os.path.join(self.filtered_images, attendant_name)
                image_paths = [os.path.join(attendant_path, im) for im in os.listdir(attendant_path)]
                
                if i < len(image_paths):
                    test_ims_path[attendant_name].append(image_paths[i])
                    train_ims_path[attendant_name] = [path for path in image_paths if path != image_paths[i]]
                    
            for attendant_name in train_ims_path:
                attendant_db_path = os.path.join(db_dir, attendant_name)
                os.makedirs(attendant_db_path, exist_ok=True)
                
                for j, im_path in enumerate(train_ims_path[attendant_name]):
                    img = cv2.imread(im_path)
                    if img is None:
                        print(f"Cannot read image: {im_path}")
                        continue
                        
                    face_img = self.face_detection(img)
                    if face_img is not None:
                        # Sử dụng tên file gốc hoặc tạo tên mới với index
                        original_filename = os.path.basename(im_path)
                        # Đảm bảo có phần mở rộng file
                        save_path = os.path.join(attendant_db_path, f"{original_filename}")
                        print(f"Saving: {save_path}")
                        success = cv2.imwrite(save_path, face_img)
                        if not success:
                            print(f"Failed to save image: {save_path}")
                
            for attendant_name in test_ims_path:
                for j, im_path in enumerate(test_ims_path[attendant_name]):
                    img = cv2.imread(im_path)
                    if img is None:
                        print(f"Cannot read test image: {im_path}")
                        continue
                        
                    # Thêm phần mở rộng file .png
                    save_path = os.path.join(test_dir, f"{attendant_name}_{i}.png")
                    print(f"Saving test image: {save_path}")
                    success = cv2.imwrite(save_path, img)
                    if not success:
                        print(f"Failed to save test image: {save_path}")
                        
if __name__ == "__main__":
    prepare_db = PrepareDb(is_get_data=True)
    prepare_db.split_db_test()

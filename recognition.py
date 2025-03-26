# arcFace
# arcFace in DeepFace
# broadface
# ghostfacenet

from abc import ABC, abstractmethod
from compreface import CompreFace
from ultralytics import YOLO
import cv2, os, requests, json
from deepface import DeepFace
from add_db import *
import time

class FaceidRecognition(ABC):
    def __init__(self):
        self.yolo_model = YOLO("yolov11n-face.pt")
        self.correct_predictions = []    # Add a list for correct predictions
        self.incorrect_predictions = []
        self.im_dirs = "./filtered_images"

    def face_dection(self, image_path, output_path):
        im = cv2.imread(image_path)
        results = self.yolo_model(im, conf=0.8)
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                print("No face detected")
                return None
            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Expand the cropping area a bit to get the entire face
                h, w = im.shape[:2]
                margin = int(min((x2 - x1), (y2 - y1)) * 0.3)

                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                face_img = im[y1:y2, x1:x2]
            cv2.imwrite(output_path, face_img)

        return output_path
    
    @abstractmethod
    def faceid_recognition(self, index):
        pass

    def commit_results(self, prediction, person_name, im_name, res):
        if prediction == person_name:
            self.correct_predictions.append({
                "image": im_name,
                "expected": person_name,
                "prediction": prediction,
                "details": res
            })
        else:
            self.incorrect_predictions.append({
                "image": im_name,
                "expected": person_name,
                "prediction": prediction,
                "details": res
            })

    @abstractmethod
    def save_results(self, index):
        pass
        

class CompareFace(FaceidRecognition):
    def __init__(self):
        super().__init__()
        self.domain = "http://localhost"
        self.port = "8000"
        self.api_key = "f74dc9a6-a66f-4b87-9d6e-7cfad3c61bab"
        self.compre_face = CompreFace(self.domain, self.port)
        self.recognition = self.compre_face.init_face_recognition(self.api_key)
        self.collection = self.recognition.get_face_collection()
        self.subjects = self.recognition.get_subjects()    # why not use this?
    
    def save_results(self, index):
        os.makedirs(f"{self.im_dirs}_ver_{index}/result", exist_ok=True)
        with open(f"{self.im_dirs}_ver_{index}/result/compare_correct.json", "w") as f:
            json.dump(self.correct_predictions, f, indent=4)
        with open(f"{self.im_dirs}_ver_{index}/result/compare_incorrect.json", "w") as f:
            json.dump(self.incorrect_predictions, f, indent=4)
    
    def faceid_recognition(self):

        def bulk_add_db(db_dir):
            self.subjects.delete_all()
            add_db = CompareAddDb()
            for celeb_name in os.listdir(db_dir):
                celeb_folder = os.path.join(db_dir, celeb_name)
                for im_name in os.listdir(celeb_folder):
                    im_path = os.path.join(celeb_folder, im_name)
                    add_db.add_db(im_path, celeb_name)

        for index in range(5):
            db_dir = f"{self.im_dirs}_ver_{index}/db"
            test_dir = f"{self.im_dirs}_ver_{index}/test"
            bulk_add_db(db_dir)
            start = time.time()
            for im_name in os.listdir(test_dir):
                person_name = "_".join(im_name.split("_")[:-1])
                im_path = os.path.join(test_dir, im_name)
                try:
                    resized_im_path = self.face_dection(im_path, "resize.jpg")
                    res = self.recognition.recognize(image_path=resized_im_path)
                    print("Res", res)
                except requests.exceptions.JSONDecodeError as e:
                    print("Error recognizing face", e)
                    continue
                if "result" not in res:
                    print("No face detected")
                    continue

                prediction = res["result"][0]["subjects"][0]["subject"]
                self.commit_results(prediction, person_name, im_name, res["result"][0]["subjects"][0])

            self.save_results(index)
            print(f"Time taken compare_face {index}: ", time.time() - start)
        print("Done")

class DeepFaceRecognition(FaceidRecognition):
    def __init__(self):
        super().__init__()

    def save_results(self, index, model):
        os.makedirs(f"{self.im_dirs}_ver_{index}/result", exist_ok=True)
        with open(f"{self.im_dirs}_ver_{index}/result/deepface_{model}_correct.json", "w") as f:
            json.dump(self.correct_predictions, f, indent=4)
        with open(f"{self.im_dirs}_ver_{index}/result/deepface_{model}_incorrect.json", "w") as f:
            json.dump(self.incorrect_predictions, f, indent=4)
    
    def faceid_recognition(self):
        for index in range(5):

            db_compare = f"{self.im_dirs}_ver_{index}/db"
            db_deepface = f"{self.im_dirs}_ver_{index}/deepface_db"
            test_dir = f"{self.im_dirs}_ver_{index}/test"
            os.makedirs(db_deepface, exist_ok=True)

            for attendant_name in os.listdir(db_compare):
                attendant_path = os.path.join(db_compare, attendant_name)
                for im_name in os.listdir(attendant_path):
                    im_path = os.path.join(attendant_path, im_name)
                    cv2.imwrite(os.path.join(db_deepface, attendant_name + "!" + im_name), cv2.imread(im_path))

            models = ["ArcFace", "GhostFaceNet", "Facenet512"]
            
            start = time.time()
            for model in models:
                for im_name in os.listdir(test_dir):
                    person_name = "_".join(im_name.split("_")[:-1])
                    im_path = os.path.join(test_dir, im_name)
                    
                    results = DeepFace.find(img_path=im_path, db_path=db_deepface,
                                            model_name=model, detector_backend="yolov8")
                    results = results[0]
                    
                    if len(results) == 0:
                        prediction = "Unknown"
                        res = 0
                    else:
                        results = results.sort_values(by="distance")
                        top_results = results.head(min(3, len(results)))
                        
                        person_names = []
                        distances = []
                        for identity, distance in zip(top_results["identity"], top_results["distance"]):
                            name = "_".join(os.path.basename(identity).split("!")[:-1])
                            print("identity", identity, "name", name)
                            person_names.append(name)
                            distances.append(distance)
                        
                        print("Person names: ", person_names)
                        if len(person_names) >= 3:
                            from collections import Counter
                            name_counts = Counter(person_names)
                            most_common = name_counts.most_common(1)[0]                    
                            if most_common[1] >= 2:
                                prediction = most_common[0]
                            else:
                                prediction = person_names[0]
                        else:
                            prediction = person_names[0]
                    
                        res = [f"{person}_{distance}" for person, distance in zip(person_names, distances)]
                    self.commit_results(prediction, person_name, im_name, res)
                self.save_results(index, model)
            print(f"Time taken deepface {index}: ", time.time() - start)

if __name__ == '__main__':
    # compare_face = CompareFace()
    # compare_face.faceid_recognition()
    deepface_recognition = DeepFaceRecognition()
    deepface_recognition.faceid_recognition()

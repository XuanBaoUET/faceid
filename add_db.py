from abc import ABC, abstractmethod
from compreface import CompreFace

class AddDB(ABC):
    def __init__(self):
        pass

    def add_db(self):
        pass

class CompareAddDb(AddDB):
    def __init__(self):
        super().__init__()
        self.domain = "http://localhost"
        self.port = "8000"
        self.api_key = "c338e85b-59bf-4267-8461-515c07883fbc"       #Facenet
        self.api_key = "0781cd99-a764-4dbf-a14e-c9b1683a442d"       #mobileNet
        self.compre_face = CompreFace(self.domain, self.port)
        self.recognition = self.compre_face.init_face_recognition(self.api_key)
        self.collection = self.recognition.get_face_collection()
        self.subjects = self.recognition.get_subjects()    # why not use this?
    
    def add_db(self, im_path, attendant_name):
        self.collection.add(image_path=im_path, subject=attendant_name)
        print(f"Added {attendant_name} to database")
    

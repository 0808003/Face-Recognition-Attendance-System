# utils.py
import os
import json
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from pathlib import Path
import pickle

def load_models():
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

def load_embeddings():
    dataset_dir = Path("static/dataset")
    embeddings, ids, student_details = [], [], {}

    for folder in dataset_dir.iterdir():
        if folder.is_dir():
            image_path = folder / "0.jpg"
            meta_path = folder / "meta.json"
            if image_path.exists() and meta_path.exists():
                img = Image.open(image_path)
                mtcnn = MTCNN(image_size=160, margin=0)
                face_tensor = mtcnn(img)
                if face_tensor is not None:
                    resnet = InceptionResnetV1(pretrained='vggface2').eval()
                    emb = resnet(face_tensor.unsqueeze(0))
                    embeddings.append(emb.detach())
                    ids.append(folder.name)
                    with open(meta_path, "r") as f:
                        student_details[folder.name] = json.load(f)

    return embeddings, ids, student_details

def recognize_face(img, mtcnn, resnet, embeddings, ids, details):
    face = mtcnn(img)
    if face is not None:
        emb = resnet(face.unsqueeze(0)).detach()
        dists = [torch.norm(emb - db_emb).item() for db_emb in embeddings]
        min_idx = int(np.argmin(dists))
        pred_id = ids[min_idx]
        prob = 1 - dists[min_idx]

        if prob > 0.5:
            return {
                "status": "success",
                "name": details[pred_id]["name"],
                "id": pred_id,
                "department": details[pred_id]["department"]
            }
        else:
            return {"status": "fail", "message": "You are not registered"}
    return {"status": "fail", "message": "No face detected"}

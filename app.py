# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import torch
import pickle
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from utils import load_models, load_embeddings, recognize_face
from datetime import datetime, timedelta
import os

app = Flask(__name__)

mtcnn, resnet = load_models()
stored_embeddings, stored_ids, student_details = load_embeddings()
recent_attendance = {}  # Store last attendance time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')

        result = recognize_face(img, mtcnn, resnet, stored_embeddings, stored_ids, student_details)

        if result['status'] == 'success':
            student_id = result['id']
            now = datetime.now()
            last_time = recent_attendance.get(student_id)

            if last_time and (now - last_time) < timedelta(seconds=30):
                result['status'] = 'fail'
                result['message'] = 'Your attendance is already marked recently.'
            else:
                with open("attendance_log.csv", "a") as f:
                    f.write(f"{result['name']},{student_id},{now}\n")
                recent_attendance[student_id] = now

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT Render provides, fallback to 5000
    app.run(host='0.0.0.0', port=port)

from flask import Flask, render_template, request
import cv2
import base64
from skimage.metrics import structural_similarity as ssim
import numpy as np

app = Flask(__name__)

def calculate_ssim(image1, image2):
    resized_img1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    gray1 = cv2.cvtColor(resized_img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def compare_images(image1, image2):
    similarity_score = calculate_ssim(image1, image2)
    return similarity_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    image1 = cv2.imdecode(np.fromstring(request.files['image1'].read(), np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.fromstring(request.files['image2'].read(), np.uint8), cv2.IMREAD_COLOR)

    similarity_score = compare_images(image1, image2)
    similarity_percentage = round(similarity_score * 100, 2)  # Convert score to percentage and round to 2 decimal places

    image1_base64 = base64.b64encode(cv2.imencode('.jpg', image1)[1]).decode()
    image2_base64 = base64.b64encode(cv2.imencode('.jpg', image2)[1]).decode()

    return render_template('result.html', similarity_score=similarity_percentage, image1=image1_base64, image2=image2_base64)


if __name__ == '__main__':
    app.run(debug=True)

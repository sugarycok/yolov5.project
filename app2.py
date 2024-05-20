import cv2
from flask import Flask, render_template, request, Response
import torch
import numpy as np
import base64
from PIL import Image

# 모델을 로드하는 함수
def load_model(model_path):
    model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local')  
    return model

# 이미지를 전처리하는 함수
def process_image(image):
    # 이미지 전처리 코드
    return image

# 추론을 수행하는 함수
def infer(model, image):
    with torch.no_grad():  # 그래디언트 계산을 비활성화
        predictions = model(image)  # 모델을 사용하여 추론 수행
    return predictions

# 추론 결과를 처리하는 함수
def process_predictions(frame, predictions):
    class_names = ['Bike', 'Bike_Number_Plate', 'Traffic_Lights']
    bboxes = predictions.xyxy[0].cpu().numpy()  # bounding boxes
    for bbox in bboxes:
        class_index = int(bbox[5])
        class_name = class_names[class_index]
        confidence = bbox[4]
        if confidence > 0.5:  # confidence threshold 설정
            xmin, ymin, xmax, ymax = bbox[:4].astype(int)  # 변경된 부분
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 웹앱 초기화
app = Flask(__name__)

# 홈 페이지
@app.route('/')
def home():
    return render_template('index.html')

# 업로드된 이미지 처리 및 결과 반환
@app.route('/upload', methods=['POST'])
def upload():
    # 이미지 파일 가져오기
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image = np.array(image)[:, :, ::-1].copy()  # BGR -> RGB 변환

    # 모델 로드
    model_path = 'C:/Users/WSU/Desktop/new2/model/best.pt'
    model = load_model(model_path)

    # 이미지 전처리 및 추론
    image = process_image(image)
    predictions = infer(model, image)

    # 이미지에 bounding box 그리기
    process_predictions(image, predictions)

    # 이미지에 bounding box 그리기
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    result_image = f"data:image/jpeg;base64,{jpg_as_text.decode()}"

    # 결과 반환
    return render_template('result.html', result_image=result_image)

# 웹앱 실행
if __name__ == '__main__':
    app.run(debug=True)

import cv2
from flask import Flask, render_template, Response
import torch
import sys

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
def process_predictions(predictions):
    # 추론 결과가 비어 있는지 확인
    if not predictions.pred:
        return "No object detected", 0.0
    
    # 추론 결과 처리 코드
    class_names = ['Bike', 'Bike_Number_Plate', 'Traffic_Lights']
    # 가장 확률이 높은 클래스의 인덱스 찾기
    max_prob_index = predictions.pred[0][:, 5].argmax()
    class_index = int(predictions.pred[0][max_prob_index][5])
    class_name = class_names[class_index]
    probability = predictions.pred[0][max_prob_index][4] * 100  # 확률을 백분율로 변환
    return class_name, probability

# 웹캠에서 이미지를 가져와 처리하는 함수
def gen_frames(model):
    cap = cv2.VideoCapture(0)  # 웹캠 비디오 캡처 시작
    while True:
        success, frame = cap.read()  # 비디오에서 프레임 읽기
        if not success:
            break
        else:
            # 이미지 전처리
            image = process_image(frame)
            # 추론 수행
            predictions = infer(model, image)
            # 추론 결과 처리
            class_name, probability = process_predictions(predictions)
            # 화면에 객체의 클래스 이름과 확률 표시
            cv2.putText(frame, f'{class_name}: {probability:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 이미지 스트림 전송

app = Flask(__name__)

# 홈 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 웹캠으로부터 실시간 이미지 가져오기
@app.route('/video_feed')
def video_feed():
    model_path = 'C:/Users/WSU/Desktop/new2/model/best.pt'
    model = load_model(model_path)
    return Response(gen_frames(model), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    sys.path.insert(0, './model')  # 모델을 로드하기 위한 경로 추가
    app.run(debug=True)

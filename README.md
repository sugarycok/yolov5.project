# 오토바이 감지 커스텀 모델.

이 프로젝트는 제가 제작한 커스텀모델을 이용해 오토바이와 그 오토바이의 번호판을 인식하는 AI를 제작하고 Flask를 이용해 이미지를 업로드해 감지하게 만드는 프로그램을 제작하였습니다.<br>

## 작업 환경(장치 사향)
장치 이름:	DESKTOP-O998J3H<br>
프로세서:	Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz   3.19 GHz<br>
설치된 RAM:	16.0GB<br>
장치 ID:	79976040-AE86-48D2-BE1C-72385C61EAC7<br>
제품 ID:	00330-80000-00000-AA923<br>
시스템 종류:	64비트 운영 체제, x64 기반 프로세서<br>
펜 및 터치:	이 디스플레이에 사용할 수 있는 펜 또는 터치식 입력이 없습니다.<br>

## Windows 사양
에디션:	Windows 10 Pro
버전:	22H2
설치 날짜:	‎2023-‎04-‎12
OS 빌드:	19045.4291
경험:	Windows Feature Experience Pack 1000.19056.1000.0

## 모댈
Yolov5

## 사용한 프로그램
Visual Studio Code, Python 3.9
***
[https://github.com/sugarycok/aiModels.git]를 참고하여 커스텀 모델을 제작하여 자신만의 모델을 만들거나 넷에서 공유중인 모델을 이용해서도 제작하실 수 있습니다.<br>

## 시작 화면
![image](https://github.com/sugarycok/yolov5.project/assets/167059284/bff33204-af1b-491d-a2a3-e67807941857)<br>
프로그램을 시작하면 이런 형식으로 메인페이지가 보일 겁니다.<br>

***

![image](https://github.com/sugarycok/yolov5.project/assets/167059284/fc6f0137-082e-422d-bad1-91f63cc35e7b)
***
![image](https://github.com/sugarycok/yolov5.project/assets/167059284/0c0f8e41-ef7f-44cc-b244-5a91b6fc7518)
***
여기서 파일 선택 버튼으로 업로드 하고자 하는 이미지를 선택하지고 Upload Image 버튼을 눌러주세요<br>

***
![image](https://github.com/sugarycok/yolov5.project/assets/167059284/630d461e-fcbe-4690-a9a8-e195f87fe35a)
***
작업이 완료되면 이러한 형태로 박스가 쳐진 이미지가 제공될 겁니다.<br>

# 다른 모델을 사용하고자 한다면

만약 다른 모델을 이용해서 이 프로그램을 이용하고자 한다면 app2.py 파일에서 몇가지만 수정해 주시면 됩니다.<br>
***
    # 모델 로드
    model_path = 'C:/Users/WSU/Desktop/new2/model/best.pt'
    model = load_model(model_path)
***
이 부분에서 당신이 사용하고자 하는 모델의 경로를 다시 입력해 주시고<br>

def process_predictions(frame, predictions):
    class_names = ['Bike', 'Bike_Number_Plate', 'Traffic_Lights']
    bboxes = predictions.xyxy[0].cpu().numpy()  

이 부분에서 class_name 부분에 들어갈 것들을 여러분이 사용하고자 하는 모델이 감지하는 것들로 바꿔주시면 됩니다.<br>

import cv2
import numpy as np
from keras.models import load_model

# 사용자 정의 모델 로드
model_path = 'y_v2_adam_lr001.h5'  # 저장된 모델의 경로
model = load_model(model_path)

def preprocess_frame(frame, target_size=224):
    """
    이미지를 모델 입력 형식에 맞게 전처리하는 함수.
    학습 시 사용된 전처리 방식과 동일하게 적용합니다.
    """
    orig_h, orig_w = frame.shape[:2]
    print(f"[전처리] 원본 이미지 크기: width={orig_w}, height={orig_h}")
    
    # 종횡비 유지하며 리사이즈
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    print(f"[전처리] 리사이즈 크기: width={new_w}, height={new_h}, scale={scale}")
    resized_frame = cv2.resize(frame, (new_w, new_h))
    
    # 패딩 추가하여 타겟 사이즈 맞추기
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    print(f"[전처리] 패딩 정보: left={pad_left}, right={pad_right}, top={pad_top}, bottom={pad_bottom}")
    
    # 패딩 추가
    image = cv2.copyMakeBorder(resized_frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    
    # BGR에서 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 정규화
    image = image / 255.0
    
    # 배치 차원 추가
    input_frame = np.expand_dims(image, axis=0).astype(np.float32)
    print(f"[전처리] 최종 입력 이미지 형태: {input_frame.shape}")
    
    return input_frame, scale, pad_left, pad_top

# OpenCV로 카메라 열기
cap = cv2.VideoCapture(0)  # 웹캠 사용 (필요에 따라 변경)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

target_size = 224  # 모델의 입력 크기

frame_count = 0  # 프레임 카운터

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("카메라로부터 프레임을 읽을 수 없습니다.")
        break
    
    frame_count += 1
    print(f"\n=== 프레임 {frame_count} ===")
    
    orig_h, orig_w = frame.shape[:2]
    print(f"[프레임] 원본 프레임 크기: width={orig_w}, height={orig_h}")
    
    # 이미지 전처리
    input_frame, scale, pad_left, pad_top = preprocess_frame(frame, target_size=target_size)
    
    # 모델 예측
    try:
        keypoints = model.predict(input_frame)[0]  # (136,)
        print(f"[예측] 키포인트 예측 완료. 예측된 키포인트 수: {len(keypoints)}")
    except Exception as e:
        print(f"[예측 오류] 키포인트 예측 실패: {e}")
        continue

    # 키포인트 배열이 [x1, y1, x2, y2, ..., x68, y68] 형식인지 확인
    if keypoints.shape[0] != 136:
        print(f"[예측 오류] 키포인트의 길이가 예상과 다릅니다: {keypoints.shape[0]}")
        continue

    num_keypoints = keypoints.shape[0] // 2

    # 키포인트를 (68, 2) 형태로 리쉐이프
    keypoints = keypoints.reshape(-1, 2)
    print(f"[예측] 키포인트를 (-1, 2) 형태로 리쉐이프했습니다.")

    # 키포인트를 [0, 224] 범위로 변환
    keypoints = keypoints * target_size  # target_size = 224
    print(f"[예측] 키포인트를 [0, {target_size}] 범위로 변환했습니다.")

    # 키포인트를 원본 이미지 크기로 복원
    keypoints[:, 0] = (keypoints[:, 0] - pad_left) / scale
    keypoints[:, 1] = (keypoints[:, 1] - pad_top) / scale
    print(f"[예측] 키포인트를 원본 이미지 크기로 복원했습니다.")

    # 스케일링된 키포인트를 1D 배열로 변환
    scaled_keypoints = keypoints.flatten()
    print(f"[예측] 키포인트를 평탄화했습니다. 길이: {len(scaled_keypoints)}")

    # 모든 키포인트 그리기
    for idx in range(num_keypoints):
        x = int(scaled_keypoints[idx * 2])
        y = int(scaled_keypoints[idx * 2 + 1])
        if 0 <= x < orig_w and 0 <= y < orig_h:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        else:
            print(f"[그리기 오류] 키포인트 {idx}가 이미지 범위를 벗어났습니다: x={x}, y={y}")

    cv2.imshow('Keypoint Detection', frame)

    # ESC를 눌러 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

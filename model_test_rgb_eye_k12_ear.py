import cv2
import numpy as np
from keras.models import load_model

# 사용자 정의 모델 로드
model_path = 'y_eye_v2_adam.h5'  # 저장된 모델의 경로
model = load_model(model_path)

def preprocess_frame(frame, target_size=224):
    """
    이미지를 모델 입력 형식에 맞게 전처리하는 함수.
    학습 시 사용된 전처리 방식과 동일하게 적용합니다.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 종횡비 유지하며 리사이즈
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    
    # 패딩 추가하여 타겟 사이즈 맞추기
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    # 패딩 추가
    image = cv2.copyMakeBorder(resized_frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    
    # BGR에서 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 정규화
    image = image / 255.0
    
    # 배치 차원 추가
    input_frame = np.expand_dims(image, axis=0).astype(np.float32)
    
    return input_frame, scale, pad_left, pad_top

# 눈 비율(EAR)을 계산하는 함수
def calculate_EAR(eye_indices, keypoints, epsilon=1e-6):
    # 눈 랜드마크 좌표 추출
    coords = keypoints[eye_indices]  # shape (6, 2)
    # EAR 계산
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3]) + epsilon  # 분모가 0이 되는 것을 방지하기 위해 epsilon 추가
    ear = (A + B) / (2.0 * C)
    return ear

# EAR 임계값 및 연속 프레임 깜박임 기준
EAR_THRESHOLD = 0.2

# OpenCV로 카메라 열기
cap = cv2.VideoCapture(0)  # 웹캠 사용 (필요에 따라 변경)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

target_size = 224  # 모델의 입력 크기

frame_count = 0  # 프레임 카운터

# 양쪽 눈 랜드마크 인덱스 정의 (0-based)
right_eye_indices = np.arange(0, 6)   # 0-5: Right Eye
left_eye_indices = np.arange(6, 12)   # 6-11: Left Eye

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("카메라로부터 프레임을 읽을 수 없습니다.")
        break
    
    frame_count += 1
    
    orig_h, orig_w = frame.shape[:2]
    
    # 이미지 전처리
    input_frame, scale, pad_left, pad_top = preprocess_frame(frame, target_size=target_size)
    
    # 모델 예측
    try:
        keypoints = model.predict(input_frame)[0]  # (24,)
    except Exception as e:
        print(f"[예측 오류] 키포인트 예측 실패: {e}")
        continue

    # 키포인트 배열이 [x1, y1, x2, y2, ..., x12, y12] 형식인지 확인
    if keypoints.shape[0] != 24:
        print(f"[예측 오류] 키포인트의 길이가 예상과 다릅니다: {keypoints.shape[0]}")
        continue

    num_keypoints = keypoints.shape[0] // 2

    # 키포인트를 (12, 2) 형태로 리쉐이프
    keypoints = keypoints.reshape(-1, 2)

    # 키포인트를 [0, 224] 범위로 변환
    keypoints = keypoints * target_size  # target_size = 224

    # 키포인트를 원본 이미지 크기로 복원
    keypoints[:, 0] = (keypoints[:, 0] - pad_left) / scale
    keypoints[:, 1] = (keypoints[:, 1] - pad_top) / scale

    # 스케일링된 키포인트를 1D 배열로 변환
    scaled_keypoints = keypoints.flatten()

    # EAR 계산을 위해 키포인트에서 눈 랜드마크 인덱스 선택
    # 이미 eye_keypoints_indices는 [0-5] (Right Eye), [6-11] (Left Eye)
    # calculate_EAR 함수에 맞게 인덱스를 전달
    try:
        right_EAR = calculate_EAR(right_eye_indices, keypoints)
        left_EAR = calculate_EAR(left_eye_indices, keypoints)
        ear = (right_EAR + left_EAR) / 2.0
    except Exception as e:
        print(f"[EAR 계산 오류] {e}")
        continue

    # EAR 값을 프레임 왼쪽 상단에 표시
    cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # EAR 임계값 미만이면 경고 메시지 표시
    if ear < EAR_THRESHOLD:
        cv2.putText(frame, 'WARNING: Eyes Closed!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 눈 주변 키포인트만 그리기
    for idx in range(num_keypoints):
        x = int(keypoints[idx, 0])
        y = int(keypoints[idx, 1])
        if 0 <= x < orig_w and 0 <= y < orig_h:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # 초록색 원으로 표시
        else:
            print(f"[그리기 오류] 눈 키포인트 {idx}가 이미지 범위를 벗어났습니다: x={x}, y={y}")

    cv2.imshow('Keypoint Detection', frame)

    # ESC를 눌러 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

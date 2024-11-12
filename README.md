# model_test

## 파일 이름 규칙 설명(model_test_<데이터셋>_<부위>_k<키포인트수>[_ear].py)
1. model_test
   - 의미: 우리가 만든 모델의 성능을 테스트하기 위한 스크립트임을 나타냅니다.
2. 데이터셋
   - gray: AI Hub에서 제공하는 졸음운전 예방을 위한 운전자 상태 정보 영상 데이터셋
   - rgb: Kaggle에서 제공하는 YouTube Faces With Facial Keypoints 데이터셋
3. 부위
   - face: 얼굴(keypoints)
   - eye: 눈(keypoints)
4. k<키포인트수>
   - 의미: 학습 시 사용된 키포인트의 개수를 나타냅니다.
   - 예: k70은 70개의 키포인트 사용
5. _ear (선택 사항)
   - 의미: 실시간으로 눈 부위의 키포인트를 사용하여 EAR(Eye Aspect Ratio)을 계산하고, EAR 값 및 경고 문구를 노출하는 기능이 포함됨을 나타냅니다.

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import random, urllib.request, pandas as pd, pickle, re
from konlpy.tag import Okt
import matplotlib.pyplot as plt
from dataset_fifth import data

# TensorBoard 
### TensorBoard 로그 저장 디렉토리 설정
log_dir = "logs"

### TensorBoard 콜백 정의
tensorboard_callback = TensorBoard(log_dir=log_dir)


# 데이터 전처리 
### 데이터 섞기
random.shuffle(data)

### duplicated data 제거
seen_values = set()
data = [item for item in data if item[0] not in seen_values and not seen_values.add(item[0])]

### 데이터 분리
symptoms_before_tuning, labels = zip(*data)
print("TOTAL_DATASET: ", len(symptoms_before_tuning))


# 토큰화
stopwords = [',','.','의','로','을','가','이','은','들','는','성','좀','잘','걍','과','고','도','되','되어','되다','를','으로','자','에','와','한','합니다','니다','하다','임','음','환자','응급','상황','상태','증상','증세','구조','구급차','구급','응급환자','구급대','구급대원','구급대원들']
okt = Okt()

### 토크나이저 저장 경로
tokenizer_path = 'tokenizer.pkl'

### stopword 제거, 토큰화
symptoms = []
for sentence in symptoms_before_tuning:
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    symptoms.append(stopwords_removed_sentence)

## 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(symptoms)
encoded_symptoms = tokenizer.texts_to_sequences(symptoms)
word_index = tokenizer.word_index
num_words = len(word_index) + 1


# 패딩
max_length = max(len(seq) for seq in encoded_symptoms)
padded_symptoms = pad_sequences(encoded_symptoms, maxlen=max_length, padding='post')
print("MAX_LEN: ", max_length)
with open("max_length.txt", 'wb') as f:
    f.write(str(max_length).encode())


# 응급 정도 레이블 전처리
num_classes = 5
encoded_labels = to_categorical(np.array(labels) - 1, num_classes=num_classes)


# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(padded_symptoms, encoded_labels, test_size=0.2, random_state=42)


# # 기존 모델 불러오기
# # model = load_model('rnn_codeblue_model.h5')


# RNN 모델 구성 (100차원, 활성화 함수:softmax - 다중 클래스 분류에 사용)
embedding_dim = 100
hidden_unit = 128 # hidden layer: 조정 대상
model = Sequential()
model.add(Embedding(num_words, embedding_dim, input_length=max_length))
model.add(LSTM(hidden_unit))
model.add(Dropout(0.3)) # dropout - 과적합 방지: 조정 대상
model.add(Dense(num_classes, activation='softmax'))


# 학습률 스케줄링 함수 정의 (100번동안은 학습률 유지 후 0.1씩 감소 -> 초기학습은 빠르게)
def lr_scheduler(epoch, lr):
    if epoch < 1000:
        return lr
    else:
        return lr * 0.1 # learning rate: 조정 대상

### 학습률 스케줄링 콜백 정의
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# 조기종료 함수 정의
class CustomEarlyStopping(Callback):
    def __init__(self, accuracy_threshold=0.95, patience=30):
        super(CustomEarlyStopping, self).__init__()
        self.accuracy_threshold = accuracy_threshold
        self.patience = patience
        self.wait = 0  # 개선 없는 횟수 세기
        self.stopped_epoch = 0  # 종료 에포크 번호
        self.best_weights = None  # 최적 가중치 저장
        self.best_val_loss = float('inf')  # 최적의 검증 손실 초기화

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')
        current_val_loss = logs.get('val_loss')

        if current_accuracy >= self.accuracy_threshold and self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print(f"\n조기 종료: 정확도 {self.accuracy_threshold} 이상에 도달하고 {self.patience}번 동안 검증 손실이 개선되지 않았습니다.")
            print(f"{self.patience}번 이전의 모델 가중치로 복원합니다.")
            self.model.set_weights(self.best_weights)

        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_weights = self.model.get_weights()
                self.best_val_loss = current_val_loss
                self.wait = 0
            else:
                self.wait += 1

# 조기 종료 콜백 정의 (10번동안 검증손실이 개선되지 않으면 조기종료)
early_stopping_callback = CustomEarlyStopping(accuracy_threshold=0.95, patience=30)


# 모델 컴파일 (알고리즘:adam, 손실함수:categorical_crossentropy, 평가지표:accuracy)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 학습 (반복횟수:1000, 한번에 처리할 데이터 샘플:32)
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), # epochs: 조정 대상
          callbacks=[early_stopping_callback, lr_scheduler_callback, tensorboard_callback], verbose=1) # mc 추가


# 토크나이저 저장
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)


# 성능 평가
loss, accuracy = model.evaluate(X_test, y_test)
print("테스트 손실:", loss)
print("테스트 정확도:", accuracy)


# 토크나이저 불러오기
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)


# 문장 예측
def emergency_level_prediction(sample_sentence):
    # 샘플 문장 전처리
    sample_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', sample_sentence)
    sample_sentence = okt.morphs(sample_sentence, stem=True) # 토큰화
    sample_sentence = [word for word in sample_sentence if not word in stopwords] # 불용어 제거
    # 샘플 문장을 토큰화하고 패딩
    encoded_sample = tokenizer.texts_to_sequences([sample_sentence])
    padded_sample = pad_sequences(encoded_sample, maxlen=max_length, padding='post')
    # 샘플 문장 응급도 예상
    prediction = model.predict(padded_sample)
    emergency_level = np.argmax(prediction, axis=1) + 1
    confidence = prediction[0][emergency_level[0]-1] # 각 클래스의 확률 중에서 선택된 클래스의 확률
    print(f"응급도: {emergency_level[0]}, 확신도: {confidence * 100.0}%")


# 예시 문장
emergency_level_prediction("지금 환자는 아주 위험한 무호흡 상태입니다.") # 1
emergency_level_prediction("지금 환자가 패혈증으로 인해 고통을 호소하고 있습니다.") # 2
emergency_level_prediction("절단으로 인한 출혈.") # 3
emergency_level_prediction("환자는 국소성 염증으로 인해 구급차 탑승") # 4
emergency_level_prediction("감기와 장염 증상이 복합적으로 일어나고 있음.") # 5

# 모델 저장
model.save('rnn_model_v4_no_cum.h5')
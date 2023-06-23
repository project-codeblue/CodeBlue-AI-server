from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pickle, re
from konlpy.tag import Okt


app = Flask(__name__)
okt = Okt()


# 전처리용 상수
MAX_LEN = 16
stopwords = [',','.','의','로','을','가','이','은','들','는','성','좀','잘','걍','과','고','도','되','되어','되다','를','으로','자','에','와','한','합니다','입니다','있습니다','니다','하다','임','음','환자','응급','상황','상태','증상','증세','구조']


@app.route('/ai', methods=['GET'])
def getEmergencyLevel():
    # body data 받기
    sentence_received = request.args.get('sentence')
    print(sentence_received)

    # 기존 모델 불러오기
    model = load_model('rnn_model_v4.h5')

    # 토크나이저 불러오기
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # 문장 예측
    def emergency_level_prediction(sample_sentence):
        # 샘플 문장 전처리
        sample_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', sample_sentence)
        sample_sentence = okt.morphs(sample_sentence, stem=True) # 토큰화
        sample_sentence = [word for word in sample_sentence if not word in stopwords] # 불용어 제거
        # 샘플 문장을 토큰화하고 패딩
        encoded_sample = tokenizer.texts_to_sequences([sample_sentence])
        padded_sample = pad_sequences(encoded_sample, maxlen=MAX_LEN, padding='post')
        # 샘플 문장 응급도 예상
        prediction = model.predict(padded_sample)
        emergency_level = np.argmax(prediction, axis=1) + 1
        confidence = prediction[0][emergency_level[0]-1] # 각 클래스의 확률 중에서 선택된 클래스의 확률
        print(f"응급도: {emergency_level[0]}, 확신도: {confidence * 100.0}%")
        return emergency_level[0]

    emergency_level = emergency_level_prediction(sentence_received)
    print(emergency_level)
    return jsonify({'result':'success', 'emergency_level': int(emergency_level)})


if __name__ == '__main__':  
    app.run('0.0.0.0', port=5000, debug=True)
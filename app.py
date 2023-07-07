from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pickle, re
from konlpy.tag import Okt


app = Flask(__name__)
okt = Okt()


# constants for preprocessing
MAX_LEN = 18
stopwords = [',','.','의','로','을','가','이','은','들','는','성','좀','잘','걍','과','고','도','되','되어','되다','를','으로','자','에','와','한','합니다','입니다','있습니다','니다','하다','임','음','환자','응급','상황','상태','증상','증세','구조']


@app.route('/ai', methods=['GET'])
def getEmergencyLevel():
    # get sentence in query string from url
    sentence_received = request.args.get('sentence')
    print(sentence_received)

    # load pretrained model
    model = load_model('rnn_model_v4_no_cum.h5')

    # load tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # function for predicting emergency level
    def emergency_level_prediction(sample_sentence):
        # sample sentence preprocessing
        sample_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', sample_sentence)
        sample_sentence = okt.morphs(sample_sentence, stem=True) # tokenizing
        sample_sentence = [word for word in sample_sentence if not word in stopwords] # removing stopwords
        # sample sentence integer encoding and padding
        encoded_sample = tokenizer.texts_to_sequences([sample_sentence])
        padded_sample = pad_sequences(encoded_sample, maxlen=MAX_LEN, padding='post')
        # sample sentence prediction
        prediction = model.predict(padded_sample)
        emergency_level = np.argmax(prediction, axis=1) + 1
        confidence = prediction[0][emergency_level[0]-1] 
        print(f"Emergency Level: {emergency_level[0]}, Confidence Rate: {confidence * 100.0}%")
        return emergency_level[0]

    emergency_level = emergency_level_prediction(sentence_received)
    print(emergency_level)
    return jsonify({'result':'success', 'emergency_level': int(emergency_level)})


if __name__ == '__main__':  
    app.run('0.0.0.0', port=5000, debug=True)
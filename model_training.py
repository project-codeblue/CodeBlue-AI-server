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
### For saving log in TensorBoard and set directory
log_dir = "logs"

### TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir)


# Data Preprocessing
### shuffle data for the random insertion
random.shuffle(data)

### remove duplicated data
seen_values = set()
data = [item for item in data if item[0] not in seen_values and not seen_values.add(item[0])]

### separate data into sentence and label (symptom_level)
symptoms_before_tuning, labels = zip(*data)
print("TOTAL_DATASET: ", len(symptoms_before_tuning))


# Tokenizing
stopwords = [',','.','의','로','을','가','이','은','들','는','성','좀','잘','걍','과','고','도','되','되어','되다','를','으로','자','에','와','한','합니다','니다','하다','임','음','환자','응급','상황','상태','증상','증세','구조','구급차','구급','응급환자','구급대','구급대원','구급대원들']
okt = Okt()

### path for saving tokenizer
tokenizer_path = 'tokenizer.pkl'

### removing stopword, tokenizing
symptoms = []
for sentence in symptoms_before_tuning:
    tokenized_sentence = okt.morphs(sentence, stem=True) # tokenizing
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    symptoms.append(stopwords_removed_sentence)

## Integer Encoding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(symptoms)
encoded_symptoms = tokenizer.texts_to_sequences(symptoms)
word_index = tokenizer.word_index
num_words = len(word_index) + 1


# Padding
max_length = max(len(seq) for seq in encoded_symptoms)
padded_symptoms = pad_sequences(encoded_symptoms, maxlen=max_length, padding='post')
print("MAX_LEN: ", max_length)
with open("max_length.txt", 'wb') as f:
    f.write(str(max_length).encode())


# Symptom Level Label Preprocessing
num_classes = 5
encoded_labels = to_categorical(np.array(labels) - 1, num_classes=num_classes)


# Separate data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(padded_symptoms, encoded_labels, test_size=0.2, random_state=42)


# define RNN model (100 dimension, Activation Function: Softmax - Used for multi-class classification)
embedding_dim = 100
hidden_unit = 128 # hidden layer
model = Sequential()
model.add(Embedding(num_words, embedding_dim, input_length=max_length))
model.add(LSTM(hidden_unit))
model.add(Dropout(0.3)) # dropout - prevent overfitting
model.add(Dense(num_classes, activation='softmax'))


# define learning rate scheduling function: Keep the learning rate unchanged for the first 100 epochs, then decrease it by 0.1 incrementally
def lr_scheduler(epoch, lr):
    if epoch < 1000:
        return lr
    else:
        return lr * 0.1 # learning rate

### define learning rate scheduling callback
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# define early stopping function
class CustomEarlyStopping(Callback):
    def __init__(self, accuracy_threshold=0.95, patience=30):
        super(CustomEarlyStopping, self).__init__()
        self.accuracy_threshold = accuracy_threshold
        self.patience = patience
        self.wait = 0  # improvement counter
        self.stopped_epoch = 0  # stopped epoch number
        self.best_weights = None  # saving optimal weights 
        self.best_val_loss = float('inf')  # initialize optimal validation loss

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')
        current_val_loss = logs.get('val_loss')

        if current_accuracy >= self.accuracy_threshold and self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print(f"\nEarly Stopping: Accuracy {self.accuracy_threshold} has reached, but no more improvement in loss for {self.patience} times.")
            print(f"The model weights are restored to the weights of the model before {self.patience} times.")
            self.model.set_weights(self.best_weights)

        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_weights = self.model.get_weights()
                self.best_val_loss = current_val_loss
                self.wait = 0
            else:
                self.wait += 1

# define early stopping callback (terminates the training if the validation loss does not improve for 30 consecutive epochs)
early_stopping_callback = CustomEarlyStopping(accuracy_threshold=0.95, patience=30)


# compile model (optimizer: adam algorithm, loss function: categorical_crossentropy, evaluation metric: accuracy)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# trainig (iteration: 1000 epochs, batch size: 32 samples per iteration)
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), # epochs: 조정 대상
          callbacks=[early_stopping_callback, lr_scheduler_callback, tensorboard_callback], verbose=1) # mc 추가


# save tokenizer
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)


# evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# load tokenizer for prediction
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)


# symptom level prediction
def emergency_level_prediction(sample_sentence):
    # sample sentence preprocessing
    sample_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', sample_sentence)
    sample_sentence = okt.morphs(sample_sentence, stem=True) # tokenizing
    sample_sentence = [word for word in sample_sentence if not word in stopwords] # removing stopwords
    # sample sentence encoding
    encoded_sample = tokenizer.texts_to_sequences([sample_sentence])
    padded_sample = pad_sequences(encoded_sample, maxlen=max_length, padding='post')
    # prediction
    prediction = model.predict(padded_sample)
    emergency_level = np.argmax(prediction, axis=1) + 1
    confidence = prediction[0][emergency_level[0]-1] 
    print(f"Emergency Level: {emergency_level[0]}, Confidence Rate: {confidence * 100.0}%")


# sample sentences
emergency_level_prediction("지금 환자는 아주 위험한 무호흡 상태입니다.") # 1
emergency_level_prediction("지금 환자가 패혈증으로 인해 고통을 호소하고 있습니다.") # 2
emergency_level_prediction("절단으로 인한 출혈.") # 3
emergency_level_prediction("환자는 국소성 염증으로 인해 구급차 탑승") # 4
emergency_level_prediction("감기와 장염 증상이 복합적으로 일어나고 있음.") # 5

# save model
model.save('rnn_model_v4_no_cum.h5')
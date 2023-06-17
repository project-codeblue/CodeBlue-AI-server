const tf = require("@tensorflow/tfjs");
const fs = require("fs");

// 전처리용 상수
const MAX_LEN = 18;
const STOPWORDS = [
  ",",
  ".",
  "의",
  "로",
  "을",
  "가",
  "이",
  "은",
  "들",
  "는",
  "성",
  "좀",
  "잘",
  "걍",
  "과",
  "고",
  "도",
  "되",
  "되어",
  "되다",
  "를",
  "으로",
  "자",
  "에",
  "와",
  "한",
  "합니다",
  "입니다",
  "있습니다",
  "니다",
  "하다",
  "임",
  "음",
  "환자",
  "응급",
  "상황",
  "상태",
  "증상",
  "증세",
  "구조",
];

// 기존 모델 및 토크나이저 불러오기
const modelPath =
  "file://C:/Users/siwon/Desktop/Voyage99/projects/CodeBlue-AI-server/predict_emergency_level_model/model.json";
const tokenizerPath =
  "C:/Users/siwon/Desktop/Voyage99/projects/CodeBlue-AI-server/tokenizer.json";

async function loadModel(modelPath) {
  const model = await tf.loadLayersModel(modelPath);
  return model;
}

// 파일 로드
let tokenizer;
fs.readFile(tokenizerPath, "utf8", (err, data) => {
  if (err) {
    console.error("파일을 읽을 수 없습니다:", err);
    return;
  }

  // 로드된 데이터 사용
  tokenizerData = JSON.parse(data);
  // TODO: tokenizerData를 사용하여 원하는 작업을 수행하세요.
});

// 문장 예측
async function emergencyLevelPrediction(sampleSentence) {
  // 모델 불러오기
  const model = await loadModel(modelPath);

  // 샘플 문장 전처리 (토큰화, 불용어 제거)
  let sampleSentenceArr = sampleSentence.split(" ");
  sampleSentenceArr = sampleSentenceArr.filter(
    (word) => !STOPWORDS.includes(word)
  );
  // 샘플 문장을 토큰화하고 패딩
  const encodedSample = sampleSentenceArr.map(
    (word) => tokenizer.word_index[word] || 0
  );
  const paddedSample = padSequences([encodedSample], MAX_LEN, "post");
  // 샘플 문장 응급도 예상
  const prediction = model.predict(tf.tensor2d(paddedSample));
  let emergencyLevel;
  let confidence;

  if (!Array.isArray(prediction)) {
    emergencyLevel = prediction.argMax(-1).dataSync()[0] + 1;
    confidence = prediction.dataSync()[emergencyLevel - 1];
  }
  console.log(`응급도: ${emergencyLevel}, 확신도: ${confidence * 100.0}%`);
}

function padSequences(
  sequences,
  maxLen,
  padding = "post",
  truncating = "post",
  value = 0
) {
  return sequences.map((seq) => {
    // truncate
    if (seq.length > maxLen) {
      if (truncating === "pre") {
        seq.splice(0, seq.length - maxLen);
      } else {
        seq.splice(maxLen, seq.length - maxLen);
      }
    }
    // pad
    if (seq.length < maxLen) {
      const pad = [];
      for (let i = 0; i < maxLen - seq.length; i++) {
        pad.push(value);
      }
      if (padding === "pre") {
        seq = pad.concat(seq);
      } else {
        seq = seq.concat(pad);
      }
    }
    return seq;
  });
}

// 예시 문장
emergencyLevelPrediction(
  "응급환자는 심장마비로 인해 의식을 잃고 쓰러졌습니다. 호흡 곤란 상태입니다."
); // 예상: 1

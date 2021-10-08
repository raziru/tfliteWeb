const MAX_SEQUENCE_LENGTH = 113;
const getKey = (obj,val) => Object.keys(obj).find(key => obj[key] === val); // For getting tags by tagid


async function start() {
  const textarea = document.querySelector("textarea");
  const resultDiv = document.querySelector(".result");
  // Load the TFLite model.
  const model = await tfTask.NLClassification.CustomModel.TFLite.load({
    model:
    "https://storage.googleapis.com/tfweb/models/movie_review_sentiment_classification.tflite"
    //"https://storage.googleapis.com/tensorflow_mine/mymodelLSTM.tflite"
  });

  document.querySelector(".btn").addEventListener("click", async () => {
    // Get the classification result for the entered text
    var temp=textarea.value;
    const result = await model.predict(temp);
    // Show the results.
    resultDiv.textContent = result.classes
      .map((c) => `${c.className}: ${c.score.toFixed(3)}`)
      .join(", ");
  });
}
start();

function word_preprocessor(word) {
  word = word.replace(/[-|.|,|\?|\!]+/g, '');
  word = word.replace(/\d+/g, '1');
  word = word.toLowerCase();
  if (word != '') {
    return word;
  } else {
    return '.'
  }
};

function make_sequences(words_array) {
  let sequence = Array();
  words_array.slice(0, MAX_SEQUENCE_LENGTH).forEach(function(word) {
    word = word_preprocessor(word);
    let id = words_vocab[word];
    if (id == undefined) {
      sequence.push(words_vocab['<UNK>']);
    } else {
      sequence.push(id);
    }  
  });

  // pad sequence
  if (sequence.length < MAX_SEQUENCE_LENGTH) {
    let pad_array = Array(MAX_SEQUENCE_LENGTH - sequence.length);
    pad_array.fill(words_vocab['<UNK>']);
    sequence = sequence.concat(pad_array);
  }

  return sequence;
};


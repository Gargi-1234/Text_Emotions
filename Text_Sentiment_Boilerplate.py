import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


sentence = ["I am happy to meet my friends. We are planning to go a party.", 
            "I had a bad day at school. i got hurt while playing football"]

vocab_size = 10000
embedding_dim = 16
oov_tok = "<OOV>"
training_size = 20000

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentence)

#Create a word_index dictionary

word_index = tokenizer.word_index
sequence = tokenizer.texts_to_sequences(sentence)



padding_type='post'
max_length = 100
trunc_type='post'


training_padded = pad_sequences(sentence, maxlen=max_length,
                                padding=padding_type, truncating=trunc_type)

print(training_padded[0:2])

model = tensorflow.keras.models.load_model('Text_Emotion.h5')
result = model.predict(training_padded)
print(result)
predict_class = np.argmax(result,axis=1)
predict_class

# Create a word_index dictionary

# Padding the sequence

# Define the model using .h5 file

# Test the model

# Print the result


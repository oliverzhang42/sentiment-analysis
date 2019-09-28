# The preprocessing is from the tutorial here:
# https://www.kaggle.com/c/word2vec-nlp-tutorial/overview/part-1-for-beginners-bag-of-words

from bs4 import BeautifulSoup
import gensim
import pandas as pd
import re
import numpy as np
import nltk.data
#nltk.download()

# Download punkt tokenizer for sentence splitting
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

def review_to_words(raw):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw).get_text()

    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. Join the words into a sentence separated by space.
    return (" ".join(words))


def review_to_sentences(review, tokenizer):
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    # 2. Loops over each sentence.
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence isn't empty, apply review_to_wordlist.
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence))

    return sentences

train = pd.read_csv("labeledTrainData.tsv", sep='\t')
test = pd.read_csv("testData.tsv", sep='\t')

train_labels = train['sentiment']
np.save("train_labels", train_labels)

clean_train = []
clean_test = []

for i in range(len(train["review"])):
    cleaned = review_to_words(train["review"][i])
    clean_train.append(cleaned)

for i in range(len(test["review"])):
    cleaned = review_to_words(test["review"][i])
    clean_test.append(cleaned)

#######################    Word to Vec    #############################

def truncate_and_pad_vecs(dataset, length, verbose=True):
    # Truncate and pad vectors to length=length.

    vecs = []

    for i in range(len(dataset)):
        if i % 1000 == 0 and verbose:
            print(i)

        vec = []
        if len(dataset[i]) > length:
            # Truncate each movie review to 200 words
            for j in range(length):
                word = dataset[i][j]
                if word in model.wv:
                    vec.append(model.wv[word])
                else:
                    vec.append(np.zeros((300,)))
            vec = np.array(vec)
        else:
            pad = np.zeros((length-len(dataset[i]), 300))
            for j in range(len(dataset[i])):
                word = dataset[i][j]
                if word in model.wv:
                    vec.append(model.wv[word])
                else:
                    vec.append(np.zeros((300,)))
            vec = np.array(vec)
            vec = np.concatenate((pad, vec), axis=0)

        vecs.append(vec)
        del vec

    return np.array(vecs, dtype='float32')


model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

train_vec = truncate_and_pad_vecs(clean_train, 200)
np.save("train_vectors", train_vec)
del train_vec
test_vec = truncate_and_pad_vecs(clean_test, 200)
np.save("test_vectors", test_vec)



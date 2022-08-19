# -*- coding: utf-8 -*-
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from SentenceGetter import SentenceGetter
from WordFeatureGetter import WordFeatureGetter
from crf_n_folder_ensemble import crf_n_folder_ensemble

word_feature_getter = WordFeatureGetter()

crf_n_folder_ensemble = crf_n_folder_ensemble("data", "final_model") #path should be given as constructor input

df, classes = crf_n_folder_ensemble.prepare_data(divide_data=True, num_splits=3)

print("Started getting the sentences")
sentence_getter = SentenceGetter(df)
sentences = sentence_getter.sentences
print("Finished getting the sentences")


def word2features(sent, i):
    word = str(sent[i][0])
    postag = str(sent[i][1])

    features = {
        'bias': 1.0,
        'word.islower()': word.islower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'capital_ratio(word)': word_feature_getter.capital_ratio(word),
        'number_of_vowels(word)': word_feature_getter.number_of_vowels(word),
        'vowel_ratio(word)': word_feature_getter.vowel_ratio(word),
        'word.isdigit()': word.isdigit(),
        'digit_count(word)': word_feature_getter.digit_count(word),
        'digit_ratio(word)': word_feature_getter.digit_ratio(word),
        'word.istitle()': word.istitle(),
        'len(word)': len(word),
        'word.isupper()': word.isupper(),
        'word.number_of_non_alphanumeric(word)': word_feature_getter.number_of_non_alphanumeric(word),
        'postag': postag,
        'postag[:2]': postag[:2],
        'IsWordContainsApostrophe': word_feature_getter.isContainsApostrophe(word),
        'IsinPrePersonList': word_feature_getter.isInPrePersonWords(word),  #
        'IsinCityList': word_feature_getter.isInCityList(word),
        'IsMahkeme': word_feature_getter.isMahkeme(word),  #
        'IsPunctioation: ': word_feature_getter.isPunctuation(word),
        'IsDaireNumarasi': word_feature_getter.daireNumarasi(word)  # 13. 15. gibi
        #TODO dictionary class
        #TODO regex class
    }

    if i > 0:
        word1 = str(sent[i - 1][0])
        postag1 = str(sent[i - 1][1])
        
        features.update({
            '-1:word.islower()': word1.islower(),
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            'capital_ratio(-1:word)': word_feature_getter.capital_ratio(word1),
            'number_of_vowels(-1:word)': word_feature_getter.number_of_vowels(word1),
            'vowel_ratio(-1:word)': word_feature_getter.vowel_ratio(word1),
            '-1:word.isdigit()': word1.isdigit(),
            'digit_count(-1:word)': word_feature_getter.digit_count(word1),
            'digit_ratio(-1:word)': word_feature_getter.digit_ratio(word1),
            '-1:word.istitle()': word1.istitle(),
            'len(-1:word)': len(word1),
            '-1:word.isupper()': word1.isupper(),
            'number_of_non_alphanumeric(-1:word)': word_feature_getter.number_of_non_alphanumeric(word1),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:IsinPrePersonList': word_feature_getter.isInPrePersonWords(word1),
            '-1:IsPunctuation': word_feature_getter.isPunctuation(word1),
            '-1:IsWordContainsApostrophe': word_feature_getter.isContainsApostrophe(word1),
            '-1:IsDaireNumarasi': word_feature_getter.daireNumarasi(word1),
            '-1:IsMahkeme': word_feature_getter.isMahkeme(word1),  #
            '-1:IsinCityList': word_feature_getter.isInCityList(word1)  #
            #TODO Is in dictionary
        })

    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word2 = str(sent[i + 1][0])
        postag2 = str(sent[i + 1][1])

        features.update({
            '+1:word.lower()': word2.islower(),
            '+1:word[-3:]': word2[-3:],
            '+1:word[-2:]': word2[-2:],
            'capital_ratio(+1:word)': word_feature_getter.capital_ratio(word2),
            'number_of_vowels(+1:word)': word_feature_getter.number_of_vowels(word2),
            'vowel_ratio(+1:word)': word_feature_getter.vowel_ratio(word2),
            '+1:word.isdigit()': word2.isdigit(),
            'digit_count(+1:word)': word_feature_getter.digit_count(word2),
            'digit_ratio(+1:word)': word_feature_getter.digit_ratio(word2),
            '+1:word.istitle()': word2.istitle(),
            'len(+1:word)': len(word2),
            '+1:word.isupper()': word2.isupper(),
            'number_of_non_alphanumeric(+1:word)': word_feature_getter.number_of_non_alphanumeric(word2),
            '+1:postag': postag2,
            '+1:postag[:2]': postag2[:2],
            '+1:IsInCityList': word_feature_getter.isInCityList(word2),
            '+1:IsMahkeme': word_feature_getter.isMahkeme(word2),
            '+1:IsInPrePersonList': word_feature_getter.isInPrePersonWords(word2),
            '+1:IsPunctuation': word_feature_getter.isPunctuation(word2),
            '+1:IsWordContainsApostrophe': word_feature_getter.isContainsApostrophe(word2),
            '+1:IsDaireNumarasi': word_feature_getter.daireNumarasi(word2)
            #TODO Is in dictionary
        })

    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


print("Starting Feature Extraction")
X = (sent2features(s) for s in sentences)
y = [sent2labels(s) for s in sentences]
print("Finished Feature Extraction")

print("Started Train Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("Finished Train Test Split")


print("Started Training Train Data Model")
crf_n_folder_ensemble.fit(X_train, y_train)

print("Test on Test Set")
y_pred = crf_n_folder_ensemble.predict(X_test)

print("Train on Test Results:")
print(metrics.flat_classification_report(y_test, y_pred, labels=classes))

crf_n_folder_ensemble.save_model()

import re
import string


class WordFeatureGetter():
    
    pre_person_word_file = open('Feature_Data/prePersonWordList.txt', 'r', encoding='UTF-8')
    pre_person_word_list = pre_person_word_file.read()
    pre_person_word_list = pre_person_word_list.split()
    word_In_City_file = open('Feature_Data/word_In_City_List.txt' , 'r', encoding='UTF-8')
    word_In_City_List = word_In_City_file.read()
    word_In_City_List = word_In_City_List.split(' ')

    def capital_ratio(self, word):
        capital_count = sum(1 for c in word if c.isupper())
        return capital_count / len(word)

    def non_alphanumeric_ratio(self, word):
        return self.number_of_nonAlphanumeric(word) / len(word)

    def number_of_non_alphanumeric(self, word):
        return sum(1 for c in word if c.isalnum())

    def number_of_vowels(self, word):
        pattern = re.compile("[AEIİOÖUÜaeıioöuü]")
        return sum(1 for c in word if pattern.match(c))

    def vowel_ratio(self, word):
        return self.number_of_vowels(word) / len(word)

    def digit_count(self, word):
        return sum(1 for c in word if c.isdigit())

    def digit_ratio(self, word):
        return self.digit_count(word) / len(word)

    def isInPrePersonWords(self,word):        # word[i-1]
        if word.title() in self.pre_person_word_list:
            return 1
        else:
            return 0

    def isInCityList(self,word):     # word[i+1]
        if word.title() in self.word_In_City_List:
            return 1
        else:
            return 0

    def isMahkeme(self,word):   # word[i+1]

        if 'Bölge' in word or 'bölge' in word:
            return 1

        elif re.match(r'[0-9]+\.', word):
            return 1
        return 0

    def isContainsApostrophe(self, word):

        if "'" in word:
            return 1
        return 0

    def isPunctuation(self, word):

        if word in string.punctuation:
            return 1
        return 0

    def daireNumarasi(self, word):

        try:
            result = re.search(r'[0-9]+\.', word).group(0)
        except AttributeError:
            return 0

        if result is None:
            return 0
        else:
            return 1
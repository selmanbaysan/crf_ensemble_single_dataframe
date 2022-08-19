import re
import string


class threeDotNER_CRF:

    prePersonWordList1 = open('Feature_Data/prePersonWordList.txt', 'r', encoding='UTF-8')
    prePersonWordList1 = prePersonWordList1.read()
    word_In_City_List = open('Feature_Data/word_In_City_List.txt' , 'r', encoding='UTF-8')
    word_In_City_List = word_In_City_List.read()
    word_In_City_List.split(' ')

    def prePersonWordList(self,word):        # word[i-1]
        self.prePersonWordList1.split(' ')
        if word in self.prePersonWordList1:
            return 1
        else:
            return 0

    def isInCityList(self,word):     # word[i+1]

        if word in self.word_In_City_List:
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





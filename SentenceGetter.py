class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
    

        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['word'].values.tolist(),
                                                           s['pos'].values.tolist(),
                                                           s['tag'].values.tolist())]
        
        self.grouped = self.data.groupby('sentid').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
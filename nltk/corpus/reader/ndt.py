from nltk.parse import DependencyGraph
from nltk.corpus.reader.api import *

class NdtCorpusReader(CorpusReader):

    # Column types:
    # token index, word form, lemma, coarse-grained part-of-speech (POS)tag,
    # fine-grained POS tag, morphological features, index of head, dependency relation
    # and optimized POS tag
    
    TOKEN_INDEX = []    #: column for token index
    WORD_FORM = []      #: column for word forms
    LEMMA = []          #: column for lemmas
    COARSE_POS = []     #: column for coarse-grained part-of-speech (POS)tag
    FINE_POS = []       #: column for fine-grained POS tag
    MORPH_FEAT = []     #: column for morphological features
    HEAD_INDEX = []     #: column for index of head
    DEP_REL = []        #: column for dependency relation
    OPT_TAGS = []       #: column for optimized tags
    

    #Stores start and end ID of each file.
    DOC_POS = {}

  

    def __init__(self, root, fileids, encoding='utf8'):     
        CorpusReader.__init__(self, root, fileids, encoding)
        self.fillColumn()
        self.getOpt()
        
    def fillColumn(self):
        for f in sorted(self._fileids):
            t = self.open(f)
            self.DOC_POS[f] = [len(self.TOKEN_INDEX)]
            for line in t:
                if line != '\n':
                    loot = line.split('\t')
                    self.TOKEN_INDEX.append(int(loot[0]))
                    self.WORD_FORM.append(loot[1])
                    self.LEMMA.append(loot[2])
                    self.COARSE_POS.append(loot[3])
                    #self.FINE_POS.append(loot[4]) # same as coarse
                    self.MORPH_FEAT.append(loot[5])
                    self.HEAD_INDEX.append(int(loot[6]))
                    self.DEP_REL.append(loot[7])
            self.DOC_POS[f].append(len(self.TOKEN_INDEX))

    def getOpt(self):
        f = self.open("optimized.tags")
        for line in f:
            self.OPT_TAGS.append(line[:-1])
        


    #Raw output of ndt
    def raw(self, fileids=None):
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, compat.string_types):
            fileids = [fileids]
        return concat([self.open(f).read() for f in fileids])


    #README file
    def readme(self):
        """
        Return the contents of the corpus README file
        """
        return self.open("README").read()

    # returns token indexes from start to end, or from selected file ids.
    def token_indexes(self, fileids=None):
        if fileids is None:
            return self.TOKEN_INDEX
        elif isinstance(fileids, compat.string_types):
            fileids = [fileids]
        retrn = []
        for i in fileids:
            retrn = retrn + self.TOKEN_INDEX[self.DOC_POS[i][0]:self.DOC_POS[i][1]]
        return retrn
        
    # returns wordss from start to end, or from selected file ids.
    def words(self, fileids=None):
        if fileids is None:
            return self.WORD_FORM
        elif isinstance(fileids, compat.string_types):
            fileids = [fileids]
        retrn = []
        for i in fileids:
            retrn = retrn + self.WORD_FORM[self.DOC_POS[i][0]:self.DOC_POS[i][1]]
        return retrn

    # returns lemmas from start to end, or from selected file ids.   
    def lemmas(self, fileids=None):
        if fileids is None:
            return self.LEMMA
        elif isinstance(fileids, compat.string_types):
            fileids = [fileids]
        retrn = []
        for i in fileids:
            retrn = retrn + self.LEMMA[self.DOC_POS[i][0]:self.DOC_POS[i][1]]
        return retrn

    # returns pos tags from start to end, or from selected file ids. 
    def pos(self, fileids=None):
        if fileids is None:
            return self.COARSE_POS
        elif isinstance(fileids, compat.string_types):
            fileids = [fileids]
        retrn = []
        for i in fileids:
            retrn = retrn + self.COARSE_POS[self.DOC_POS[i][0]:self.DOC_POS[i][1]]
        return retrn

    # returns morphological features from start to end, or from selected file ids.
    def morp_feats(self, fileids=None):
        if fileids is None:
            return self.MORPH_FEAT
        elif isinstance(fileids, compat.string_types):
            fileids = [fileids]
        retrn = []
        for i in fileids:
            retrn = retrn + self.MORPH_FEAT[self.DOC_POS[i][0]:self.DOC_POS[i][1]]
        return retrn
        
    # returns head indexes from start to end, or from selected file ids.
    def head_indexes(self, fileids=None):
        if fileids is None:
            return self.HEAD_INDEX
        elif isinstance(fileids, compat.string_types):
            fileids = [fileids]
        retrn = []
        for i in fileids:
            retrn = retrn + self.HEAD_INDEX[self.DOC_POS[i][0]:self.DOC_POS[i][1]]
        return retrn

    # returns dependency relations from start to end, or from selected file ids.
    def dep_rels(self, fileids=None):
        if fileids is None:
            return self.DEP_REL
        elif isinstance(fileids, compat.string_types):
            fileids = [fileids]
        retrn = []
        for i in fileids:
            retrn = retrn + self.DEP_REL[self.DOC_POS[i][0]:self.DOC_POS[i][1]]
        return retrn
    
    # returns optimized tags from start to end, or from selected file ids.
    def opt_pos(self, fileids=None):
        if fileids is None:
            return self.OPT_TAGS
        elif isinstance(fileids, compat.string_types):
            fileids = [fileids]
        retrn = []
        for i in fileids:
            retrn = retrn + self.OPT_TAGS[self.DOC_POS[i][0]:self.DOC_POS[i][1]]
        return retrn
     
    # returns sentences from start to end, or from selected file ids.
    def sents(self, fileids=None):
        return self.extract(fileids=fileids)

    # returns tagged words from start to end, or from selected file ids.
    # There is also two tagset, None and 'opt'.
    def tagged_words(self, tagset=None, fileids=None):
        if tagset == None:
            return self.extract(sent=False, tags=True, fileids=fileids)
        elif tagset == 'opt':
            return self.extract(sent=False, opt=True, fileids=fileids)
    
   # returns tagged sentences from start to end, or from selected file ids.
   # There is also two tagset, None and 'opt'
    def tagged_sents(self, tagset=None, fileids=None):
        if tagset==None:
            return self.extract(tags=True, fileids=fileids)
        elif tagset == 'opt':
            return self.extract(opt=True, fileids=fileids)

    #helper method for extracting wanted data.
    def get_data(self, id=0, index=False, word=False, lma=False, tags=False, morph=False, head=False, dep=False, opt=False):
        uniq = []
        if index:
            uniq.append(self.TOKEN_INDEX[id])
        if word:
            uniq.append(self.WORD_FORM[id])
        if lma:
            uniq.append(self.LEMMA[id])
        if tags:
            uniq.append(self.COARSE_POS[id])
        if opt:
            uniq.append(self.OPT_TAGS[id])
        if morph:
            uniq.append(self.MORPH_FEAT[id])
        if head:
            uniq.append(self.HEAD_INDEX[id])
        if dep:
            uniq.append(self.DEP_REL[id])
        if len(uniq) > 1:
            return tuple(uniq)
        elif len(uniq) > 0:
            return(uniq[0])


    # Method for extracting wanted info from corpus.
    # Returnes words or sentences, from start to end, or from selected file ids.
    # sent  | True, get sentences  | False, get only words
    # word  | True, get word form  | False, not include word form
    # tags  | True, get pos tags   | False, not include pos tag
    # lma   | True, get lemmas     | False, not include lemmas
    # morph | True, get morph feat | False, not include morphological feature
    # head  | True, get head index | False, not include head index
    # dep   | True, get dep rel    | False, not include dependency relation
    # opt   | Ture, get opt tags   | False, not include optimized tags

    def extract(self, index=False, sent=True, word=True, lma=False, tags=False, morph=False, head=False, dep=False, opt=False, fileids=None):
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, compat.string_types):
            fileids = [fileids]

        doclist = []
        for i in fileids:
            doclist = doclist + list(range(self.DOC_POS[i][0],self.DOC_POS[i][1]))

        doclist = sorted(doclist)
            
        if sent:
            all_sents = []
            cur_sent = []
            x = 0
            for i in doclist:
                if self.TOKEN_INDEX[i] < x:
                    all_sents.append(cur_sent)
                    cur_sent = []
                    cur_sent.append(self.get_data(i, index, word, lma, tags, morph, head, dep, opt))
                    x = self.TOKEN_INDEX[i]
                else:
                    x = self.TOKEN_INDEX[i]
                    cur_sent.append(self.get_data(i, index, word, lma, tags, morph, head, dep, opt))                  
            if len(cur_sent) > 1:
                all_sents.append(cur_sent)
            return all_sents
        else:
            all_words = []
            for i in doclist:
                all_words.append(self.get_data(i, index, word, lma, tags, morph, head, dep, opt))
            return all_words


    def lemma_word(self, fileids=None):
        return self.extract(sent=False, lma=True, fileids=fileids)
    
    def lemma_sent(self, fileids=None):
        return self.extract(lma=True, fileids=fileids)

    def morph_word(self, fileids=None):
        return self.extract(sent=False, morph=True, fileids=fileids)
    
    def morph_sent(self, fileids=None):
        return self.extract(morph=True, fileids=fileids)


    #only 3 cell parsere, as 10 cell requires modification to the corpus.
    def parsed_sents(self, tagset=None, fileids=None):
        return self.parsed_sents_3cells(tagset, fileids=fileids)
                    

    def parsed_sents_3cells(self, tagset=None, fileids=None):
        sentences = []
        if tagset == 'opt':
            sentences = self.extract(opt=True, head=True, fileids=fileids)
        else:
            sentences = self.extract(tags=True, head=True, fileids=fileids)
        restruct = []
        for j in sentences:
            str = ""
            for i in j:
                str += '{0}\t{1}\t{2}\n'.format(i[0], i[1], i[2])
            restruct.append(str)
        return [DependencyGraph(s) for s in restruct]

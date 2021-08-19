import numpy as np
import math 
import pandas as pd
from collections import Counter
from tqdm import trange, tqdm
import spacy 

from src.extractor.thresholders import thresholders
from src.utils.assistant import query_masker
import json


from src.importance_metrics.salient_scorers import linguist

from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

np.random.seed(25)

nlp = spacy.load('en_core_web_sm')

# taken from https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
# Accessed 23 Jul 2020
class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""

        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag

                if token.pos_ in candidate_pos and token.is_stop is False:
                    
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)

            sentences.append(selected_words)

        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1

        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        
        return dict(node_weight)
        
        
    def analyze(self, text, 
                candidate_pos=None, 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences

        if candidate_pos is not None:
        
            sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        else:
            
            sentences = [doc.text.split()]
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight

    # added to make it more functionable for our purposes
    def return_scored_sequence(self, text, tokenizer, candidates, window_size = 8, lower = False):
        self.node_weight  = None

        add_after = False

        if tokenizer.cls_token in text.split():
        
            add_after = True
            or_length = len(text.split())

            if tokenizer.pad_token in text.split():

                ind = text.split().index(tokenizer.pad_token)

            else:

                ind = len(text.split())

            text = " ".join(text.split()[1:ind-1])

        self.analyze(text, candidate_pos = candidates, 
                    window_size = window_size, lower = lower)

        keywords = self.get_keywords(len(text))

        scored_sequence = np.asarray([keywords[w] if w in keywords.keys() else 0 for w in text.split()])

        if add_after:
            
            scored_sequence = [float(0)] + list(scored_sequence)

            diff = or_length - len(scored_sequence)

            scored_sequence += [float(0)]*diff

            scored_sequence = [float(x) for x in scored_sequence]

        return list(scored_sequence)

import logging 

def text_ranker(data, tokenizer = None, extract_rationales = False):
    
    import config.cfg
    from config.cfg import AttrDict

    with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
        args = AttrDict(json.load(f))
        
    """ 
    text-rank scorer
    returns the text-rank scores for words in the vocabulary  
    from the training set
    """

    tr4w = TextRank4Keyword()


    if extract_rationales:
        
        data["text__"] = data.text
    
    else:
        
        data["text__"] = data.text.apply(lambda x : " ".join(tokenizer.convert_ids_to_tokens(x["input_ids"])))

    logging.info("calculating text rank scores")
    
    tqdm.pandas(desc = "calculating text-rank scores", position = 0, leave = True)

    data["salient_scores"] = data.text__.progress_apply(
        lambda x : tr4w.return_scored_sequence(x, candidates = args["linguistic_feature"], tokenizer = tokenizer),
    )

    logging.info("extracted textrank rationales")

    return {"scored_data":data.drop(columns = "text__")}
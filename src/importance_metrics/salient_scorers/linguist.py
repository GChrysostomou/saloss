#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import spacy

def reduce_to_pos(doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for token in doc:
            # Store words only with cadidate POS tag
            if token.pos_ in candidate_pos:# and token.is_stop is False:
                if lower is True:
                    sentences.append(token.text.lower())
                else:
                    sentences.append(token.text)

        if len(sentences) == 0:
            
            sentences = doc.text.split()

        return sentences

def ent_piece(x, feature):

    """
    accepts spacy loaded text
    and returns entity list
    """

    tokens = [tok for tok in x]

    if feature == "ner":

        word_entity_pairs = [x.text + "_" + x.ent_type_ for x in tokens]
    
    elif feature == "pos-limiter":

        word_entity_pairs = reduce_to_pos(x, ['ADJ', 'ADV', 'VERB','AUX'], lower = True)

    else:

        word_entity_pairs = [x.text  + "_" + x.pos_ for x in tokens]

    return " ".join(word_entity_pairs)

def tagger(data, feature = "pos", only_tags = False, column_to_allocate = "tagger"):


    accepted_features = ["pos", "ner", "pos-limiter"]

    assert feature in accepted_features

    nlp = spacy.load("en_core_web_sm")

    data[column_to_allocate] = data.text.apply(lambda x : nlp(x))

    data[column_to_allocate] = data[column_to_allocate].apply(lambda x : ent_piece(x, feature))

    if only_tags:

        data[column_to_allocate] = data[column_to_allocate].apply(lambda x : [w.split("_")[1] for w in x.split()])

    return data


class collective_tagger:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def stop_word_tagger(self,data, column_to_allocate = "stopword"):

        from nltk.corpus import stopwords
        
        stop_word_list = stopwords.words('english') + ["[PAD]", "[CLS]", "[SEP]", "[UNKN]"]

        data[column_to_allocate] = data.text.apply(lambda x: [0 if z in stop_word_list else 1 for z in x.split()])

        return data

    def ner_tagger(self, data, column_to_allocate = "ner"):

        data = tagger(data, feature = "ner", only_tags = True, column_to_allocate= column_to_allocate)

        import pdb; pdb.set_trace();

        data[column_to_allocate] = data[column_to_allocate].apply(lambda x : [1. if z != "" else 0. for z in x ])

        return data

    def pos_tagger(self, data, column_to_allocate = "pos"):

        data = tagger(data, feature = "pos", only_tags = True, column_to_allocate= column_to_allocate)

        return data
        
    def return_all(self, data):
        
        data = data.rename(columns = {"text":"text__"})

        data["text"] = data.text__.apply(lambda x : " ".join(self.tokenizer.convert_ids_to_tokens(x["input_ids"])))

        data = self.stop_word_tagger(data)
        # data = self.ner_tagger(data)
        
        data = data.drop(columns = "text")
        data = data.rename(columns = {"text__":"text"})

        return data

    def sort_tokenizer(self, text):

        pass

        return data

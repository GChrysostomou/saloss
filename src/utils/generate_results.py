import pandas as pd
import glob
import numpy as np
from transformers import AutoTokenizer
import spacy
from tqdm import tqdm
import sys, os
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import torch



def encode_it(tokenizer, max_length, *arguments):

    """
    returns token type ids, padded doc and 
    """

    if len(arguments) > 1:

        dic = tokenizer.encode_plus(arguments[0], arguments[1],
                                        add_special_tokens = True,
                                        max_length = max_length,
                                        padding = 'max_length',
                                        return_token_type_ids = True,
                                        truncation = True)

    else:
  
        dic = tokenizer.encode_plus(arguments[0],
                                        add_special_tokens = True,
                                        max_length = max_length,
                                        padding = 'max_length',
                                        return_token_type_ids = True,
                                        truncation = True)
       
    return dic

def wpiece2word(tokenizer, sentence, weights, print_err = False):  

    """
    converts word-piece ids to words and
    importance scores/weights for word-pieces to importance scores/weights
    for words by aggregating them
    """

    tokens = tokenizer.convert_ids_to_tokens(sentence)

    new_words = {}
    new_score = {}

    position = 0

    for i in range(len(tokens)):

        word = tokens[i]
        score = weights[i].clone().detach().data

        if "##" not in word:
            
            position += 1
            new_words[position] = word
            new_score[position] = score
            
        else:
            
            new_words[position] += word.split("##")[1]
            new_score[position] += score

    return np.asarray(list(new_words.values())), torch.tensor(list(new_score.values()))
    

def _pos_analysis_(data_dir : str = "datasets", dataset : str = "sst", 
                  importance_metric :str = "scaled attention", salience_scorer : str = "textrank",
                    extracted_rat_dir : str = "extracted_rationales",
                  ):
    
    query = False

    if dataset == "evinf" or dataset == "multirc":

        query = True
        
    test = pd.read_csv(f"{data_dir}/{dataset}/data/test.csv")

    van_imp_scores = np.load(f"{extracted_rat_dir}/{dataset}/data/importance_scores/test-importance_scores.npy", allow_pickle=True).flatten()[0]
    text_imp_scores = np.load(f"{extracted_rat_dir}/{dataset}/data/importance_scores/test-{salience_scorer}_importance_scores.npy", allow_pickle=True).flatten()[0]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if query:

        max_len = round(max([len(x.split()) for x in test.document.values])) + \
                    max([len(x.split()) for x in test["query"].values])
        max_len = round(max_len)

    else:

        max_len = round(max([len(x.split()) for x in test.text.values]))
    
    
    max_len = min(max_len, 512)

    # load the pretrained tokenizer
    pretrained_weights = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

    if query:
        test["text"] = test.apply(lambda x: encode_it(tokenizer, 
                        max_len, x["document"], x["query"]), axis = 1)

    else:

        test["text"] = test.apply(lambda x: encode_it(tokenizer, 
                        max_len, x["text"]), axis = 1)

    test["text"] = test["text"].apply(lambda x: x["input_ids"])
    test = test[["annotation_id", "text"]]
    test["vanilla_scores"] = test.annotation_id.apply(lambda x: van_imp_scores[x][importance_metric])
    test["textrank_scores"] = test.annotation_id.apply(lambda x: text_imp_scores[x][importance_metric])
    test = test.to_dict("records")
    
    
    nlp = spacy.load("en_core_web_sm")

    new_data = []
    strings = []
    ## get word scores
    with tqdm(total=len(test), file=sys.stdout, desc = "assigning word importance scores") as pbar:

        for item in test:

            van_scores = torch.softmax(torch.tensor(item["vanilla_scores"]), -1)
            sal_scores = torch.softmax(torch.tensor(item["textrank_scores"]), -1)
            text = item["text"][:len(van_scores)]

            if query:

                if tokenizer.sep_token_id in text:

                    indx = text.index(tokenizer.sep_token_id)

                    sal_scores = sal_scores[:indx]
                    van_scores = van_scores[:indx]
                    text = text[:indx]

            words, van_scores = wpiece2word(tokenizer, text, van_scores)

            doc = nlp(" ".join(words))

            _ , sal_scores = wpiece2word(tokenizer, text, sal_scores)


            new_data.append({
                "annotation_id": item["annotation_id"],
                "pos_tags" : [token.pos_ for token in doc], 
                "vanilla_word_scores": van_scores,
                "textrank_word_scores": sal_scores,
                "sequence" : words
                })

            pbar.update(1)

            strings.append(" ".join(words))
            
    ## lets filter out some high freq low freq words
    tfidf = TfidfVectorizer(min_df=5, max_df = 0.8, stop_words='english')

    tfidf.fit(strings)
    
    vocab = tfidf.vocabulary_

    vanilla_scores = {}
    sal_scores = {}
    counts = {}

    for test_instance in new_data:

        van = list(zip(test_instance["pos_tags"], test_instance["vanilla_word_scores"].numpy(), test_instance["sequence"]))
        sal = list(zip(test_instance["pos_tags"], test_instance["textrank_word_scores"].numpy(), test_instance["sequence"]))

        for _i, pair in enumerate(van):

            # filter out
            if pair[2] in vocab:

                if pair[0] in vanilla_scores:

                    vanilla_scores[pair[0]] += pair[1]

                else:

                    vanilla_scores[pair[0]] = pair[1]

                if pair[0] in sal_scores:

                    sal_scores[pair[0]] += sal[_i][1]

                else:

                    sal_scores[pair[0]] = sal[_i][1]

                if pair[0] in counts:

                    counts[pair[0]] += 1

                else:

                    counts[pair[0]] = 1

            else:
                pass
    
    ## plot
    vanilla_scores = {k:v/counts[k] for k,v in vanilla_scores.items()}
    sal_scores = {k:v/counts[k] for k,v in sal_scores.items()}
    
    scores = pd.DataFrame({"Baseline": vanilla_scores, "SaLoss": sal_scores})
    
    fig, ax = plt.subplots()

    scores.plot.bar(figsize = (14,8), ax = ax)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.legend(fontsize = 20)

    fname = os.path.join(
        os.getcwd(),
        "plots_and_tabs",
        dataset, 
        ""
    )

    os.makedirs(fname, exist_ok=True)

    plt.savefig(f"{fname}{dataset}-{importance_metric}.png", dpi = 300, bbox_inches = 0 )
    
    print("*** saved fig for pos!")

    return


def _get_frac_of_(eval_dir : str = "evaluation_results",
                saliency_scorer : str = "textrank",
                data_split : str = "test"):

    all_datasets = {
        "baseline" : {},
        "saloss" : {}  
    }

    for dataset in ["sst", "evinf", "multirc", "semeval", "agnews"]:  


        fname = f"{eval_dir}/{dataset}/{data_split}-fraction-of-summary.json"

        with open(fname, "r") as file : base_data = json.load(file)

        fname = f"{eval_dir}/{dataset}/{saliency_scorer}_{data_split}-fraction-of-summary.json"

        with open(fname, "r") as file : sal_data = json.load(file)

        del sal_data["random"]

        all_datasets["baseline"][dataset] = base_data
        all_datasets["saloss"][dataset] = sal_data
        
    base = pd.DataFrame(all_datasets["baseline"]).round(2)    
    sal = pd.DataFrame(all_datasets["saloss"]).round(2)

    base["feat_attr"] = base.index
    sal["feat_attr"] = sal.index

    base["index"] = "baseline"
    sal["index"] = "saloss"

    base.index = base["index"]
    sal.index = sal["index"]

    del base["index"]
    del sal["index"]

    df = pd.concat([base,sal ], axis = 0)

    fname = os.path.join(
        os.getcwd(),
        "plots_and_tabs",
        ""
    )

    os.makedirs(fname, exist_ok=True)

    df.to_csv(f"{fname}{data_split}-frac_off.csv")

    print("**** saved frac of!")
    
    return


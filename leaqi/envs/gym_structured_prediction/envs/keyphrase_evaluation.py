import argparse
import sys
import launch
import numpy as np
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import sys
sys.path.append('/mnt/kiante-workspace/peers/position-rank')
from position_rank import position_rank
from tokenizer import StanfordCoreNlpTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
args = parser.parse_args()

# Semeval-2017 https://github.com/UKPLab/semeval2017-scienceie/blob/master/code/reader.py
# https://scienceie.github.io/resources.html

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def main():
    str  = f' Did you start the standord java parser?\n'
    str += f' if not run the following command in the stanford-corenlp-full-2018-10-05 foler\n'
    str += f'run: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &\n'
    str += f' --------------------\n'
    print(str)
    tokenizer = StanfordCoreNlpTokenizer("http://localhost", port = 9000)
    name = 'train'

    with open(f'keyphrase_dataset_files/{name}_semaeval_keyphrase.txt', 'r') as the_file:
        entries = the_file.read().strip().split("\n\n")
        entries_idx = list(range(len(entries)))

        sents, tags_li = [], [] # list of lists
        org_sents, org_tags = [], []
        doc_start = []
        doc_value = False
        for idx in entries_idx:
            entry = entries[idx]
            if entry.startswith('-DOCSTART'):
                doc_value=True
                continue

            words = [line.split()[0].lower() for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(words)
            tags_li.append(tags)
            doc_start.append(doc_value*1.0)
            doc_value = False
            w_tags = []

        for sent_idx in range(len(sents)):
            sent = sents[sent_idx]
            sent_str = ' '.join(sent)
            unsupervise_phrases = position_rank(sent_str, tokenizer)
            print(f'unsupervise_phrases: {unsupervise_phrases}')

            tag_idxs = []
            if unsupervise_phrases is not None:
                for phrase_idx in range(len(unsupervise_phrases)):
                        phrase = unsupervise_phrases[phrase_idx].split('_')
                        tag_matches =  find_sub_list(phrase, sent)
                        for tag_idx in tag_matches:
                            if tag_idx:
                                tag_idxs += list(range(tag_idx[0], tag_idx[1]+1))

            idx_to_tag = ['O' for _ in range(len(sent))]
            for idx in tag_idxs:
                idx_to_tag[idx] = 'I-MISC'
            w_tags.append(idx_to_tag)

        y_true = [x for y in tags_li for x in y]
        y_pred = [x for y in w_tags for x in y]
        sent_lens = [len(tags_li[y]) for y in range(len(tags_li))]
        print(f' f1: {f1_score(y_true, y_pred)}')
        print(f'recall: {recall_score(y_true, y_pred)}')
        print(f'precisio: {precision_score(y_true, y_pred)}')
        print(f'f1 score: {f1_score(y_true, y_pred)}')
        print(f'num_sents: {len(sents)}')
        print(f'avg lens: {np.mean(sent_lens)}')
        print(f'num tags: {len(y_pred)}')

if __name__== "__main__":
    main()

from nltk.tokenize import word_tokenize, sent_tokenize
import nltk.data
import re
import numpy as np
import sys
import codecs
import os
import sys
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
import string
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

# Dataset and basic code from this script comes from this website:
# https://scienceie.github.io/resources.html
# - Train: (https://drive.google.com/open?id=0B2Z1kbILu3YtYjkwMHd3TmNPWDQ)
# - Test: (https://drive.google.com/open?id=0B2Z1kbILu3YtMUlfaWZDN0FSUms)
# - Foundation of this script: (https://drive.google.com/file/d/0B2Z1kbILu3YtNDl3bVNQVnVjeDg/view)

# Please specify the location of the position rank github package
loaction_of_postion_rank_package = '/mnt/kiante-workspace/peers/position-rank'
sys.path.append(loaction_of_postion_rank_package)
from position_rank import position_rank
from tokenizer import StanfordCoreNlpTokenizer

tokenizer = StanfordCoreNlpTokenizer("http://localhost", port = 9000)

def merge(lower, upper, anns_to_idx, anns, idx_to_anns):
    anns[lower] += anns[upper]
    anns_to_idx[lower].update(anns_to_idx[upper])
    idx_to_anns[lower].update(idx_to_anns[upper])

    for idx in range(upper, max(sorted(list(anns.keys()))) + 1):
        if idx in sorted(list(anns.keys())):
            anns[idx] = anns[idx+1]
            anns_to_idx[idx] = anns_to_idx[idx+1]
            idx_to_anns[idx] = idx_to_anns[idx+1]
    print(' ----- merged ----- ')

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)

    l = [x.translate(str.maketrans('', '', string.punctuation)).encode('utf-8') for x in l]
    sl = [x.encode('utf-8') for x in sl]

    for ind in (i for i,e in enumerate(l) if e[:1]==sl[0][:1] or sl[0] in e):
        if l[ind:ind+sll]==sl:
            results.append({'contiguous':(ind,ind+sll)})

        if len(sl) == 1:
             if sl[0] in l[ind]:
                 results.append({'discontiguous': ([ind])})
        elif len(sl) > 1:
            for offset in range(1, sll):
                if l[ind:ind+offset] + l[ind+offset+1:ind+offset+1+(sll-offset)] == sl:
                    results.append({'contiguous':(ind,ind+offset)})
                    results.append({'contiguous':(ind+offset+1,ind+offset+1+(sll-offset))})
    return results

# Function modified from: (https://scienceie.github.io/resources.html)
def readAnn(*, textfolder, name):
    '''
    Read .ann files and look up corresponding spans in .txt files
    :param textfolder:
    :return:
    '''

    w_tags_total = []
    y_tags_total = []
    sent_lens = []

    nltk_detokenizer = lambda x: "".join([" "+i if not i.startswith("'") and\
             i not in string.punctuation else i for i in x]).strip()

    with open(f'keyphrase_dataset_files/{name}_semaeval_keyphrase.txt', 'w') as the_file:
        flist = os.listdir(textfolder)
        for f in flist:
            if not f.endswith(".ann"):
                continue
            if '._S' in f:
                continue

            f_anno = open(os.path.join(textfolder, f), "rU")
            f_text = open(os.path.join(textfolder, f.replace(".ann", ".txt")), "rU")

            for l in f_text:
                text = l

            sentences = sent_tokenize(text)

            for idx in range(len(sentences)):
                sentences[idx] = sentences[idx].replace(u'\xa0', u' ').lower().replace("_", "-")

            assert(len(" ". join(sentences)))
            anns = defaultdict(lambda: [])
            anns_to_idx = defaultdict(lambda: {})
            idx_to_anns = defaultdict(lambda: {})

            for l in f_anno:
                anno_inst = l.strip("\n").split("\t")
                if len(anno_inst) == 3:
                    anno_inst1 = anno_inst[1].split(" ")
                    if len(anno_inst1) == 3:
                        keytype, start, end = anno_inst1
                    else:
                        keytype, start, _, end = anno_inst1
                    if not keytype.endswith("-of"):

                        # look up span in text and print error message if it doesn't match the .ann span text
                        keyphr_text_lookup = text[int(start):int(end)].replace(u'\xa0', u' ').replace("_", "-")
                        start = int(start)
                        end = int(end)
                        keyphr_ann = anno_inst[2]
                        if keyphr_text_lookup != keyphr_ann:
                            print("Spans don't match for anno " + l.strip() + " in file " + f)

                        found_sentence = False
                        keyphr_text_lookup = keyphr_text_lookup.lower()
                        for idx, sent in enumerate(sentences,start=0):
                            if keyphr_text_lookup in sentences[idx]:
                                anns[idx].append(keyphr_text_lookup)
                                anns_to_idx[idx][keyphr_text_lookup] = (start, end)
                                idx_to_anns[idx][tuple((start, end))] = keyphr_text_lookup
                                found_sentence = True

                        if not found_sentence:
                            found_sentence = False
                            keyphr_shorten = " ".join(keyphr_text_lookup.split()[:-1])
                            for idx, sent in enumerate(sentences,start=0):
                                if keyphr_shorten in sentences[idx]:
                                    if keyphr_text_lookup in "".join(sentences[idx:idx+2]):
                                        sentences[idx:idx+2] = ["".join(sentences[idx:idx+2])]
                                        anns[idx].append(keyphr_text_lookup)
                                        anns_to_idx[idx][keyphr_text_lookup] = (start, end)
                                        idx_to_anns[idx][tuple((start, end))] = keyphr_text_lookup
                                        merge(idx, idx+1, anns_to_idx, anns, idx_to_anns)
                                        found_sentence = True
                                        break
                                    elif keyphr_text_lookup in " ".join(sentences[idx:idx+2]):
                                        sentences[idx:idx+2] = [" ".join(sentences[idx:idx+2])]
                                        anns[idx].append(keyphr_text_lookup)
                                        anns_to_idx[idx][keyphr_text_lookup] = (start, end)
                                        idx_to_anns[idx][tuple((start, end))] = keyphr_text_lookup
                                        merge(idx, idx+1, anns_to_idx, anns, idx_to_anns)
                                        found_sentence = True
                                        break

                            if not found_sentence:
                                raise Exception(f"Did not find keyphrase: {keyphr_text_lookup} sentence")

            for sent_idx, sent in enumerate(sentences, start=0):
                the_file.write(f'-DOCSTART- -X- -X- O')
                the_file.write(f'\n\n')
                prefix = sentences[:sent_idx]
                words = word_tokenize(sent)
                sent_lens.append(len(sent))

                tag_idxs = []
                sent_without_puncts = "".join([tok for tok in sent if tok not in string.punctuation])
                unsupervise_phrases = position_rank(sent_without_puncts, tokenizer, lang='en')

                if unsupervise_phrases is not None:
                    for phrase_idx in range(len(unsupervise_phrases)):
                            phrase = unsupervise_phrases[phrase_idx].lower().split('_')
                            tag_matches =  find_sub_list(phrase, words)
                            for matches in tag_matches:
                                if 'contiguous' in matches:
                                    tag_idx = matches['contiguous']
                                    tag_idxs += list(range(tag_idx[0], tag_idx[1]))
                                elif 'discontiguous' in matches:
                                    tag_idx = matches['discontiguous']
                                    tag_idxs.append(tag_idx[0])

                w_tags = ['O' for _ in range(len(sent))]
                for idx in tag_idxs:
                    w_tags[idx] = 'I-MISC'

                for word_idx, word in enumerate(words):
                    tag = 'O'
                    for (start, end) in list(anns_to_idx[sent_idx].values()):
                        detokenizer = nltk_detokenizer(prefix + words[:word_idx + 1])
                        if start <= len(detokenizer) <= end:
                            tag = 'I-MISC'
                            break
                    y_tags_total.append(tag)
                    w_tags_total.append(w_tags[word_idx])
                    the_file.write(f'{word} {w_tags[word_idx]} {tag} \n')
                the_file.write(f'\n')

    y_true = y_tags_total
    y_pred = w_tags_total
    print(f' f1: {f1_score(y_true, y_pred)}')
    print(f'recall: {recall_score(y_true, y_pred)}')
    print(f'precisio: {precision_score(y_true, y_pred)}')
    print(f'f1 score: {f1_score(y_true, y_pred)}')
    print(f'num_sents: {len(sent_lens)}')
    print(f'avg sent lenth: {np.mean(sent_lens)}')
    print(f'num tags: {len(y_pred)}')

if __name__== "__main__":
    readAnn(textfolder='keyphrase_dataset_files/scienceie2017_train/train2', name='train')
    readAnn(textfolder='keyphrase_dataset_files/semeval_articles_test/', name='test')

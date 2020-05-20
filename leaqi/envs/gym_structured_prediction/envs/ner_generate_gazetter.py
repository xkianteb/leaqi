import multiprocessing as mp
import random
import numpy as np
import nltk
from nltk.corpus import stopwords
import csv
import gzip
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.util import ngrams
from flashtext import KeywordProcessor
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import os

default_path_to_gazetter = '/mnt/kiante-workspace/datasets/gazetteer/gazetteers/'

def openfile(filename=None, filepath=None, num_char=0):
    with gzip.open(filepath+filename, 'rb') as f:
        file = f.read().splitlines()
        func_len = lambda x: " ".join(str(x,"utf-8").lower().split()).split(' ')
        func_str = lambda x: str(x,"utf-8").lower()
        return [func_str(x) for x in file if len(func_len(x)) > num_char]

def get_label(entry):
    sent = entry[1][0]
    str_sent = " ".join([word.lower() for word in sent])
    tags = entry[1][1]

    def values_to_dict(list_, tag):
        list_ = [y for x in list_ for y in x.split()]
        return dict(zip(list_, [tag]*len(list_)))

    # People, Location, Organization, Miscellanous Tags
    people_tag = values_to_dict(people_pattern.extract_keywords(str_sent), 'I-PER')
    location_tag = values_to_dict(location_pattern.extract_keywords(str_sent), 'I-LOC')
    org_tag = values_to_dict(organizations_pattern.extract_keywords(str_sent), 'I-ORG')
    misc_tag = values_to_dict(misc_pattern.extract_keywords(str_sent), 'I-MISC')
    all_tags = {**misc_tag,**location_tag, **org_tag, **people_tag}

    pred_tags = ['O'] * len(sent)
    for idx, word in enumerate(sent):
        word = word.lower()
        #if word in all_tags:
        #    pred_tags[idx] = all_tags[word]
        if word in people_tag:
            pred_tags[idx] = 'I-PER'
        elif word in location_tag:
            pred_tags[idx] = 'I-LOC'
        elif word in org_tag:
            pred_tags[idx] = 'I-ORG'
        elif word in misc_tag:
            pred_tags[idx] = 'I-MISC'
    return pred_tags

def main():
    os.system('mkdir -p ner_dataset_files/')
    def crete_dataset(name=''):
        with open(f'ner_dataset_files/{name}_gazetter.txt', 'w') as the_file:
            #the_file.write('Hello\n')
            try:
                conll_ = f'ner_dataset_files/{name}_conll.txt'
            except:
                raise Exception('Original conll file does not exist')
            entries = open(conll_, 'r').read().strip().split("\n\n")
            entries_idx = list(range(len(entries)))

            sents, tags_li = [], [] # list of lists
            org_sents, org_tags = [], []
            doc_start = []
            doc_value = False
            for idx in entries_idx:
                entry = entries[idx]
                if entry.startswith('-DOCSTART'):
                    #the_file.write(f'{entry}')
                    doc_value=True
                    continue

                words = [line.split()[0] for line in entry.splitlines()]
                tags = ([line.split()[-1] for line in entry.splitlines()])
                sents.append(words)
                tags_li.append(tags)
                doc_start.append(doc_value*1.0)
                doc_value = False

            corpus = {id: (sents[id], tags_li[id]) for id in range(len(sents))}

            with mp.Pool() as pool:
                labels = pool.map(get_label, corpus.items())

            y_true = [x for y in tags_li for x in y]
            y_pred = [x for y in labels for x in y]

            sent_lens = [len(tags_li[y]) for y in range(len(tags_li))]

            print(f'{name} ----- results')
            print(f'recall: {recall_score(y_true, y_pred)}')
            print(f'precisio: {precision_score(y_true, y_pred)}')
            print(f'f1 score: {f1_score(y_true, y_pred)}')
            print(f'avg lens: {np.mean(sent_lens)}')
            print(f'num tags: {len(y_pred)}')
            print(classification_report(y_true, y_pred, digits=2))

            # Write Results to File
            for x in range(len(sents)):
                if doc_start[x]:
                    the_file.write(f'-DOCSTART- -X- -X- O')
                    the_file.write(f'\n\n')

                for word_, gazz_label_, org_label_ in zip(sents[x], labels[x], tags_li[x]):
                    the_file.write(f'{word_} {gazz_label_} {org_label_}\n')
                the_file.write(f'\n')
                #tmp = [f"{word_} {label_}\n" for word_, label_ in zip(sents[x], labels[x])]
                #print('HERE')
                #the_file.write(f'

            #np.save(f'{name}_gazetteer.sents', sents)
            #np.save(f'{name}_gazetteer.tags', labels)

    crete_dataset(name='train')
    crete_dataset(name='test')
    crete_dataset(name='valid')

if __name__== "__main__":
  people = set(openfile(filename='People.gz',num_char=1, filepath=default_path_to_gazetter))
  people_pattern = KeywordProcessor()
  _ = [people_pattern.add_keyword(word) for word in people]

  locations = set(openfile(filename='Locations.Countries.gz',filepath=default_path_to_gazetter))
  location_pattern = KeywordProcessor()
  _ = [location_pattern.add_keyword(loc) for loc in locations]

  organizations = set(openfile(filename='Corporations.Organizations.Terrorist.gz', num_char=2,filepath=default_path_to_gazetter))
  organizations_pattern = KeywordProcessor()
  _ = [organizations_pattern.add_keyword(org) for org in organizations]

  misc = set(openfile(filename='Nationalities.gz', num_char=1, filepath=default_path_to_gazetter))
  misc_pattern = KeywordProcessor()
  _ = [misc_pattern.add_keyword(m) for m in misc]

  main()

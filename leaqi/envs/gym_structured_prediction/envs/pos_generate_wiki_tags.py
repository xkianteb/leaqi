import numpy as np
from wiktionaryparser import WiktionaryParser
from conllu import parse_incr
import multiprocessing as mp
from sklearn.metrics import classification_report
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score,recall_score
import urllib
import tarfile
import os

def get_wiki_tags(entry):
    convert = {'adjective':'ADJ', 'adposition':'ADP', 'preposition':'ADP',\
               'adverb': 'ADV', 'auxiliary':'AUX', 'coordinating conjunction': 'CCONJ',\
               'determiner': 'DET', 'interjection':'INTJ', 'noun':'NOUN',\
               'numeral':'NUM', 'particle':'PART', 'pronoun':'PRON', 'proper noun':'PROPN',\
               'punctuation':'PUNCT', 'subordinating conjunction':'SCONJ', 'symbol':'SYM',\
               'verb':'VERB', 'other':'X', 'article':'DET', 'conjunction':'PART'}
    # ADJ: adjective
    # ADP: adposition
    # ADV: adverb
    # AUX: auxiliary
    # CCONJ: coordinating conjunction
    # DET: determiner
    # INTJ: interjection
    # NOUN: noun
    # NUM: numeral
    # PART: particle
    # PRON: pronoun
    # PROPN: proper noun
    # PUNCT: punctuation
    # SCONJ: subordinating conjunction
    # SYM: symbol
    # VERB: verb
    # X: other

    parser = WiktionaryParser()
    words = entry[1][0]
    wikitionary_tags = []

    for word in words:
        wiki_pos = 'X'
        try:
            results = parser.fetch(word, 'greek') #[0]['definitions']
            if results:
                if results[0]['definitions']:
                    for wiki_idx in range(len(results[0]['definitions'])):
                        wiki_pos = results[0]['definitions'][wiki_idx]['partOfSpeech']
                        if wiki_pos in convert:
                            wiki_pos = convert[wiki_pos]
                            break
                        else:
                            print(f'** cant convert wiki_pos: {wiki_pos}')
                            print(word)
                            print('--------------------------------------')
                            wiki_pos = 'X'
        except AttributeError as error:
            print(f'Error: {error}')
        wikitionary_tags.append(wiki_pos)
    return wikitionary_tags



def main(language='UD_Greek-GDT'):
    directory = f'./pos_dataset_files/ud-treebanks-v2.5'
    if not os.path.isdir(directory):
        udp_treebanks_url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz?sequence=1"
        file_tmp = urllib.request.urlretrieve(udp_treebanks_url, filename=None)[0]
        base_name = os.path.basename(udp_treebanks_url)
        os.system('mkdir -p pos_dataset_files')

        tar = tarfile.open(file_tmp)
        tar.extractall('pos_dataset_files')


    for type in ['train', 'test']:
        filename = f'{directory}/{language}/el_gdt-ud-{type}.conllu'

        sents = []
        y_tags = []
        w_tags = []
        with open(filename, "r") as conllu_file:
            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                annotation = [x for x in annotation if isinstance(x["id"], int)]

                #heads = [x["head"] for x in annotation]
                #tags = [x["deprel"] for x in annotation]
                words = [x["form"] for x in annotation]
                #xpos_tags = [x["xpostag"] for x in annotation]
                upos_tags = [x["upostag"] for x in annotation]
                sents.append(words)
                y_tags.append(upos_tags)
        corpus = {id: (sents[id], y_tags[id]) for id in range(len(sents))}

        with mp.Pool(processes=256) as pool:
           w_tags = pool.map(get_wiki_tags, corpus.items())

        y_true = [x for y in y_tags for x in y]
        y_pred = [x for y in w_tags for x in y]
        sent_lens = [len(y_tags[y]) for y in range(len(y_tags))]

        print(classification_report(y_true, y_pred))
        print(f'accuracy: {accuracy_score(y_true, y_pred)}')
        print(f'recall: {recall_score(y_true, y_pred)}')
        print(f'precision: {precision_score(y_true, y_pred)}')
        print(f'f1 score: {f1_score(y_true, y_pred)}')
        print(f'avg lens: {np.mean(sent_lens)}')
        print(f'num tags: {len(y_pred)}')
        print(classification_report(y_true, y_pred))

        with open(f'pos_dataset_files/{type}_{language}_pos.txt', 'w') as the_file:
            # Write Results to File
            for x in range(len(sents)):
                the_file.write(f'-DOCSTART- -X- -X- O')
                the_file.write(f'\n\n')
                for word_, dic_label_, pos_label_ in zip(sents[x], w_tags[x], y_tags[x]):
                    the_file.write(f'{word_} {dic_label_} {pos_label_}\n')
                the_file.write(f'\n')

if __name__== "__main__":
    main()

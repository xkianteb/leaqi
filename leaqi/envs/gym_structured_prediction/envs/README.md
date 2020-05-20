## NER
* Donwload the conll 2003 ner files 
* place them in `ner_dataset_files` folder
* run `python ner_generate_gazetter.py`

## POS
* Donwload `ud-treebanks-v2.5.tgz`
* untar and place in `pos_dataset_files`
* run `pos_generate_wiki_tags.py`

## Keyphrase
* Download files from https://scienceie.github.io/resources.html
* Train: (https://drive.google.com/open?id=0B2Z1kbILu3YtYjkwMHd3TmNPWDQ)
* Test: (https://drive.google.com/open?id=0B2Z1kbILu3YtMUlfaWZDN0FSUms)
* Place download files in `keyphrase_dataset_files`
* Install `https://github.com/ymym3412/position-rank`
* run `ner_generate_gazetter.py`
* Notes Foundation of `ner_generate_gazetter.py` script: (https://drive.google.com/file/d/0B2Z1kbILu3YtNDl3bVNQVnVjeDg/view)

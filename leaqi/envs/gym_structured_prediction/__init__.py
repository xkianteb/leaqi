import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Ner-v0',
    entry_point='leaqi.envs.gym_structured_prediction.envs:StructuredPredictionEnv',
    kwargs={'bert_model' : 'bert-base-cased',
            'VOCAB': ('<PAD>', 'O', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC'),
            'num_prev_actions': 1,
            'update_interval': 1,
            'ID': 'Ner'},
)

register(
    id='Keyphrase-v0',
    entry_point='leaqi.envs.gym_structured_prediction.envs:StructuredPredictionEnv',
    kwargs={'bert_model' : f'scibert_scivocab_uncased',
            'VOCAB': ('<PAD>', 'O', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC'),
            'num_prev_actions': 1,
            'update_interval': 1,
            'ID': 'Keyphrase'},
)

register(
    id='Pos-v0',
    entry_point='leaqi.envs.gym_structured_prediction.envs:StructuredPredictionEnv',
    kwargs={'bert_model': 'bert-base-multilingual-cased',
            'VOCAB': ('<PAD>','ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'),
    'num_prev_actions': 1,
    'update_interval': 1,
    'ID': 'Pos'},
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
)

from gensim.models import word2vec
from data_load import *
from hparams import Hparams


hparams = Hparams
parser = hparams.parser
hp = parser.parse_args()
"""
en_input, en_label = load_data_iwslt(hp.en_fp, 0)

en_model = word2vec.Word2Vec(en_input, size=hp.d_model, sg=1, iter=8, min_count=0)
en_model.save(hp.en_w2v_model_path)
"""
de_input, de_label = load_data_iwslt(hp.de_fp, 1)

de_model = word2vec.Word2Vec(de_input, size=hp.d_model, sg=1, iter=8, min_count=0)
de_model.save(hp.de_w2v_model_path)

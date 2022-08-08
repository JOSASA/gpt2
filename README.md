from transformers import pipeline
unmasker = pipeline('fill-mask', model='xlm-roberta-base')
unmasker("Hello I'm a <mask> model.")

[{'score': 0.10563907772302628,
  'sequence': "Hello I'm a fashion model.",
  'token': 54543,
  'token_str': 'fashion'},
 {'score': 0.08015287667512894,
  'sequence': "Hello I'm a new model.",
  'token': 3525,
  'token_str': 'new'},
 {'score': 0.033413201570510864,
  'sequence': "Hello I'm a model model.",
  'token': 3299,
  'token_str': 'model'},
 {'score': 0.030217764899134636,
  'sequence': "Hello I'm a French model.",
  'token': 92265,
  'token_str': 'French'},
 {'score': 0.026436051353812218,
  'sequence': "Hello I'm a sexy model.",
  'token': 17473,
  'token_str': 'sexy'}]

Here is how to use this model to get the features of a given text in PyTorch:

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# prepare input
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

# forward pass
output = model(**encoded_input)

BibTeX entry and citation info
@article{DBLP:journals/corr/abs-1911-02116,
  author    = {Alexis Conneau and
               Kartikay Khandelwal and
               Naman Goyal and
               Vishrav Chaudhary and
               Guillaume Wenzek and
               Francisco Guzm{\'{a}}n and
               Edouard Grave and
               Myle Ott and
               Luke Zettlemoyer and
               Veselin Stoyanov},
  title     = {Unsupervised Cross-lingual Representation Learning at Scale},
  journal   = {CoRR},
  volume    = {abs/1911.02116},
  year      = {2019},
  url       = {http://arxiv.org/abs/1911.02116},
  eprinttype = {arXiv},
  eprint    = {1911.02116},
  timestamp = {Mon, 11 Nov 2019 18:38:09 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1911-02116.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
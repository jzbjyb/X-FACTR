# X-FACTR

## Install

Run `conda create -n xfactr -y python=3.7 && conda activate xfactr && ./setup.sh` to prepare the environment.

## Data

- `data/TREx_prompts.csv` prompts of various languages.
- `data/TREx-relations.jsonl` 46 relations.
- `data/mTRExf_unicode_escape.txt` entities and their translations in a variety of languages.
- `data/mTRExf_gender.txt` entities and their gender.
- `data/mTRExf_instanceof.txt` "instance-of" relation of entities.
- `data/alias` aliases of entities.
- `data/mTRExf` facts of 46 relations.

## Probe

To probe LMs with prompts in multiple languages, first build [LAMA](https://github.com/facebookresearch/LAMA) in `../LAMA`.
Then run `python scripts/probe.py --model $LM --lang $LANG` where `$LM` is the LM to probe and `$LANG` is the language.

Supported LMs:
- `mbert_base`: multilingual BERT
- `xlm_base`: XLM
- `xlmr_base`: XLM-R
- `bert_base`, `fr_roberta_base`, `nl_bert_base`, `es_bert_base`, `ru_bert_base`, `zh_bert_base`, `tr_bert_base`, `el_bert_base`: Language-specific BERT

Supported languages: `en`, `fr`, `nl`, `es`, `ru`, `zh`, `he`, `tr`, `ko`, `vi`, `el`, `mr`, `yo`

## Exp

[Google sheet](https://docs.google.com/spreadsheets/d/1oZkH4AFTwoK3ZkNNIy98Ha-D7_Jx4_03n2Mo6Z67hFw/edit?usp=sharing)

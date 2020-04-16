# multiligual LAMA

## Install

Run `conda create -n mlama37 -y python=3.7 && conda activate mlama37 && ./setup.sh` to prepare the environment.

## Data

- `data/TREx_unicode_escape.txt` entities and their translations in a variety of languages.

  format: "entity_id \tab lang1 \tab lang2 \tab ..."
- `data/TREx_gender.txt` entities and their gender.
  
  format: "entity_id \tab gender"
- `data/TREx_instanceof.txt` "instance-of" relation of entities.
  
  format: "entity_id \tab instance-of-id1,name1 \tab instance-of-id2,name2 \tab ..."
- `data/TREx_prompts.csv` prompts of various languages.

## Probe

To probe LMs with prompts in multiple languages, first build [LAMA](https://github.com/facebookresearch/LAMA) in `../LAMA`.
Then run `python scripts/probe.py --model $LM --lang $LANG` where `$LM` is the LM to probe and `$LANG` is the language.

Supported LMs:
- `mbert_base`: multilingual BERT
- `bert_base`: English BERT
- `zh_bert_base`: Chinese BERT

Supported languages: `en`, `zh-cn`, `fr`

## Exp

[Google sheet](https://docs.google.com/spreadsheets/d/1oZkH4AFTwoK3ZkNNIy98Ha-D7_Jx4_03n2Mo6Z67hFw/edit?usp=sharing)

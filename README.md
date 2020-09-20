X-FACTR is a multilingual benchmark for probing factual knowledge in language models.
Prompts in 13 languages are created by native speakers to probe factual knowledge in LMs by having them fill in the blanks of prompts such as "Punta Cana is located in \_."
We provide both the benchmark containing prompts and facts and the code to evaluate LMs.
For more details, check out our paper [X-FACTR: Multilingual Factual Knowledge Retrieval from Pretrained Language Models](link)

# Install

Clone the Github repository and run the following command:
```shell
conda create -n xfactr -y python=3.7 && conda activate xfactr && ./setup.sh
```

# Probing

## Default Decoding
Run LMs on the X-FACTR benchmark with the default decoding (i.e., independently predict multiple tokens) methods:
```
python scripts/probe.py --model $LM --lang $LANG --pred_dir $OUTPUT
```
where `$LM` is the LM to probe (e.g., `mbert_base`), `$LANG` is the language (e.g., `nl`), and `$OUTPUT` is the folder to store predictions.
Evaluate the predictions with `python scripts/ana.py --model $LM --lang $LANG --inp $OUTPUT`

## Confidence-based Decoding
Alternatively, you can run LMs on the X-FACTR benchmark with the confidence-based decoding methods:
```
python scripts/probe.py --model $LM --lang $LANG --pred_dir $OUTPUT --init_method confidence --iter_method confidence --max_iter 10
```

# Benchmark

## Supported languages
- `en` (English), `fr` (French), `nl` (Dutch), `es` (Spanish), `ru` (Russian), `zh` (Chinese), `he` (Hebrew), `tr` (Turkish), `ko` (Korean), `vi` (Vietnamese), `el` (Greek), `mr` (Marathi), `yo` (Yoruba)

## Supported LMs
- [multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md): `mbert_base` 
- [XLM](https://arxiv.org/abs/1901.07291): `xlm_base`
- [XLM-R](https://arxiv.org/abs/1911.02116): `xlmr_base`
- Language-specific BERT: `bert_base` ([the original BERT in English](https://arxiv.org/abs/1810.04805)), `fr_roberta_base` ([CamemBERT](https://arxiv.org/abs/1911.03894)), `nl_bert_base` ([BERTje](https://arxiv.org/abs/1912.09582)), `es_bert_base` ([BETO](https://users.dcc.uchile.cl/~jperez/papers/pml4dc2020.pdf)), `ru_bert_base` ([RuBERT](https://arxiv.org/abs/1905.07213)), `zh_bert_base` ([Chinese BERT](https://github.com/google-research/bert/blob/master/multilingual.md)), `tr_bert_base` ([BERTurk](https://github.com/stefan-it/turkish-bert)), `el_bert_base` ([GreekBERT](https://arxiv.org/abs/2008.12014))

## Dataset
- `data/TREx-relations.jsonl`: metadata of 46 relations.
- `data/TREx_prompts.csv`: manually created prompts in 13 languages.
- `data/mTRExf_unicode_escape.txt`: entity names in different languages.
- `data/mTRExf_gender.txt`: gender information of entities.
- `data/mTRExf_instanceof.txt`: "instance-of" property of entities.
- `data/alias/mTRExf`: aliases of entities.
- `data/mTRExf/sub`: facts (i.e., subject-relation-object tuples) of 46 relations.

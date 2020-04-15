#!/usr/bin/env bash

# multi word lama
# Q11720 has a bug
python scripts/entity_lang.py --task get_lang \
    --inp data/mTREx/sub:data/mTREx_multi_rel.txt \
    --out data/mTREx_unicode_escape.txt

python scripts/check_gender.py \
    --inp data/mTREx_unicode_escape.txt \
    --out data/mTREx_gender.txt

python scripts/check_instanceof.py \
    --inp data/mTREx_unicode_escape.txt \
    --out data/mTREx_instanceof.txt

for lang in en el es fr ja ko mr nl zh; do
    python scripts/entity_lang.py --task get_alias \
        --inp data/mTREx_unicode_escape.txt \
        --out data/alias/mTREx/${lang}.txt \
        --lang ${lang}
done

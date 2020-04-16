#!/usr/bin/env bash

probe=mTREx

# multi word lama
# Q11720, Q5136332 have bugs
python scripts/entity_lang.py --task get_lang \
    --inp data/${probe}/sub:data/${probe}_multi_rel.txt \
    --out data/${probe}_unicode_escape.txt

python scripts/check_gender.py \
    --inp data/${probe}_unicode_escape.txt \
    --out data/${probe}_gender.txt

python scripts/check_instanceof.py \
    --inp data/${probe}_unicode_escape.txt \
    --out data/${probe}_instanceof.txt

for lang in en el es fr ja ko mr nl zh; do
    python scripts/entity_lang.py --task get_alias \
        --inp data/${probe}_unicode_escape.txt \
        --out data/alias/${probe}/${lang}.txt \
        --lang ${lang}
done

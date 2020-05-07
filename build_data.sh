#!/usr/bin/env bash

probe=$1
langs=$2  # a list of languages joined by "," en,zh,fr,nl,es,mr,vi,ko,he,yo,el,tr,ru

IFS=','
read -ra LANGS <<< "${langs}"

# multi word lama
# Q11720, Q5136332 have bugs
: '
python scripts/entity_lang.py --task get_lang \
    --inp data/${probe}/sub:data/${probe}_multi_rel.txt \
    --out data/${probe}_unicode_escape.txt

python scripts/check_gender.py \
    --inp data/${probe}_unicode_escape.txt \
    --out data/${probe}_gender.txt

python scripts/check_instanceof.py \
    --inp data/${probe}_unicode_escape.txt \
    --out data/${probe}_instanceof.txt
'

for lang in "${LANGS[@]}"; do
    python scripts/entity_lang.py --task get_alias \
        --inp data/${probe}_unicode_escape.txt \
        --out data/alias/${probe}/${lang}.txt \
        --lang ${lang}
done

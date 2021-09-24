#!/bin/bash
set -euox pipefail
FILEID=$1
FILENAME=$2
cookie=$(FILEID=$1 ; wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
    --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$cookie&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt

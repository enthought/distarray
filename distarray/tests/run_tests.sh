#!/usr/bin/env sh

for t in `ls test*.py`
do
    python $t
done

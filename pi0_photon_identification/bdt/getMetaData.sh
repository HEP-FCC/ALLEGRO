#!/usr/bin/bash
cd ~/work/fcc/allegro/fullsim
source env.sh > /dev/null
cd run
python printMetaData.py $1 $2 | grep -v Printing | grep -v '^$' | sed 's/^[ \t]*//'

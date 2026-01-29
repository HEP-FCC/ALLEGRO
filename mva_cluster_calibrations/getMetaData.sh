#!/usr/bin/bash
# source the key4hep environment
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh > /dev/null
python printMetaData.py $1 $2 | grep -v Printing | grep -v '^$' | sed 's/^[ \t]*//'

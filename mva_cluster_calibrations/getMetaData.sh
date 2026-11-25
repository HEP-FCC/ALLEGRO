#!/usr/bin/bash
# source the key4hep environment, or restore it if the caller saved it (e.g. the ALLEGRO workflow, which runs the
# python scripts of this directory in the Key4hep environment without its python variables)
if [ -n "$KEY4HEP_SAVED_PYTHONPATH" ]; then
    export PYTHONPATH=$KEY4HEP_SAVED_PYTHONPATH LD_LIBRARY_PATH=$KEY4HEP_SAVED_LD_LIBRARY_PATH
else
    # the arguments of this script would be passed to setup.sh, which would take them as its options
    file=$1 collection=$2
    set --
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh > /dev/null
    set -- "$file" "$collection"
fi
python "$(dirname "$(readlink -f "$0")")"/printMetaData.py "$1" "$2" | grep -v Printing | grep -v '^$' | sed 's/^[ \t]*//'

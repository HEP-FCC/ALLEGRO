#!/usr/bin/bash
a=`root -l -b -q ../ALLEGRO/utils/checkFile.C\(\"$1\",$2\) | tail -n 1`
echo $a

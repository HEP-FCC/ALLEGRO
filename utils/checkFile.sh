#!/usr/bin/bash
a=`root -l -b -q ../FCC-scripts/checkFile.C\(\"$1\",$2\) | tail -n 1`
echo $a

#!/usr/bin/env python

# printMetaData.py
# author: Giovanni Marchiori
#
# prints list of shower shape for given collection

from podio.reading import get_reader
import sys

inputfile = sys.argv[1]
if len(sys.argv)>2:
    collection = sys.argv[2]
else:
    collection = "AugmentedCaloClusters"

print("\nPrinting shapeParameter names for collection %s in file %s:\n" % (collection, inputfile))
reader = get_reader(inputfile)
frames = reader.get("metadata")
frame = frames[0]
decorations = frame.get_parameter('%s__shapeParameterNames' % collection)
if isinstance(decorations, list):
    for i in range(len(decorations)):
        print("{:3}  {}".format(i, decorations[i]))
else:
    print("{:3}  {}".format(0, decorations))
print("")



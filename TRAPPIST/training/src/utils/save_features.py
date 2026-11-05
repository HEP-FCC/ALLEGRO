import random
import string
import os
import pickle

randstr = lambda N: ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))

def save_features(path, object):
    p = os.path.join(path, randstr(10) + ".pkl")
    pickle.dump(object, open(p, "wb"))

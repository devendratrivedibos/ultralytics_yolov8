import os
import glob
from random import shuffle


shuffle(filename)
shuffle(filename)
shuffle(filename)
shuffle(filename)
shuffle(filename)
shuffle(filename)



with open('train_cls.txt', 'w+') as file:
    for f in train:
        file.write(f + '\n')
file.close()

with open('test_cls.txt', 'w+') as file:
    for f in val:
        file.write(f + '\n')
file.close()
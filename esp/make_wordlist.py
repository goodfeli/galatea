#Throwaway script added by Ian Goodfellow

import os

base = '/data/lisa/data/esp_game/ESPGame100k/labels/'
paths = sorted(os.listdir(base))
assert len(paths)==100000

words = {}

for i, path in enumerate(paths):

    if i % 1000 == 0:
        print i
    path = base+path
    f = open(path,'r')
    lines = f.readlines()
    for line in lines:
        word = line[:-1]
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1

ranked_words = sorted(words.keys(), key=lambda x: -words[x])

ranked_words = [word + '\n' for word in ranked_words[0:4000]]


f = open('/data/lisatmp/goodfeli/esp/wordlist.txt','w')
f.writelines(ranked_words)
f.close()

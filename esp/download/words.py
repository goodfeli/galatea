words = open('/data/lisatmp/goodfeli/esp/wordlist.txt')
words = [word[:-1] for word in words.readlines()]

from numpy import random
import time

failed_words = []

from googleImageQuery import googleImageQuery

import os
for i, word in enumerate(words[:1200]):
    if i < 722:
        continue
    if os.path.exists('/data/lisatmp/goodfeli/esp/word_labels/'+word+'.txt'):
        print word,'labeled already'
        continue
    try:
        print 'word',i
        print '\tword=' + word
        print '\tquerying at',time.ctime()
        t1 = time.time()
        results = googleImageQuery(word, nbPages=1, useFullSize=1)
        t2 = time.time()
        print '\tquery returned after',t2-t1
        results[0][0].save('/data/lisatmp/goodfeli/esp/word_images/%s.png' % word)
        time.sleep(random.randint(20, 25))
    except Exception, e:
        failed_words.append(word)

print 'failed words: '
print failed_words

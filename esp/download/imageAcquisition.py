"""
This script is responsible for the first step of the facial expression dataset
creation. Its role is to perform multiple Google image search queries to
obtain images showing people with various emotions.

The queries serach terms are performed by combining various keywords and
translating the results into different languages.
"""

import googleImageQuery
from itertools import product
from sets import Set
import hashlib
import pickle
import dbConnect
import sys
import time
import random

# Number of pages of image results to take for any google image query
NB_PAGES_QUERY = 50

STATUS_FILENAME = "imageAcquisitionStatus.txt"


def loadKeywordList(filepath):
    keywords = []

    # Try to load and normalize the content of the keywords file
    try:
        f = open(filepath)

        # Obtain keywords. Comments (lines starting with '#') are ignored.
        keywords = [keyword.strip().lower() for keyword in f.readlines()
                    if keyword[0] != '#']

        # Remove duplicates
        if keywords:
            keywords.sort()
            last = keywords[-1]
            for i in range(len(keywords) - 2, -1, -1):
                if last == keywords[i]:
                    del keywords[i]
                else:
                    last = keywords[i]

        f.close()
    except:
        pass

    return keywords


if __name__ == '__main__':

    # Load the execution status dictionary to see if image acquisition must be
    # resumed from a point previouly reached.
    try:
        status = pickle.load(open(STATUS_FILENAME))
    except:
        # Create a default execution status dictionary
        status = {"initDatabase": False,
                  "insertKeywords": False,
                  "insertQueries": 0,
                  "mergeDuplicates": False}

    # Initialize the database
    db = dbConnect.DBConnect()
    if status['initDatabase']:
        # Database has already been initialized. This step is skipped.
        print "Database has already been initialized."
    else:
        success = db.initDatabase()
        if success:
            # Inform the user and update the status dictionary
            print "Database initialization completed"
            status['initDatabase'] = True
            try:
                pickle.dump(status, open(STATUS_FILENAME, 'w'))
                print "Execution status dictionary updated "
            except:
                print "Could not update execution status dictionary. "
        else:
            # Inform the user and stop the script
            print "Database initialization has failed"
            print "Stopping script"
            sys.exit()

    # Load the keywords lists
    # emotionWordss : Names of emotions (anger, happiness, ...) +
    #                 Adjectives/Verns indicating an emotion (happy, ...)
    # identity : Words indicating sex and/or age : baby, boy, girl, man, ...)
    # ethnicity : Words refering to the ethnicity of the subjects.
    emotionWords = loadKeywordList('./keywords/emotionWords.txt')
    identities = loadKeywordList('./keywords/identity.txt')
    ethnicities = loadKeywordList('./keywords/ethnicity.txt')

    # Generate composite search terms for the identity and ethnicity keywords.
    identitiesSearchTerms = '"' + ('"|"'.join(identities)) + '"'
    ethnicitiesSearchTerms = '"' + ('"|"'.join(ethnicities)) + '"'

    # Compute the set associated with each keyword list for easier comparison
    # of the specificity of request search terms.
    emotionWordsSet = Set(emotionWords)
    identitiesSet = Set(identities)
    ethnicitiesSet = Set(ethnicities)

    # Insert the keywords in the database
    if status['insertKeywords']:
        print "Keywords have already been inserted in the database."
    else:
        success = True
        success = success & db.deleteKeywords()
        success = success & db.insertKeywords(emotionWords, "emotionWord")
        success = success & db.insertKeywords(identities, "identity")
        success = success & db.insertKeywords(ethnicities, "ethnicity")

        if success:
            # Inform the user and update the status dictionary
            print "Keywords insertion completed"
            status['insertKeywords'] = True
            try:
                pickle.dump(status, open(STATUS_FILENAME, 'w'))
                print "Execution status dictionary updated "
            except:
                print "Could not update execution status dictionary. "
        else:
            # Inform the user and stop the script
            print "Keywords insertion has failed"
            print "Stopping script"
            sys.exit()

    # Generate a lists of search terms by combining the various keywords
    # Keywords are combined according to the following patterns :
    # 1  - EmotionWord
    # 2  - EmotionWord + Identity
    # 3  - EmotionWord + Identity + Ethnicity
    """
    searchTerms = (list(emotionWords) +
                   [w + ' ' + identitiesSearchTerms for w in emotionWords] +
                   [w + ' ' + identitiesSearchTerms + ' ' + ethnicitiesSearchTerms for w in emotionWords])
    """
    searchTerms = [i for i in identities]

    # Generate a list that indicates, for each search query, how many results
    # pages should be kept.
    searchTermsNbPages = [NB_PAGES_QUERY, ] * len(searchTerms)

    # Perform the queries, one by one, and insert the results in the database
    print "Starting images queries"

    #for idx in range(status['insertQueries'], len(searchTerms)):
    for idx in range(len(searchTerms)):

        searchTerm = searchTerms[idx]

        if searchTerm is list or isinstance(searchTerm, tuple):
            cleanSearchTerm = ' '.join(searchTerm)
        else:
            cleanSearchTerm = searchTerm

        print "Google image search for " + str(cleanSearchTerm)

        try:
            results = googleImageQuery.googleImageDataQuery(cleanSearchTerm, searchTermsNbPages[idx])

            time.sleep(random.randint(20, 25))
            print len(results[0])

            success = db.insertQuery(cleanSearchTerm, results[1], results[2], results[3], results[4])
        except:
            success = False

        if success:
            # Inform the user and update the status dictionary
            print ("Google image search for " + str(cleanSearchTerm) +
                   " completed.")
            status['insertQueries'] = idx + 1
            try:
                pickle.dump(status, open(STATUS_FILENAME, 'w'))
                print "Execution status dictionary updated "
            except:
                print "Could not update execution status dictionary. "
        else:
            # Inform the user and stop the script
            print ("Google image search for " +
                   str(cleanSearchTerm) + " has failed.")
            print "Stopping script"
            sys.exit()

    print "Finished images queries"

    # Go through the database and merge the images that share the same source
    # URL.
    # ALERT : This section has been deactivated. The process to merge
    # identical images (or from the same URLs) has been moved to after the
    # manual labeling phase.
    """
    if status['mergeDuplicates'] == False:

        duplicates = db.getImageDuplicatesByURL()
        if duplicates == None:
            # Inform the user and stop the script
            print "Query for the duplicate images in the database has failed."
            print "Stopping script"
            sys.exit()

        while len(duplicates) > 0:

            # Select one pair of duplicates
            id1 = duplicates[0][0]
            searchTerms1 = duplicates[0][1]
            id2 = duplicates[0][2]
            searchTerms2 = duplicates[0][3]

            # Figure out if they need to be merged
            searchTerms1Set = Set(searchTerms1.split(" ")) - fillerWordsSet
            searchTerms2Set = Set(searchTerms2.split(" ")) - fillerWordsSet

            if searchTerms1Set.issubset(searchTerms2Set):
                success = db.deleteImageById(id1)
                idDelete = id1
            elif searchTerms2Set.issubset(searchTerms1Set):
                success = db.deleteImageById(id2)
                idDelete = id2
            else:
                # No merge to do. Automatic success.
                success = True
                idDelete = None

            if success:

                if idDelete == None:
                    # Remove the entry from the duplicates list because
                    # there is no obvious was to merge the two images.
                    del duplicates[0]
                else:
                    # Remove, from the duplicates list, the entries refering
                    # to the image, if any, that was deleted by the merging
                    # process
                    idx = 0
                    while idx < len(duplicates):
                        if (duplicates[idx][0] == idDelete or
                            duplicates[idx][2] == idDelete):

                            del duplicates[idx]
                        else:
                            idx += 1

            else:
                # Inform the user and stop the script
                print "Merge of images %i and %i has failed." % (id1, id2)
                print "Stopping script"
                sys.exit()

        print ("Merging of image duplicates is completed.")
        status['mergeDuplicates'] = True
        try:
            pickle.dump(status, open(STATUS_FILENAME, 'w'))
            print "Execution status dictionary updated "
        except:
            print "Could not update execution status dictionary. "
    """

    print "Image acquisition is complete."

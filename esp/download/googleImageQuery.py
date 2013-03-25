from PIL import Image
import requests
from requests.exceptions import ConnectionError

import google
import urllib
import hashlib

from StringIO import StringIO


def downloadImage(url):
    image = None

    try:
        image_r = requests.get(url)
        image_content = StringIO(image_r.content)
        image = Image.open(image_content)
    except:
        print "Could not download image : %s" % url

    return image


def googleImageDataQuery(keyword, nbPages):

    """
    Perform a google query with the given keyword and return
    data about the resulting images without downloading them.

    keyword : Expression to use as image query

    nbPages : Number of result pages from which to obtain the images

    return : List of lists detailing the results of the image search query.
             return[0] = List of PIL.Image.Image instances corresponding to
                         the images themselves
             return[1] = URLs of the fullsize images
             return[2] = URLs of the thumbnail images
             return[3] = hashes of the downloaded images
             return[4] = captions displayed alongside the thumbnails in the
                         the google image search result page.

    """

    # Perform the image query for the given parameters
    options = google.ImageOptions()
    results = google.Google.search_images(keyword, options, nbPages)

    # Individualy download every image in the result, parse them to PIL.Image
    # objects and return them.
    images = [None, ] * len(results)
    imagesURL = [None, ] * len(results)
    thumbsURL = [None, ] * len(results)
    captions = [None, ] * len(results)
    hashes = [None, ] * len(results)

    for i in range(len(results)):
        imagesURL[i] = urllib.unquote(results[i].link)
        thumbsURL[i] = urllib.unquote(results[i].thumb)
        captions[i] = urllib.unquote(results[i].caption)

    # iamges and hashes will be none
    return (images, imagesURL, thumbsURL, captions, hashes)


def googleImageQuery(keyword, nbPages, useFullSize):

    """
    Perform a google query with the given keyword and return
    the resulting images.

    keyword : Expression to use as image query

    nbPages : Number of result pages from which to obtain the images

    useFullSize : False if the script should download and return the
                  thumbnails displayed in the result pages or True for the
                  full size images themselves.

    return : List of lists detailing the results of the image search query.
             return[0] = List of PIL.Image.Image instances corresponding to
                         the images themselves
             return[1] = URLs of the fullsize images
             return[2] = URLs of the thumbnail images
             return[3] = hashes of the downloaded images
             return[4] = captions displayed alongside the thumbnails in the
                         the google image search result page.

    """

    # Perform the image query for the given parameters
    options = google.ImageOptions()
    #options.image_type = google.ImageType.FACE
    results = google.Google.search_images(keyword, options, nbPages)

    # Individualy download every image in the result, parse them to PIL.Image
    # objects and return them.
    images = [None, ] * len(results)
    imagesURL = [None, ] * len(results)
    thumbsURL = [None, ] * len(results)
    captions = [None, ] * len(results)
    hashes = [None, ] * len(results)

    for i in range(len(results)):
        imagesURL[i] = urllib.unquote(results[i].link)
        thumbsURL[i] = urllib.unquote(results[i].thumb)
        captions[i] = urllib.unquote(results[i].caption)
        try:
            if useFullSize:
                image_r = requests.get(results[i].link)
            else:
                image_r = requests.get(results[i].thumb)
            image_content = StringIO(image_r.content)

            hashes[i] = hashlib.md5(image_content.getvalue()).hexdigest()

            #Image.open(StringIO(image_r.content)).save(str(i-1) + ".png")
            images[i] = Image.open(image_content)
        except ConnectionError, e:
            print 'could not download %s' % imagesURL[i]
        except IOError, e:
            print str(e)

    # Clean out the None values of the images and imagesURL lists
    imagesClean = []
    imagesURLClean = []
    thumbsURLClean = []
    captionsClean = []
    hashesClean = []

    for i in range(len(images)):
        if images[i] != None:
            imagesClean.append(images[i])
            imagesURLClean.append(imagesURL[i])
            thumbsURLClean.append(thumbsURL[i])
            captionsClean.append(captions[i])
            hashesClean.append(hashes[i])

    return (imagesClean, imagesURLClean, thumbsURLClean,
            captionsClean, hashesClean)

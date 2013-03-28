#!/usr/bin/python

from BeautifulSoup import BeautifulSoup
from BeautifulSoup import NavigableString
from pprint import pprint
import os
import threading
import httplib
import urllib
import urllib2
import sys
import re
import random
try:
    import json
except ImportError:
    import simplejson as json

__author__ = "Anthony Casagrande <birdapi@gmail.com>"
__version__ = "0.9"

"""
Represents a google image search result
"""
class ImageResult:  
    def __init__(self):
        self.name = None
        self.link = None
        self.thumb = None
        self.thumb_width = None
        self.thumb_height = None
        self.width = None
        self.height = None
        self.filesize = None
        self.format = None
        self.domain = None
        self.page = None
        self.index = None
        self.caption = None
        
class ImageOptions:
    def __init__(self):
        self.image_type = None
        self.size_category = None
        self.larger_than = None
        self.exact_width = None
        self.exact_height = None
        self.color_type = None
        self.color = None
        
    def get_tbs(self):
        tbs = None
        if self.image_type:
            # clipart
            tbs = add_to_tbs(tbs, "itp", self.image_type)
        if self.size_category and not (self.larger_than or (self.exact_width and self.exact_height)): 
            # i = icon, l = large, m = medium, lt = larger than, ex = exact
            tbs = add_to_tbs(tbs, "isz", self.size_category)
        if self.larger_than:   
            # qsvga,4mp
            tbs = add_to_tbs(tbs, "isz", SizeCategory.LARGER_THAN)
            tbs = add_to_tbs(tbs, "islt", self.larger_than)
        if self.exact_width and self.exact_height:
            tbs = add_to_tbs(tbs, "isz", SizeCategory.EXACTLY)
            tbs = add_to_tbs(tbs, "iszw", self.exact_width)
            tbs = add_to_tbs(tbs, "iszh", self.exact_height)
        if self.color_type and not self.color:
            # color = color, gray = black and white, specific = user defined
            tbs = add_to_tbs(tbs, "ic", self.color_type)
        if self.color:
            tbs = add_to_tbs(tbs, "ic", ColorType.SPECIFIC)
            tbs = add_to_tbs(tbs, "isc", self.color)
        return tbs
        
"""
Defines the public static api methods
"""
class Google:
    DEBUG_MODE = False
    
    @staticmethod
    def search_images(query, image_options = None, page = 1):
        
        results = []
        for i in range(page):
            url = get_image_search_url(query, image_options, i)
            html = get_html(url)
                        
            if html:
                if Google.DEBUG_MODE:
                    write_html_to_file(html, "images_{0}_{1}.html".format(query.replace(" ", "_"), i))
                soup = BeautifulSoup(html)
                j = 0
                tds = soup.findAll("td")

                for td in tds:
                    a = td.find("a")
                    if a and a["href"].find("imgurl") != -1:
                                                
                        res = ImageResult()
                        res.page = i
                        res.index = j
                        tokens = a["href"].split("&")
                        match = re.search("imgurl=(?P<link>[^&]+)", tokens[0])
                        if match:
                            res.link = match.group("link")
                            res.format = res.link[res.link.rfind(".") + 1:]
                        img = td.find("img")
                        if img:
                            res.thumb = img["src"]
                            res.thumb_width = img["width"]
                            res.thumb_height = img["height"]
                        match = re.search("(?P<width>[0-9]+) &times; (?P<height>[0-9]+) - (?P<size>[^&]+)", td.text)
                        if match:
                            res.width = match.group("width")
                            res.name = td.text[:td.text.find(res.width)]
                            res.height = match.group("height")
                            res.filesize = match.group("size")
                        cite = td.find("cite")
                        if cite:
                            res.domain = cite["title"]
                            
                            if (cite.next != None and
                                cite.next.next != None and
                                cite.next.next.name == 'br' and
                                cite.next.next.next != None):
                                   
                                res.caption = ''
                                currentNode = cite.next.next.next
                                
                                keepParsing = True
                                while currentNode != None and keepParsing:
                                    if isinstance(currentNode, NavigableString):
                                        res.caption += currentNode
                                    else:
                                        keepParsing = currentNode.name != 'br'
                                    currentNode = currentNode.next
                                    
                                res.caption = BeautifulSoup(res.caption, convertEntities=BeautifulSoup.HTML_ENTITIES).text                                 
                        
                        results.append(res)
                        j = j + 1
        return results
    
        
  
def normalize_query(query):
    return query.strip().replace(":", "%3A").replace("+", "%2B").replace("&", "%26").replace(" ", "+")
  
class ImageType:
    NONE = None
    FACE = "face"
    PHOTO = "photo"
    CLIPART = "clipart"
    LINE_DRAWING = "lineart"
    
class SizeCategory:
    NONE = None
    ICON = "i"
    LARGE = "l"
    MEDIUM = "m"
    SMALL = "s"
    LARGER_THAN = "lt"
    EXACTLY = "ex"
    
class LargerThan:
    NONE = None
    QSVGA = "qsvga" # 400 x 300
    VGA = "vga"     # 640 x 480
    SVGA = "svga"   # 800 x 600
    XGA = "xga"     # 1024 x 768
    MP_2 = "2mp"    # 2 MP (1600 x 1200)
    MP_4 = "4mp"    # 4 MP (2272 x 1704)
    MP_6 = "6mp"    # 6 MP (2816 x 2112)
    MP_8 = "8mp"    # 8 MP (3264 x 2448)
    MP_10 = "10mp"  # 10 MP (3648 x 2736)
    MP_12 = "12mp"  # 12 MP (4096 x 3072)
    MP_15 = "15mp"  # 15 MP (4480 x 3360)
    MP_20 = "20mp"  # 20 MP (5120 x 3840)
    MP_40 = "40mp"  # 40 MP (7216 x 5412)
    MP_70 = "70mp"  # 70 MP (9600 x 7200)

class ColorType:
    NONE = None
    COLOR = "color"
    BLACK_WHITE = "gray"
    SPECIFIC = "specific"
    
def get_image_search_url(query, image_options=None, page=0, per_page=20):
    query = query.strip().replace(":", "%3A").replace("+", "%2B").replace("&", "%26").replace(" ", "+")
    url = "http://images.google.com/images?q=%s&sa=N&start=%i&ndsp=%i&sout=1" % (query, page * per_page, per_page)
    if image_options:
        tbs = image_options.get_tbs()
        if tbs:
            url = url + tbs
    return url
    
def add_to_tbs(tbs, name, value):
    if tbs:
        return "%s,%s:%s" % (tbs, name, value)
    else:
        return "&tbs=%s:%s" % (name, value) 
           
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
        
def get_rand_user_agent():
    user_agent_strings = ["Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
                          "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.1) Gecko/2008070208 Firefox/3.0.1"
                          "Mozilla/5.001 (windows; U; NT4.0; en-US; rv:1.0) Gecko/25250101"
                          "Opera/9.25 (Windows NT 6.0; U; en)"
                          "Mozilla/4.8 [en] (Windows NT 6.0; U)",
                          "Mozilla/5.0 (Windows NT 6.0; WOW64) AppleWebKit/534.24 (KHTML, like Gecko) Chrome/11.0.696.16 Safari/534.24"]
    """
    user_agent_strings = ["Mozilla/5.001 (windows; U; NT4.0; en-US; rv:1.0) Gecko/25250101"
                          "Mozilla/5.0 (Windows NT 6.0; WOW64) AppleWebKit/534.24 (KHTML, like Gecko) Chrome/11.0.696.16 Safari/534.24"]
    """
    user_agent_idx = random.randint(0, len(user_agent_strings) - 1)
    
    return user_agent_strings[user_agent_idx]
    
    
def get_html(url):
    try:
        request = urllib2.Request(url)
        #request.add_header("User-Agent", "Mozilla/5.001 (windows; U; NT4.0; en-US; rv:1.0) Gecko/25250101")
        request.add_header("User-Agent", get_rand_user_agent())
        html = urllib2.urlopen(request).read()
        return html
    except KeyboardInterrupt:
        raise
    except Exception, e:
        import pdb
        pdb.set_trace()
        print "Error accessing:", url
        return None        

def write_html_to_file(html, filename):
    of = open(filename, "w")
    of.write(html)
    of.flush()
    of.close()
        
def test():
    search = Google.search("github")
    if search is None or len(search) == 0: 
        print "ERROR: No Search Results!"
    else: 
        print "PASSED: {0} Search Results".format(len(search))
    
    options = ImageOptions()
    options.image_type = ImageType.CLIPART
    options.larger_than = LargerThan.MP_4
    options.color = "green"
    images = Google.search_images("banana", options)
    if images is None or len(images) == 0: 
        print "ERROR: No Image Results!"
    else:
        print "PASSED: {0} Image Results".format(len(images))
        
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        Google.DEBUG_MODE = True
        print "DEBUG_MODE ENABLED"
    test()
        
if __name__ == "__main__":
    main()
    
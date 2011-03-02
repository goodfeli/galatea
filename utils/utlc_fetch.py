import mechanize
import re
from auc.embed import score
import zipfile
from tempfile import TemporaryFile
import numpy
import sys
import traceback
#import pylab

def main(data="terry", exp_list=None):
    br = mechanize.Browser()
    br.open("http://www.causality.inf.ethz.ch/unsupervised-learning.php?page=login")
    br.select_form(nr=0)
    br.form["repo_login"] = "lisa@iro.umontreal.ca"
    br.form["repo_password"] = "deep99"
    br.submit()

    print "Opening web page"
    res = br.open("http://www.causality.inf.ethz.ch/unsupervised-learning.php?page=mylab&model="+data).read()
   
    print "Finding all submissions"
    subfiles = re.findall(r"""<td>(.*)</td>\s+?<td>(xxxxxx|NA)</td>\s+?<td>(xxxxxx|NA)</td>\s+?<td>\[<a href="(http://mllab1.inf.ethz.ch/VIRTUAL_LAB_DOCUMENTS/queries/.*?.zip)""", res, re.MULTILINE)

    exp_list_str = ""
    if exp_list != None:
        exp_list_str = "_indexes_" + "_".join([str(i) for i in exp_list])
    output_file = open("alc_vs_vt_"+data+exp_list_str+".log", "w")
    
    real_alc = []
    validtest_alc = []

    for i, (alc, test_score, test_global_score, subfile) in enumerate(subfiles):
        entry_nb = i+1
        
        if exp_list != None and not (entry_nb in exp_list):
            print "Skipping entry " + str(i+1)
            continue

        print "Processing " + str(subfile)
        r = br.open(subfile)
        f = TemporaryFile()
        f.write(r.read())
        f.seek(0)
        
        zf = zipfile.ZipFile(f, "r")
        
        # Needs to contain both valid and test
        if len(zf.namelist()) != 2:
            continue
        try:
            print "Preparing data"
            valid_filename = filter(lambda x: x.find("valid") != -1, zf.namelist())[0]
            test_filename = filter(lambda x: x.find("final") != -1, zf.namelist())[0]
        
            valid_file = TemporaryFile()
            test_file = TemporaryFile()
            
            valid_file.write(zf.read(valid_filename))
            test_file.write(zf.read(test_filename))
            
            valid_file.seek(0)
            test_file.seek(0)
            
            valid_arr = numpy.loadtxt(valid_file)
            test_arr = numpy.loadtxt(test_file)
            

            if numpy.all(valid_arr == test_arr):
                continue
           
            print "Found a valid test dataset"
            real_alc.append(alc)
            dataset = numpy.vstack((valid_arr, test_arr))
            labels_valid = numpy.hstack((numpy.ones((valid_arr.shape[0], 1)),\
                                        numpy.zeros((valid_arr.shape[0], 1))))
            labels_test = numpy.hstack((numpy.zeros((test_arr.shape[0], 1)),\
                                        numpy.ones((test_arr.shape[0], 1))))
            labels  = numpy.vstack((labels_valid, labels_test))
            print "Computing score"
            validtest_alc.append(score(dataset, labels))

            print real_alc[-1], validtest_alc[-1], entry_nb
            print >> output_file, real_alc[-1], validtest_alc[-1], entry_nb
        except Exception, e:
            print traceback.print_exc()
            continue
    
    #pylab.plot(validtest_alc, real_alc, "rx")
    #pylab.show()


if __name__ == "__main__":
    args = sys.argv

    if len(args) == 2:
        main(data=args[1])
    elif len(args) == 3:
        main(data=args[1], exp_list=[ int(i) for i in args[2].split(",") ])
    elif len(args) == 4:
        main(data=args[1], exp_list=range(int(args[2]), int(args[3]) + 1))
    else:
        main(data="terry")

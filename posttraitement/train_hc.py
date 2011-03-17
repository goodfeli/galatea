from numpy import *
from posttraitement.hc import train, save_submission
from auc.embed import score
import traceback

def optimize(data_valid, data_test, dataset_name):
    best_alc = 0
    best_rep_valid = None
    best_rep_test  = None
    best_options = None


    # Find the best combination of hyper-parameters
    for step in range(1,6):
        for k in range(2,4):
            for prob in ["posterior", "likelihood"]:
                options = {\
                    "whiten":True,\
                    "probs":prob,\
                    "k":k,\
                    "steps":step\
                }
                data = vstack((data_valid, data_test))
                
                print "Training with " + str(options)
                dataset = train(data, options)

                labels_valid = hstack((ones((data_valid.shape[0], 1)), zeros((data_valid.shape[0], 1))))
                labels_test  = hstack((zeros((data_test.shape[0], 1)), ones((data_test.shape[0], 1))))

                labels = vstack((labels_valid, labels_test))

                print "Computing score"
                try:
                    alc = score(dataset, labels)
                except Exception, e:
                    print "Setting alc to -1 because the following exception was received: " + str(e)
                    print "Stack trace: "
                    traceback.print_exc()
                    alc = -1

                if alc > best_alc:
                    print "Best alc found: " + str(alc)
                    best_alc = alc
                    del best_rep_valid
                    del best_rep_test
                    best_rep_valid = dataset[:data_valid.shape[0],:]
                    best_rep_test  = dataset[data_valid.shape[0]:,:]
                    best_options = options
                else:
                    del dataset

                del labels
                del labels_valid
                del labels_test

    if best_alc == 0:
        print "No interesting result found, returning"
        return

    savename = dataset_name + "_alc_" + str(best_alc).replace(".","_")
    if best_options == None:
        savename += "_base"
    else:
        savename +="_whiten_" + str(best_options["whiten"]) +\
                 "_steps_" + str(best_options["steps"]) +\
                 "_k_" + str(best_options["k"]) +\
                 "_probs_" + str(best_options["probs"])

    save_submission(best_rep_valid, savename, "./Results", True)
    save_submission(best_rep_test, savename, "./Results", False)

if __name__ == "__main__":
    print "Loading valid:"
    data_valid = loadtxt("/u/lavoeric/ift6266h11/harry/harry_lisa_valid.prepro")
    print data_valid
    print "Training"
    rep = train(data_valid, {"whiten":False, "probs":"posterior", "k":2, "steps":6})
    print "Saving submission"
    save_submission(rep, "hc_whiten_False_probs_posterior_k_2_steps_6_fullcov_harry", "./Results", True)
    #data_valid = loadtxt("/u/lavoeric/ift6266h11/rita/NEW_VALID_PCA_5whiten_Falserita_dl_valid.prepro")
    #print "Loading test:"
    #data_test  = loadtxt("/u/lavoeric/ift6266h11/rita/NEW_VALID_PCA_5whiten_Falserita_dl_final.prepro")
    #dataset_name = "rita"
    #data = load("/data/lisa/exp/dauphiya/stackedterry/best_layer0/terry_valid.npy")
    #optimize(data_valid, data_test, dataset_name)

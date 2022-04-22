import numpy as np
import pickle
import matplotlib.pyplot as plt

from qtest_list import qtest_list_dict
from qtest_library import *

qtest_cut_dict = {qtest_tt.__name__:5e-3, qtest_chi2.__name__:2, qtest_ks.__name__:5e-3, qtest_residuals_per_image.__name__:5, qtest_residuals_per_rowCol.__name__:5, qtest_chi2FromFit.__name__:2}


def execute_qtests(data_path, data_ref_path, n_data, n_ref, dqmdir, **kwargs):
    """ Executes the qtests 
    """
    data = np.load(data_path, allow_pickle=True)["arr_0"].item()
    data_ref = np.load(data_ref_path, allow_pickle=True)["arr_0"].item()

    n_std = kwargs["n_std"] if "n_std" in kwargs else 3
    display = kwargs["display"] if "display" in kwargs else False

    ### Create ref dict
    me_ref = {}
    for n_im in range(len(data_ref["Images"])):
        for n_ccd in range(len(data_ref["Images"][n_im]["CCDs"])):
            for n_ME in range(len(data_ref["Images"][n_im]["CCDs"][n_ccd]["MEs"])):
                me_i = data_ref["Images"][n_im]["CCDs"][n_ccd]["MEs"][n_ME]
                id_ME = me_i["name"]
                                
                if id_ME not in me_ref: me_ref[id_ME] = {}
                if n_ccd not in me_ref[id_ME]: me_ref[id_ME][n_ccd] = {}

                for amp in ["U2","L2"]:
                    if "x" not in me_ref[id_ME][n_ccd]: me_ref[id_ME][n_ccd]["x"] = []
                    if "y" not in me_ref[id_ME][n_ccd]: me_ref[id_ME][n_ccd]["y"] = []
                    if "amplifier" not in me_ref[id_ME][n_ccd]: me_ref[id_ME][n_ccd]["amplifier"] = []
                    if "run" not in me_ref[id_ME][n_ccd]: me_ref[id_ME][n_ccd]["run"] =n_ref 
                    if "title" not in me_ref[id_ME][n_ccd]: me_ref[id_ME][n_ccd]["title"] = id_ME

                    x = np.array(me_i["x"])[np.array(me_i["amplifier"])==amp].tolist()
                    y = np.array(me_i["y"])[np.array(me_i["amplifier"])==amp].tolist()
                    if len(x) == 1:
                        x = x[-1]
                        y = y[-1]
                    me_ref[id_ME][n_ccd]["x"].append(x) 
                    me_ref[id_ME][n_ccd]["y"].append(y) 
                    me_ref[id_ME][n_ccd]["amplifier"].append(amp)
                    #if id_ME == "MEOVSPedestalMu":
                    #    print(amp)
                    #    print(x)


    q_test_results = {}
    ### Run QTESTS over CCDs
    for n_im in range(len(data["Images"])):
        for n_ccd in range(len(data["Images"][n_im]["CCDs"])):
            for n_ME in range(len(data["Images"][n_im]["CCDs"][n_ccd]["MEs"])):
                me_i = data["Images"][n_im]["CCDs"][n_ccd]["MEs"][n_ME]
                id_ME = me_i["name"]
                #print(id_ME)

                ### Execute qtest per image
                #print(id_ME)


                ### To store qtests in pkl should be something like this
                data["Images"][n_im]["CCDs"][n_ccd]["MEs"][n_ME]["tests"] = []

                for qtest in qtest_list_dict[id_ME]:
                    #print(qtest.__name__)
                    qcut = qtest_cut_dict[qtest.__name__]
                    for amp in ["U2","L2"]:
                        x = np.array(me_i["x"])[np.array(me_i["amplifier"])==amp].tolist()
                        y = np.array(me_i["y"])[np.array(me_i["amplifier"])==amp].tolist()
 
                        if len(x) > 0:# and x is not None:
                            if len(x) == 1:
                                x,y = x[-1],y[-1]
                            else:
                                x,y = x,y

                            if qtest == qtest_chi2FromFit:
                                chi2 = np.array(me_i["goodness"])[np.array(me_i["amplifier"])==amp]
                                #if len(chi2) == 2:## 2 => Number of amps
                                #    chi2 = chi2[-1]
                                #print(qtest.__name__)
                                qt = qtest(x,chi2[0],None,amp,None,qcut,is_image=True)
                            else:
                                #print(qtest.__name__)
                                qt = qtest(x,y,me_ref[id_ME][n_ccd],amp,n_std,qcut,is_image=True)

                            #if "FitDC" in id_ME:
                            #        print(id_ME)
                            #        print(qtest.__name__)
                            #        print(qt["result"])
                            ### TO BE DECIDED IF ME WITH NO QTEST ARE STORED
                            if id_ME not in q_test_results: q_test_results[id_ME] = {}
                            if n_ccd not in q_test_results[id_ME]: q_test_results[id_ME][n_ccd] = {}
                            ### Store results per ME/amp/Qtest/Image
                            if amp not in q_test_results[id_ME][n_ccd]: q_test_results[id_ME][n_ccd][amp] = {}

                            ### Temporary dont store Nan qtest
                            if qt["result"] is not np.nan:
                                if qtest.__name__ not in q_test_results[id_ME][n_ccd][amp]: q_test_results[id_ME][n_ccd][amp][qtest.__name__] = []
                                q_test_results[id_ME][n_ccd][amp][qtest.__name__].append(qt["result"])
                            data["Images"][n_im]["CCDs"][n_ccd]["MEs"][n_ME]["tests"].append(qt)

                ### Empty ME uses QTNone
                if len(data["Images"][n_im]["CCDs"][n_ccd]["MEs"][n_ME]["tests"]) == 0:
                   data["Images"][n_im]["CCDs"][n_ccd]["MEs"][n_ME]["tests"] = [{'name':'QTnone','result':True, 'x':None, 'y':None,'mean_ref':0,'n_std':0, 'std_ref':0, 'run_ref': 0}] * len(["U2","L2"])
                ### TODO npz should be saved again?

    ### Dictionary ordered by ME/Amp/List of Qtests
    bad_images = {} 
    for id_ME in q_test_results:
        if id_ME not in bad_images: bad_images[id_ME] = {}
        for n_ccd in q_test_results[id_ME]:
            if n_ccd not in bad_images[id_ME]: bad_images[id_ME][n_ccd] = {}
            for amp in q_test_results[id_ME][n_ccd]:
                if amp not in bad_images[id_ME][n_ccd]: bad_images[id_ME][n_ccd][amp] = []
                for q_test in q_test_results[id_ME][n_ccd][amp]:
                    bad_images[id_ME][n_ccd][amp].append(q_test_results[id_ME][n_ccd][amp][q_test])

    ### Dictionary ordered by ME/Amp/averaged failed qtest results per type of ME
    bad_images_per_type = {"PerRow":{}, "PerCol":{},"FitDC":{}}#, "OVS":{}}
    for id_ME in bad_images:
        for ME_type in bad_images_per_type:
            for n_ccd in q_test_results[id_ME]:
                if n_ccd not in bad_images_per_type[ME_type]: bad_images_per_type[ME_type][n_ccd] = {}
                for amp in bad_images[id_ME][n_ccd]:
                    if amp not in bad_images_per_type[ME_type][n_ccd]: bad_images_per_type[ME_type][n_ccd][amp] = []
                    if ME_type in id_ME and "OVS" not in id_ME:
                        bad_images_per_type[ME_type][n_ccd][amp].append(1-np.mean(bad_images[id_ME][n_ccd][amp],axis=0))
                        #if "FitDC" in id_ME:
                        #            print(id_ME)
                        #            print(bad_images[id_ME][n_ccd][amp])
                    elif "OVS" in ME_type and ME_type in id_ME:
                        bad_images_per_type["OVS"][n_ccd][amp].append(1-np.mean(bad_images[id_ME][n_ccd][amp],axis=0))


    ############# PLOTS ##############
    #################################

    ### Qtest summary plot per ME type
    ### Averages the all the ME failed qtests per type of ME
    for ME_type in bad_images_per_type:
        for n_ccd in bad_images_per_type[ME_type]:
            for amp in bad_images_per_type[ME_type][n_ccd]:
                fails = np.mean(bad_images_per_type[ME_type][n_ccd][amp], axis = 0)
                mean, std = np.mean(fails), np.std(fails)
                std = np.median(np.abs(np.median(fails)-fails))
                n_im = np.arange(0,len(fails),1)
                if display:
                    print("Mean, std = ", mean,std)
                    print("Bad Images in {} ccd {} amp {}: ".format(ME_type, n_ccd, amp), n_im[fails>mean+std])
                    print("Number of Bad Images: ", len(n_im[fails>mean+std]))     
                plt.plot(fails, 'o', color="g", label="Good Images")
                plt.plot(n_im[fails>mean+std], fails[fails>mean+std], 'o', color="r", label="Failed Images")
                plt.title("ME Type: {} ccd {} Amplifier: {}".format(ME_type,n_ccd, amp))
                plt.xlabel("image number")
                plt.ylabel("Percentage of failed qtest")
                plt.legend(loc="best")
                plt.ylim(-0.05,1.05)
                plt.savefig(dqmdir+"Qtest_{}_ccd{}_amp{}_run{}_ref{}.png".format(ME_type,n_ccd,amp, n_data, n_ref))
                #plt.show()
                plt.clf()
    
    ### Dictionary ordered by ME/Amp/averaged failed qtest results
    bad_full = {}
    for id_ME in bad_images:
        for n_ccd in bad_images[id_ME]:
            if n_ccd not in bad_full: bad_full[n_ccd] = {} 
            for amp in bad_images[id_ME][n_ccd]:
                if amp not in bad_full[n_ccd]: bad_full[n_ccd][amp] = []
                #print(bad_images[id_ME][amp])
                bad_full[n_ccd][amp].append(1-np.mean(bad_images[id_ME][n_ccd][amp],axis=0))
                """
                plt.plot(1-np.mean(bad_images[id_ME][amp], axis = 0), 'o')
                plt.title(id_ME + "   " + q_test + "     " + amp)
                plt.xlabel("image number")
                plt.ylabel("Percentage of failed qtest")
                plt.show()
                """
    
    ### Full Qtest summary plot
    ### Averages the all the ME failed qtests.
    for n_ccd in bad_full:
        for amp in bad_full[n_ccd]:
            fails = np.mean(bad_full[n_ccd][amp], axis = 0)
            mean, std = np.mean(fails), np.std(fails)
            std = np.median(np.abs(np.median(fails)-fails))
            n_im = np.arange(0,len(fails),1)

            if display:
                print("Mean, std = ", mean,std)
                print("Bad Images: ccd {} amp {} ".format(n_ccd,amp), n_im[fails>mean+std])
                print("Number of Bad Images: ", len(n_im[fails>mean+std]))     

            plt.plot(fails, 'o', color="g", label="Good Images")
            plt.plot(n_im[fails>mean+std], fails[fails>mean+std], 'o' ,color="r", label="Failed Images")
 
            plt.title("CCD {} Amp {}".format(n_ccd, amp))
            plt.xlabel("image number")
            plt.ylabel("Percentage of failed qtest and ME")
            plt.legend(loc="best")
            plt.ylim(-0.05,1.05)
            plt.savefig(dqmdir+"Qtest_full_ccd{}_amp{}_run{}_ref{}.png".format(n_ccd,amp,n_data,n_ref))
            plt.clf()
            #plt.show()

if __name__ == "__main__":
    
    import sys
    import argparse
    from argparse import Action
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument("--me-ref",
            action="store",
            dest="me_ref",
            help="Reference Run file path")

    parser.add_argument("--me-data",
            action="store",
            dest="me_data",
            help="Data Run file path.")

    parser.add_argument("--run-ref",
            action="store",
            dest="run_ref",
            help="Reference Run number ID.")

    parser.add_argument("--run-data",
            action="store",
            dest="run_data",
            help="Reference Run number ID.")

    parser.add_argument("--nstd",
            action="store",
            dest="n_std",
            type=int,
            help="Number of sigma deviations for QTResiduals.")

    parser.add_argument("-o", "--ouptut",
            action="store",
            dest="output",
            default=os.path.abspath("."),
            help="Directory for the outputs (plots and images will be both recorded here).")

    parser.add_argument("--display",
            action="store_true",
            dest="display",
            help="[BOOL] Display info of bad images and Qtests Summary plots")

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    execute_qtests(args.me_data, args.me_ref, args.run_data, args.run_ref, args.output, n_std=args.n_std, display=args.display)

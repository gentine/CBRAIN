from tensorflow.python.summary import event_accumulator
#import numpy as np
import pandas as pd
from os import walk, mkdir
import shutil

#import sys

def create_csv(mypath,outpath):
    try:
        shutil.rmtree(outpath)
    except:
        pass
    try:
        mkdir(outpath)
    except:
        pass
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    for di in dirnames:
        folder = dirpath + '/' + di
        ff = []
        for (dirpath2, dirnames2, filenames2) in walk(folder):
            ff.extend(filenames2)
            break 
        for ff in filenames2:
            if ff.startswith('events'):
                fullfile = folder + '/' + ff
                print(fullfile)
                sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 1,
                      event_accumulator.IMAGES: 1,
                      event_accumulator.AUDIO: 1,
                      event_accumulator.SCALARS: 0,
                      event_accumulator.HISTOGRAMS: 1}
                ea = event_accumulator.EventAccumulator(fullfile, size_guidance=sg)
                ea.Reload()
                scalar_tags = ea.Tags()['scalars']
                df = pd.DataFrame(columns=scalar_tags) # create data frame
                #df[scalar_tags]=1 
                for tag in scalar_tags : # read each tag
                    if tag != "step/sec":
                        events = ea.Scalars(tag)
                        scalars = list(map(lambda x: x.value, events)) # feed into dataframe
                        #scalars = np.array(scalars)
                        df.loc[:, tag] = scalars 
                out_file =  outpath + '/' +  di + '.csv'  
                df.to_csv(out_file)
    
    print("end reading")

if __name__ == '__main__':
#    args = sys.argv
#    inpath = args[1]
#    outpath = args[2]
    mypath = "./logs/"
    outpath = "./csv_files/"
    create_csv(mypath, outpath)
from tensorflow.python.summary import event_accumulator
import numpy as np
import pandas as pd
import sys

def create_csv(inpath, outpath):
    sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 1,
          event_accumulator.IMAGES: 1,
          event_accumulator.AUDIO: 1,
          event_accumulator.SCALARS: 0,
          event_accumulator.HISTOGRAMS: 1}
    ea = event_accumulator.EventAccumulator(inpath, size_guidance=sg)
    ea.Reload()
    scalar_tags = ea.Tags()['scalars']
    df = pd.DataFrame(columns=scalar_tags) # create data frame
    #df[scalar_tags]=1 
    for tag in scalar_tags: # read each tag
        events = ea.Scalars(tag)
        scalars = np.array(map(lambda x: x.value, events)) # feed into dataframe
        df.loc[:, [tag]] = scalars 
    df.to_csv(outpath)

if __name__ == '__main__':
#    args = sys.argv
#    inpath = args[1]
#    outpath = args[2]
    inpath = "./logs/0212_151433_TPHYSTND_NORAD,PHQ_layers_10000_lr_0.00025_ac_leakyrelu_conv_False_locconv_False_vars_TBP,QBP,PS,lat,SOLIN,SHFLX,LHFLX,dTdt_adiabatic,dQdt_adiabatic_batchs_256_loss_mse/events.out.tfevents.1518444878.public-docking-cx-0174.ethz.ch"
    outpath = "./logs/0212_151433_TPHYSTND_NORAD,PHQ_layers_10000_lr_0.00025_ac_leakyrelu_conv_False_locconv_False_vars_TBP,QBP,PS,lat,SOLIN,SHFLX,LHFLX,dTdt_adiabatic,dQdt_adiabatic_batchs_256_loss_mse/"
    
    create_csv(inpath, outpath)
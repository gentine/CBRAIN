
# -*- coding: utf-8 -*-
"""
    Created on Mon May 22, 2017
    
    @author: gentine
    """
from __future__ import print_function
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
import argparse
import sys
import time
import datetime
import os
from optparse import OptionParser
from os import walk

#parser = OptionParser()
#parser.add_option("-f", "--nc_file", dest="nc_file",
#                  help="write report to FILE", metavar="FILE")
#(options, args) = parser.parse_args()

print("Reading Netcdf for Normalization")
mypath = '../SP-CAM/Pritchard_Aquaplanet'
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break
counter = 0
for f in filenames:
    if f.endswith('.nc'):
        fullfile = mypath + '/' + f
        print(fullfile)
        fh       = Dataset(fullfile, mode='r')
        if(counter==0):
            # average across times and longitudes
            # need to retrieve mean and standard deviation of the full dataset first
            PS       = fh.variables['PS'][:]
            ntimes   = PS.shape[0]
            nlats    = PS.shape[1]
            nlons    = PS.shape[2]
            mean_PS  = np.mean(PS.flatten(), axis=0)
            std_PS   = np.std(PS.flatten(), axis=0)
            del PS
            lat       = fh.variables['lat'][:]
            mean_lat  = np.mean(lat, axis=0)
            std_lat   = np.std(lat, axis=0)
            del lat
            SOLIN      = fh.variables['SOLIN'][:]
            mean_SOLIN  = np.mean(SOLIN.flatten(), axis=0)
            std_SOLIN   = np.std(SOLIN.flatten(), axis=0)
            del SOLIN
            QAP      = fh.variables['QAP'][:]
            nlevels  = QAP.shape[1] 
            #QAP      = QAP.reshape(nlevels,ntimes*nlats*nlons)
            mean_QAP = np.mean(QAP, axis=(0,3),keepdims=True)
            std_QAP  = np.std(QAP, axis=(0,3),keepdims=True)
            del QAP
            TAP      = fh.variables['TAP'][:]
            mean_TAP = np.mean(TAP, axis=(0,3),keepdims=True)
            std_TAP  = np.std(TAP, axis=(0,3),keepdims=True)
            del TAP
            QBP      = fh.variables['QBP'][:]
            mean_QBP = np.mean(QBP, axis=(0,3),keepdims=True)
            std_QBP  = np.std(QBP, axis=(0,3),keepdims=True)
            del QBP
            TBP      = fh.variables['TBP'][:]
            mean_TBP = np.mean(TBP, axis=(0,3),keepdims=True)
            std_TBP  = np.std(TBP, axis=(0,3),keepdims=True)
            del TBP
            #OMEGA    = fh.variables['OMEGA'][:]
            #mean_OMEGA = np.mean(OMEGA, axis=1)
            #std_OMEGA = np.std(OMEGA, axis=1)
            #del OMEGA
            SHFLX    = fh.variables['SHFLX'][:]
            mean_SHFLX = np.mean(SHFLX, axis=(0,2),keepdims=True)
            std_SHFLX = np.std(SHFLX, axis=(0,2),keepdims=True)
            del SHFLX
            LHFLX    = fh.variables['LHFLX'][:]
            mean_LHFLX = np.mean(LHFLX, axis=(0,2),keepdims=True)
            std_LHFLX = np.std(LHFLX, axis=(0,2),keepdims=True)
            del LHFLX
            dTdt_adiabatic    = fh.variables['dTdt_adiabatic'][:]
            mean_dTdt_adiabatic = np.mean(dTdt_adiabatic, axis=(0,3),keepdims=True)
            std_dTdt_adiabatic = np.maximum(np.std(dTdt_adiabatic, axis=(0,3),keepdims=True),1e-20)
            del dTdt_adiabatic
            dQdt_adiabatic    = fh.variables['dQdt_adiabatic'][:]
            mean_dQdt_adiabatic = np.mean(dQdt_adiabatic, axis=(0,3),keepdims=True)
            std_dQdt_adiabatic = np.maximum(np.std(dQdt_adiabatic, axis=(0,3),keepdims=True),1e-20)
            del dQdt_adiabatic
             
            SPDT    = fh.variables['SPDT'][:]
            mean_SPDT = np.mean(SPDT, axis=(0,3),keepdims=True)
            std_SPDT = np.maximum(np.std(SPDT, axis=(0,3),keepdims=True),1e-20)
            del SPDT
            SPDQ    = fh.variables['SPDQ'][:]
            mean_SPDQ = np.mean(SPDQ, axis=(0,3),keepdims=True)
            std_SPDQ = np.maximum(np.std(SPDQ, axis=(0,3),keepdims=True),1e-20)
            del SPDQ
            TPHYSTND_NORAD    = fh.variables['TPHYSTND_NORAD'][:]
            mean_TPHYSTND_NORAD = np.mean(TPHYSTND_NORAD, axis=(0,3),keepdims=True)
            std_TPHYSTND_NORAD = np.maximum(np.std(TPHYSTND_NORAD, axis=(0,3),keepdims=True),1e-20)
            del TPHYSTND_NORAD
            PHQ    = fh.variables['PHQ'][:]
            mean_PHQ = np.mean(PHQ, axis=(0,3),keepdims=True)
            std_PHQ = np.maximum(np.std(PHQ, axis=(0,3),keepdims=True),1e-20)
            del PHQ
            
            
        else:
        
            # average across times and longitudes
            # need to retrieve mean and standard deviation of the full dataset first
            PS       = fh.variables['PS'][:]
            ntimes   = PS.shape[0]
            nlats    = PS.shape[1]
            nlons    = PS.shape[2]
            mean_PS  += np.mean(PS.flatten(), axis=0)
            std_PS   += np.std(PS.flatten(), axis=0)
            del PS
            lat       = fh.variables['lat'][:]
            mean_lat  += np.mean(lat, axis=0)
            std_lat   += np.std(lat, axis=0)
            del lat
            SOLIN      = fh.variables['SOLIN'][:]
            mean_SOLIN  += np.mean(SOLIN.flatten(), axis=0)
            std_SOLIN   += np.std(SOLIN.flatten(), axis=0)
            del SOLIN
            QAP      = fh.variables['QAP'][:]
            nlevels  = QAP.shape[1] 
            #QAP      = QAP.reshape(nlevels,ntimes*nlats*nlons)
            mean_QAP += np.mean(QAP, axis=(0,3),keepdims=True)
            std_QAP  += np.std(QAP, axis=(0,3),keepdims=True)
            del QAP
            TAP      = fh.variables['TAP'][:]
            mean_TAP += np.mean(TAP, axis=(0,3),keepdims=True)
            std_TAP  += np.std(TAP, axis=(0,3),keepdims=True)
            del TAP
            QBP      = fh.variables['QBP'][:]
            mean_QBP += np.mean(QBP, axis=(0,3),keepdims=True)
            std_QBP  += np.std(QBP, axis=(0,3),keepdims=True)
            del QBP
            TBP      = fh.variables['TBP'][:]
            mean_TBP += np.mean(TBP, axis=(0,3),keepdims=True)
            std_TBP  += np.std(TBP, axis=(0,3),keepdims=True)
            del TBP
            #OMEGA    = fh.variables['OMEGA'][:]
            #mean_OMEGA = np.mean(OMEGA, axis=1)
            #std_OMEGA = np.std(OMEGA, axis=1)
            #del OMEGA
            SHFLX    = fh.variables['SHFLX'][:]
            mean_SHFLX += np.mean(SHFLX, axis=(0,2),keepdims=True)
            std_SHFLX += np.std(SHFLX, axis=(0,2),keepdims=True)
            del SHFLX
            LHFLX    = fh.variables['LHFLX'][:]
            mean_LHFLX += np.mean(LHFLX, axis=(0,2),keepdims=True)
            std_LHFLX += np.std(LHFLX, axis=(0,2),keepdims=True)
            del LHFLX
            dTdt_adiabatic    = fh.variables['dTdt_adiabatic'][:]
            mean_dTdt_adiabatic += np.mean(dTdt_adiabatic, axis=(0,3),keepdims=True)
            std_dTdt_adiabatic += np.maximum(np.std(dTdt_adiabatic, axis=(0,3),keepdims=True),1e-20)
            del dTdt_adiabatic
            dQdt_adiabatic    = fh.variables['dQdt_adiabatic'][:]
            mean_dQdt_adiabatic += np.mean(dQdt_adiabatic, axis=(0,3),keepdims=True)
            std_dQdt_adiabatic += np.maximum(np.std(dQdt_adiabatic, axis=(0,3),keepdims=True),1e-20)
            del dQdt_adiabatic
             
            SPDT    = fh.variables['SPDT'][:]
            mean_SPDT += np.mean(SPDT, axis=(0,3),keepdims=True)
            std_SPDT += np.maximum(np.std(SPDT, axis=(0,3),keepdims=True),1e-20)
            del SPDT
            SPDQ    = fh.variables['SPDQ'][:]
            mean_SPDQ += np.mean(SPDQ, axis=(0,3),keepdims=True)
            std_SPDQ += np.maximum(np.std(SPDQ, axis=(0,3),keepdims=True),1e-20)
            del SPDQ
            TPHYSTND_NORAD    = fh.variables['TPHYSTND_NORAD'][:]
            mean_TPHYSTND_NORAD += np.mean(TPHYSTND_NORAD, axis=(0,3),keepdims=True)
            std_TPHYSTND_NORAD += np.maximum(np.std(TPHYSTND_NORAD, axis=(0,3),keepdims=True),1e-20)
            del TPHYSTND_NORAD
            PHQ    = fh.variables['PHQ'][:]
            mean_PHQ += np.mean(PHQ, axis=(0,3),keepdims=True)
            std_PHQ += np.maximum(np.std(PHQ, axis=(0,3),keepdims=True),1e-20)
            del PHQ
        
        fh.close()
        counter = counter + 1

#mean_in  = np.append([mean_PS],mean_QAP, axis=0)
#mean_in  = np.append(mean_in,mean_TAP, axis=0)
#mean_in  = np.append(mean_in,mean_TBP, axis=0)
#mean_in  = np.append(mean_in,mean_QBP, axis=0)
#mean_in  = np.append(mean_in,mean_dTdt_adiabatic, axis=0)
#mean_in  = np.append(mean_in,mean_dQdt_adiabatic, axis=0)
#mean_in  = np.append(mean_in,[mean_SHFLX], axis=0)
#mean_in  = np.append(mean_in,[mean_LHFLX], axis=0)
#
#std_in  = np.append([std_PS],std_QAP, axis=0)
#std_in  = np.append(std_in,std_TAP, axis=0)
#std_in  = np.append(std_in,std_TBP, axis=0)
#std_in  = np.append(std_in,std_QBP, axis=0)
#std_in  = np.append(std_in,std_dTdt_adiabatic, axis=0)
#std_in  = np.append(std_in,std_dQdt_adiabatic, axis=0)
#std_in  = np.append(std_in,[std_SHFLX], axis=0)
#std_in  = np.append(std_in,[std_LHFLX], axis=0)
try:
    os.remove("mean.nc")
except:
    pass
dataset = Dataset("mean.nc", "w")
level   = dataset.createDimension("level", nlevels)
lats     = dataset.createDimension("lats", nlats)
lon     = dataset.createDimension("lon", 1)
#time    = dataset.createDimension("time", 1)
#time    = dataset.createDimension( 1)
temp    = dataset.createVariable("QAP",np.float32,("level","lats","lon"))
temp[:] = mean_QAP[0,:,:,:]/counter
temp    = dataset.createVariable("TAP",np.float32,("level","lats","lon"))
temp[:] = mean_TAP[0,:,:,:]/counter
temp    = dataset.createVariable("QBP",np.float32,("level","lats","lon"))
temp[:] = mean_QBP[0,:,:,:]/counter
temp    = dataset.createVariable("TBP",np.float32,("level","lats","lon"))
temp[:] = mean_TBP[0,:,:,:]/counter
temp    = dataset.createVariable("dTdt_adiabatic",np.float32,("level","lats","lon"))
temp[:] = mean_dTdt_adiabatic[0,:,:,:]/counter
temp    = dataset.createVariable("dQdt_adiabatic",np.float32,("level","lats","lon"))
temp[:] = mean_dQdt_adiabatic[0,:,:,:]/counter
temp    = dataset.createVariable("SPDT",np.float32,("level","lats","lon"))
temp[:] = mean_SPDT[0,:,:,:]/counter
temp    = dataset.createVariable("SPDQ",np.float32,("level","lats","lon"))
temp[:] = mean_SPDQ[0,:,:,:]/counter
temp    = dataset.createVariable("TPHYSTND_NORAD",np.float32,("level","lats","lon"))
temp[:] = mean_TPHYSTND_NORAD[0,:,:,:]/counter
temp    = dataset.createVariable("PHQ",np.float32,("level","lats","lon"))
temp[:] = mean_PHQ[0,:,:,:]/counter
temp    = dataset.createVariable("SHFLX",np.float32,("lats","lon"))
temp[:] = mean_SHFLX[0,:,:]/counter
temp    = dataset.createVariable("LHFLX",np.float32,("lats","lon"))
temp[:] = mean_LHFLX[0,:,:]/counter
temp    = dataset.createVariable("PS",np.float32)
temp[:] = mean_PS/counter
temp    = dataset.createVariable("lat",np.float32)
temp[:] = mean_lat/counter
temp    = dataset.createVariable("SOLIN",np.float32)
temp[:] = mean_SOLIN/counter
dataset.close()


try:
    os.remove("std.nc")
except:
    pass
dataset = Dataset("std.nc", "w")
level   = dataset.createDimension("level", nlevels)
lats     = dataset.createDimension("lats", nlats)
lon     = dataset.createDimension("lon", 1)
#time    = dataset.createDimension("time", 1)
#time    = dataset.createDimension( 1)
temp    = dataset.createVariable("QAP",np.float32,("level","lats","lon"))
temp[:] = std_QAP[0,:,:,:]/counter
temp    = dataset.createVariable("TAP",np.float32,("level","lats","lon"))
temp[:] = std_TAP[0,:,:,:]/counter
temp    = dataset.createVariable("QBP",np.float32,("level","lats","lon"))
temp[:] = std_QBP[0,:,:,:]/counter
temp    = dataset.createVariable("TBP",np.float32,("level","lats","lon"))
temp[:] = std_TBP[0,:,:,:]/counter
temp    = dataset.createVariable("dTdt_adiabatic",np.float32,("level","lats","lon"))
temp[:] = std_dTdt_adiabatic[0,:,:,:]/counter
temp    = dataset.createVariable("dQdt_adiabatic",np.float32,("level","lats","lon"))
temp[:] = std_dQdt_adiabatic[0,:,:,:]/counter
temp    = dataset.createVariable("SPDT",np.float32,("level","lats","lon"))
temp[:] = std_SPDT[0,:,:,:]/counter
temp    = dataset.createVariable("SPDQ",np.float32,("level","lats","lon"))
temp[:] = std_SPDQ[0,:,:,:]/counter
temp    = dataset.createVariable("TPHYSTND_NORAD",np.float32,("level","lats","lon"))
temp[:] = std_TPHYSTND_NORAD[0,:,:,:]/counter
temp    = dataset.createVariable("PHQ",np.float32,("level","lats","lon"))
temp[:] = std_PHQ[0,:,:,:]/counter
temp    = dataset.createVariable("SHFLX",np.float32,("lats","lon"))
temp[:] = std_SHFLX[0,:,:]/counter
temp    = dataset.createVariable("LHFLX",np.float32,("lats","lon"))
temp[:] = std_LHFLX[0,:,:]/counter
temp    = dataset.createVariable("PS",np.float32)
temp[:] = std_PS/counter
temp    = dataset.createVariable("lat",np.float32)
temp[:] = std_lat/counter
temp    = dataset.createVariable("SOLIN",np.float32)
temp[:] = std_SOLIN/counter
dataset.close()

print("End Reading Netcdf for Normalization")

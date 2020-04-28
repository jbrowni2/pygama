#!/usr/bin/env python3
import os
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('clint.mpl')
from pprint import pprint
import scipy.signal as signal
import itertools

from pygama import DataSet
import pygama.utils as pu
import pygama.analysis.histograms as ph
import pygama.analysis.peak_fitting as pf
import h5py, sys
import pygama.io.lh5 as lh5

from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.processors import *
from pygama.dsp.units import *


def main():
    """
    To get the best energy resolution, we want to explore the possible values
    of our DSP processor list, especially trap filter and RC decay constants.
    Inclusion of dedicated functions for the optimization of the ZAC filter.
    
    Modified by:
    V. D'Andrea
    """
    
    par = argparse.ArgumentParser(description="pygama dsp optimizer")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-d", "--dir", nargs=1, action="store", help="analysis directory")
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-g", "--grid", action=st, help="set DSP parameters to be varied")
    arg("-w", "--window", action=st, help="generate a small waveform file")
    arg("-p", "--process", action=st, help="run DSP processing")
    arg("-f", "--fit", action=st, help="fit outputs to peakshape function")
    arg("-t", "--plot", action=st, help="find optimal parameters & make plots")
    arg("-c", "--case", nargs=1, help="case: 0 = trap filter, 1= ZAC filter")
    arg("-v", "--verbose", action=st, help="set verbose mode")
    args = vars(par.parse_args())
    local_dir = "."
    if args["dir"]: local_dir = args["dir"][0]
    
    case = 0
    if args["case"]: case = int(args["case"][0])
    if case == 0:
        filt = "trap"
        print("Run trap optimization")
    elif case == 1:
        filt = "zac"
        print("Run ZAC optimization")
    else:
        print("Case not valid")
        return
    
    ds = pu.get_dataset_from_cmdline(args, f"{local_dir}/meta/runDB.json", f"{local_dir}/meta/calDB.json")
    #pprint(ds.paths)
    
    d_out = f"{local_dir}/cage"
    try: os.mkdir(d_out)
    except FileExistsError: print ("Directory '%s' already exists" % d_out)
    else: print ("Directory '%s' created" % d_out)
    
    f_tier1 = f"{d_out}/cage_optimizer_raw.h5"
    f_grid = f"{d_out}/{filt}_optimizer_grid.h5"
    f_opt = f"{d_out}/{filt}_optimizer_dsp.h5"
            
    # -- run routines --
    if args["grid"]: set_grid(f_grid,case)
    
    # generate a small single-peak file w/ uncalibrated energy to reanalyze
    if args["window"]: window_ds(ds, f_tier1)
    
    # create a file with DataFrames for each set of parameters
    if args["process"]: process_ds(f_grid, f_opt, f_tier1,case)
    
    # fit all outputs to the peakshape function and find the best resolution
    if args["fit"]: get_fwhm(f_grid, f_opt, case, verbose=args["verbose"])
    
    # show results
    if args["plot"]:
        d_plot = f"{d_out}/plots"
        try: os.mkdir(d_plot)
        except FileExistsError: print ("Directory '%s' already exists" % d_plot)
        else: print ("Directory '%s' created" % d_plot)
        plot_fwhm(f_grid,f_opt,d_plot,case) 
    
    
def set_grid(f_grid,case):
    """
    create grid with set of parameters
    """
    if case==1:
        print("Creation of grid for ZAC optimization")
        sigmas = np.linspace(1, 50, 10, dtype='int16')
        flats =  np.linspace(1, 3, 3, dtype='int16')
        decays =  np.linspace(160, 160, 1, dtype='int16')
        lists = [sigmas, flats, decays]
    if case==0:
        print("Creation of grid for trap optimization")
        e_rises = np.linspace(1, 6, 10, dtype='float')
        e_flats = np.linspace(1.5, 3.5, 3, dtype='float')
        rc_consts = np.linspace(50, 150, 1, dtype='float')
        lists = [e_rises, e_flats, rc_consts]
    
    prod = list(itertools.product(*lists))
    
    if case==1: df = pd.DataFrame(prod, columns=['sigma', 'flat','decay']) 
    if case==0: df = pd.DataFrame(prod, columns=['rise','flat','rc']) 
    print(df)
    df.to_hdf(f_grid, key="pygama_optimization")
    print("Wrote grid file:", f_grid)
    

def window_ds(ds, f_tier1):
    """
    Take a DataSet and window it so that the output file only contains 
    events near the calibration peak at 2614.5 keV.
    """
    print("Creating windowed raw file:",f_tier1)
    f_win = h5py.File(f_tier1, 'w')
    
    raw_dir = ds.config["raw_dir"]
    geds = ds.config["daq_to_raw"]["ch_groups"]["g{ch:0>3d}"]["ch_range"]

    for ged in range(geds[0],geds[1]):
        ged = f"g{ged:0>3d}"
        #if ged!="g060": continue
        count = 0
        for p, d, files in os.walk(raw_dir):
            for f in files:
                if f.endswith(".lh5"):
                    f_raw = h5py.File(f"{raw_dir}/{f}",'r')
                    dset = f_raw[ged]['raw']
                    if count == 0:
                        energies = dset['energy'][()]
                        bl   = dset['baseline'][()]
                        ene  = dset['energy'][()]
                        ievt = dset['ievt'][()]
                        ntr  = dset['numtraces'][()]
                        time = dset['timestamp'][()]
                        wf_max = dset['wf_max'][()]
                        wf_std = dset['wf_std'][()]
                        wf = dset['waveform']['values'][()][()]
                        wf_dt = dset['waveform']['dt'][()][()]
                    else:
                        energies = np.append(energies,dset['energy'][()],axis=0)
                        bl = np.append(bl,dset['baseline'][()],axis=0)
                        ievt = np.append(ievt,dset['ievt'][()],axis=0)
                        ntr = np.append(ntr,dset['numtraces'][()],axis=0)
                        time = np.append(time,dset['timestamp'][()],axis=0)
                        wf_max = np.append(wf_max,dset['wf_max'][()],axis=0)
                        wf_std = np.append(wf_std,dset['wf_std'][()],axis=0)
                        wf = np.append(wf,dset['waveform']['values'][()][()],axis=0)
                        wf_dt = np.append(wf_dt,dset['waveform']['dt'][()][()],axis=0)
                    count += 1
                    
        
        # search for 2.6 MeV peak
        maxe = np.amax(energies)
        h, b, v = ph.get_hist(energies, bins=3500, range=(maxe/4,maxe))
        bin_max = b[np.where(h == h.max())][0]
        min_ene = int(bin_max*0.95)
        max_ene = int(bin_max*1.05)
        hist, bins, var = ph.get_hist(energies, bins=500, range=(min_ene, max_ene))
        print(ged,"Raw energy max",maxe,"histogram max",h.max(),"at",bin_max )
        
        # windowing
        bl_win = bl[(energies>min_ene) & (energies<max_ene)]
        ene_win = energies[(energies>min_ene) & (energies<max_ene)]
        ievt_win = ievt[(energies>min_ene) & (energies<max_ene)]
        ntr_win = ntr[(energies>min_ene) & (energies<max_ene)]
        time_win = time[(energies>min_ene) & (energies<max_ene)]
        wf_max_win = wf_max[(energies>min_ene) & (energies<max_ene)]
        wf_std_win = wf_std[(energies>min_ene) & (energies<max_ene)]
        wf_win = wf[()][(energies>min_ene) & (energies<max_ene)]
        wf_win_dt = wf_dt[()][(energies>min_ene) & (energies<max_ene)]

        # create datasets
        f_win.create_dataset(ged+"/raw/energy",dtype='f',data=ene_win)
        f_win.create_dataset(ged+"/raw/ievt",dtype='i',data=ievt_win)
        f_win.create_dataset(ged+"/raw/baseline",dtype='f',data=bl_win)
        f_win.create_dataset(ged+"/raw/numtraces",dtype='i',data=ntr_win)
        f_win.create_dataset(ged+"/raw/timestamp",dtype='i',data=time_win)
        f_win.create_dataset(ged+"/raw/wf_max",dtype='f',data=wf_max_win)
        f_win.create_dataset(ged+"/raw/wf_std",dtype='f',data=wf_std_win)
        f_win.create_dataset(ged+"/raw/waveform/values",dtype='f',data=wf_win)
        d_dt = f_win.create_dataset(ged+"/raw/waveform/dt",dtype='f',data=wf_win_dt)
        d_dt.attrs['units'] = 'ns'
        f_win.attrs['datatype'] = 'table{energy,ievt,baseline,numtraces,timestamp,wf_max,wf_std,waveform}'
        print("Created datasets",ged+"/raw")
        
    f_win.close()        
    print("wrote file:", f_tier1)
    
    
def process_ds(f_grid, f_opt, f_tier1, case):
    """
    process the windowed raw file 'f_tier1' and create the DSP file 'f_opt'
    
    """
    print("Grid file:",f_grid)
    df_grid = pd.read_hdf(f_grid)
    
    if os.path.exists(f_opt):
        os.remove(f_opt)
    
    # open raw file
    lh5_in = lh5.Store()
    #groups = lh5_in.ls(f_tier1, '*/raw')
    f = h5py.File(f_tier1,'r')
    #print("File info: ",f.keys())
    
    t_start = time.time()
    #for group in groups:
    for group in f.keys():#loop on detectors
        #if group!='g060': continue
        if group == 'g028':
            diff = time.time() - t_start
            tot = diff/5 * len(df_grid) / 60
            tot -= diff / 60
            print(f"Estimated remaining time: {tot:.2f} mins")
        
        print("Detector:",group)
        #data = lh5_in.read_object(group, f_tier1)
        data =  f[group]['raw']
        
        #wf_in = data['waveform']['values'].nda
        #dt = data['waveform']['dt'].nda[0] * unit_parser.parse_unit(data['waveform']['dt'].attrs['units'])
        wf_in = data['waveform']['values'][()]
        dt = data['waveform']['dt'][0] * unit_parser.parse_unit(data['waveform']['dt'].attrs['units'])
        # Set up DSP processing chain -- very minimal
        block = 8 #waveforms to process simultaneously
        proc = ProcessingChain(block_width=block, clock_unit=dt, verbosity=False)
        proc.add_input_buffer("wf", wf_in, dtype='float32')
        wsize = wf_in.shape[1]
        dt0 = data['waveform']['dt'][0]

        # basic processors
        proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
        proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
        proc.add_processor(pole_zero, "wf_blsub", 145*us, "wf_pz")
        
        for i, row in df_grid.iterrows():
            if case==1:
                sigma, flat, decay = row/dt0*1000
                proc.add_processor(zac_filter(wsize, sigma, flat, decay),"wf", f"wf_zac_{i}(101, f)")
                proc.add_processor(np.amax, f"wf_zac_{i}", 1, f"zacE_{i}", signature='(n),()->()', types=['fi->f'])
            if case==0:
                rise, flat, rc = row
                proc.add_processor(trap_norm, "wf_pz", rise*us, flat*us, f"wf_trap_{i}")
                proc.add_processor(asymTrapFilter, "wf_pz", rise*us, flat*us, rc*us, "wf_atrap")
                proc.add_processor(time_point_thresh, "wf_atrap[0:2000]", 0, "tp_0")
                proc.add_processor(np.amax, f"wf_trap_{i}", 1, f"trapEmax_{i}", signature='(n),()->()', types=['fi->f'])
                proc.add_processor(fixed_time_pickoff, f"wf_trap_{i}", "tp_0+(5*us+9*us)", f"trapEftp_{i}")
                proc.add_processor(trap_pickoff, "wf_pz", 1.5*us, 0, "tp_0", "ct_corr")
                
        # Set up the LH5 output
        lh5_out = lh5.Table(size=proc._buffer_len)
        for i, row in df_grid.iterrows():#loop on parameters
            if case==1:
                lh5_out.add_field(f"zacE_{i}", lh5.Array(proc.get_output_buffer(f"zacE_{i}"), attrs={"units":"ADC"}))
            if case==0:
                lh5_out.add_field(f"trapEmax_{i}", lh5.Array(proc.get_output_buffer(f"trapEmax_{i}"), attrs={"units":"ADC"}))
                lh5_out.add_field(f"trapEftp_{i}", lh5.Array(proc.get_output_buffer(f"trapEftp_{i}"), attrs={"units":"ADC"}))
        
        print("Processing:\n",proc)
        proc.execute()
        
        #groupname = group[:group.rfind('/')+1]+"data"
        #groupname = df_key+"/"+group+"/data"
        groupname = group+"/data"
        print("Writing to: " + f_opt + "/" + groupname)
        lh5_in.write_object(lh5_out, groupname, f_opt)
        print("")
    
    #list the datasets of the output file
    data_opt = lh5_in.ls(f_opt)
    #data_opt_0 = lh5_in.ls(f_opt,'opt_0/*')
    data_opt_0 = lh5_in.ls(f_opt,'g024/data/*')
    diff = time.time() - t_start
    print(f"Time to process: {diff:.2f} s")
        

def get_fwhm(f_grid, f_opt, case, verbose=False):
    """
    this code fits the 2.6 MeV peak using the gauss+step function
    and writes new columns to the df_grid "fwhm", "fwhmerr"
    """
    print("Grid file:",f_grid)
    print("DSP file:",f_opt)
    df_grid = pd.read_hdf(f_grid)

    f = h5py.File(f_opt,'r')
    for group in f.keys():
        #if group!='g060': continue
        print("Detector:",group)
        data =  f[group]['data']
        
        # declare some new columns for df_grid
        cols = [f"fwhm_{group}", f"fwhmerr_{group}", f"rchi2_{group}"]
        for col in cols:
            df_grid[col] = np.nan
            
        for i, row in df_grid.iterrows():
            try:
                if case==1: energies = data[f"zacE_{i}"][()]
                if case==0: energies = data[f"trapEmax_{i}"][()]
                mean = np.mean(energies)
                bins = 12000
                hE, xE, vE = ph.get_hist(energies,bins,(mean/2,mean*2))
            except:
                print("Energy not find in",group,"and entry",i)
            
            # set histogram centered and symmetric on the peak
            mu = xE[np.argmax(hE)]
            imax = np.argmax(hE)
            hmax = hE[imax]
            idx = np.where(hE > hmax/2) # fwhm
            ilo, ihi = idx[0][0], idx[0][-1]
            sig = (xE[ihi] - xE[ilo]) / 2.355
            idx = np.where(((xE-mu) > -8 * sig) & ((xE-mu) < 8 * sig))
            idx0 = np.where(((xE-mu) > -4.5 * sig) & ((xE-mu) < 4.5 * sig))
            try:
                ilo, ihi = idx[0][0], idx[0][-1]
                ilo0, ihi0 = idx0[0][0], idx0[0][-1]
                xE, hE, vE = xE[ilo:ihi+1], hE[ilo:ihi], vE[ilo:ihi]
            except:
                continue
            
            # set initial guesses for the peakshape function
            hstep = 0
            tau = np.mean(hE[:10])
            bg0 = 1
            x0 = [hmax, mu, sig, bg0, hstep]
            
            try:
                xF, xF_cov = pf.fit_hist(pf.gauss_step, hE, xE, var=vE, guess=x0)
                xF_err = np.sqrt(np.diag(xF_cov))
                
                # goodness of fit
                chisq = []
                for j, h in enumerate(hE):
                    model = pf.gauss_step(xE[j], *xF)
                    diff = (model - h)**2 / model
                    chisq.append(abs(diff))
                # update the master dataframe
                fwhm = xF[2] * 2.355 * 2614.5 / mu
                fwhmerr = xF_err[2] * 2.355 * 2614.5 / mu 
                rchi2 = sum(np.array(chisq) / len(hE))
                
                df_grid.at[i, f"fwhm_{group}"] = fwhm
                df_grid.at[i, f"fwhmerr_{group}"] = fwhmerr
                df_grid.at[i, f"rchi2_{group}"] = rchi2
            except:
                print("Fit not computed for detector",group,"and entry",i)
                
            if verbose:
                plt.cla()
                plt.plot(xE, pf.gauss_step(xE, *xF), c='r', label='peakshape')
                gaus, step = pf.gauss_step(xE, *xF, components=True)
                gaus = np.array(gaus)
                step = np.array(step)
                plt.plot(xE, gaus, ls="--", lw=2, c='g', label="gaus")
                plt.plot(xE, step, ls='--', lw=2, c='m', label='step + bg')
                plt.plot(xE[1:], hE, lw=1, c='b', label=f"data {group}")
                plt.xlabel(f"ADC channels", ha='right', x=1)
                plt.ylabel("Counts", ha='right', y=1)
                plt.legend(loc=2, fontsize=10,title=f"FWHM = {fwhm:.2f} $\pm$ {fwhmerr:.2f} keV")
                plt.show()
                
                
            # write the updated df_grid to the output file.  
            if not verbose:
                df_grid.to_hdf(f_grid, key="pygama_optimization")

    if not verbose:
        print("Update grid file:",f_grid,"with detector",group)
        print(df_grid)
            
def plot_fwhm(f_grid,f_opt,d_plot,case):
    """
    select the best energy resolution, plot best result fit and fwhm vs parameters
    """
    print("Grid file:",f_grid)
    df_grid = pd.read_hdf(f_grid)
    f = h5py.File(f_opt,'r')
    for group in f.keys():
        #if group!='g060': continue
        d_det = f"{d_plot}/{group}"
        try: os.mkdir(d_det)
        except FileExistsError: print ("Directory '%s' already exists" % d_det)
        else: print ("Directory '%s' created" % d_det)

        data =  f[group]['data']
        print("Detector:",group)
        # find fwhm minimum values
        try:
            df_grid = df_grid.loc[(df_grid[f"rchi2_{group}"]<20)&(df_grid[f"fwhm_{group}"]>0)]
            minidx = df_grid[f'fwhm_{group}'].idxmin()
            df_min = df_grid.loc[minidx]
            print("Best parameters:\n",df_min)
            #plot best result fit
            if case==0: energies = data[f"trapEmax_{minidx}"][()]
            if case==1: energies = data[f"zacE_{minidx}"][()]
            mean = np.mean(energies)
            bins = 12000
            hE, xE, vE = ph.get_hist(energies,bins,(mean/2,mean*2))
            mu = xE[np.argmax(hE)]
            hmax = hE[np.argmax(hE)]
            idx = np.where(hE > hmax/2)
            ilo, ihi = idx[0][0], idx[0][-1]
            sig = (xE[ihi] - xE[ilo]) / 2.355
            idx = np.where(((xE-mu) > -8 * sig) & ((xE-mu) < 8 * sig))
            ilo, ihi = idx[0][0], idx[0][-1]
            xE, hE, vE = xE[ilo:ihi+1], hE[ilo:ihi], vE[ilo:ihi]
            x0 = [hmax, mu, sig, 1, 0]
            xF, xF_cov = pf.fit_hist(pf.gauss_step, hE, xE, var=vE, guess=x0)
            xF_err = np.sqrt(np.diag(xF_cov))
            fwhm = xF[2] * 2.355 * 2614.5 / mu
            fwhmerr = xF_err[2] * 2.355 * 2614.5 / mu 
            plt.plot(xE, pf.gauss_step(xE, *xF), c='r', label='peakshape')
            gaus, step = pf.gauss_step(xE, *xF, components=True)
            gaus = np.array(gaus)
            step = np.array(step)
            plt.plot(xE, gaus, ls="--", lw=2, c='g', label="gaus")
            plt.plot(xE, step, ls='--', lw=2, c='m', label='step + bg')
            plt.plot(xE[1:], hE, lw=1, c='b', label=f"data {group}")
            plt.xlabel(f"ADC channels", ha='right', x=1)
            plt.ylabel("Counts", ha='right', y=1)
            plt.legend(loc=2, fontsize=10,title=f"FWHM = {fwhm:.2f} $\pm$ {fwhmerr:.2f} keV")
            if case==1: plt.savefig(f"{d_det}/Fit_{group}-zac.pdf")
            if case==0: plt.savefig(f"{d_det}/Fit_{group}-trap.pdf")
            plt.cla()
        except: continue
        if case==1:
            #try:
            sigma, flat, decay = df_min[:3]
            # 1. vary the sigma cusp
            df_sigma = df_grid.loc[(df_grid.flat==flat)&(df_grid.decay==decay)&(df_grid.decay==decay)]
            x, y, err =  df_sigma['sigma'], df_sigma[f'fwhm_{group}'], df_sigma[f'fwhmerr_{group}']
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("Sigma Cusp ($\mu$s)", ha='right', x=1)
            plt.ylabel(r"FWHM (keV)", ha='right', y=1)
            plt.savefig(f"{d_det}/FWHM_vs_Sigma_{group}-zac.pdf")
            plt.cla()
            # 2. vary the flat time
            df_flat = df_grid.loc[(df_grid.sigma==sigma)&(df_grid.decay==decay)]
            x, y, err =  df_flat['flat'], df_flat[f'fwhm_{group}'], df_flat[f'fwhmerr_{group}']
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("Flat Top ($\mu$s)", ha='right', x=1)
            plt.ylabel("FWHM (keV)", ha='right', y=1)
            plt.savefig(f"{d_det}/FWHM_vs_Flat_{group}-zac.pdf")
            plt.cla() 
            # 3. vary the rc constant
            df_decay = df_grid.loc[(df_grid.sigma==sigma)&(df_grid.flat==flat)]
            x, y, err =  df_decay[f'decay'], df_decay[f'fwhm_{group}'], df_decay[f'fwhmerr_{group}']
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("Decay constant ($\mu$s)", ha='right', x=1)
            plt.ylabel(r"FWHM (keV)", ha='right', y=1)
            plt.savefig(f"{d_det}/FWHM_vs_Decay_{group}-zac.pdf")
            plt.cla()
            #except:
            #print("")
        if case==0:
            rise, flat, rc = df_min[:3]
            # 1. vary the rise time
            df_rise = df_grid.loc[(df_grid.flat==flat)&(df_grid.rc==rc)]
            x, y, err =  df_rise['rise'], df_rise[f'fwhm_{group}'], df_rise[f'fwhmerr_{group}']
            #plt.plot(x,y,".b")
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("Ramp time ($\mu$s)", ha='right', x=1)
            plt.ylabel(r"FWHM (kev)", ha='right', y=1)
            # plt.ylabel(r"FWHM", ha='right', y=1)
            plt.savefig(f"{d_det}/FWHM_vs_Rise_{group}-trap.pdf")
            plt.cla()
            
            # 2. vary the flat time
            df_flat = df_grid.loc[(df_grid.rise==rise)&(df_grid.rc==rc)]
            x, y, err =  df_flat['flat'], df_flat[f'fwhm_{group}'], df_flat[f'fwhmerr_{group}']
            #plt.plot(x,y,'.b')
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("Flat time ($\mu$s)", ha='right', x=1)
            plt.ylabel("FWHM (keV)", ha='right', y=1)
            plt.savefig(f"{d_det}/FWHM_vs_Flat_{group}-trap.pdf")
            plt.cla() 
            # 3. vary the rc constant
            df_rc = df_grid.loc[(df_grid.rise==rise)&(df_grid.flat==flat)]
            x, y, err =  df_rc['rc'], df_rc[f'fwhm_{group}'], df_rc[f'fwhmerr_{group}']
            #plt.plot(x,y,'.b')
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("RC constant ($\mu$s)", ha='right', x=1)
            plt.ylabel(r"FWHM (keV)", ha='right', y=1)
            plt.savefig(f"{d_det}/FWHM_vs_RC_{group}-trap.pdf")
            plt.cla()
        
                

    
if __name__=="__main__":
    main()

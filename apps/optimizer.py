
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
    arg("-d", "--dir", nargs=1, action="store", help="data directory")
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-g", "--grid", action=st, help="set DSP parameters to be varied")
    arg("-w", "--window", action=st, help="generate a small waveform file")
    arg("-p", "--process", action=st, help="run DSP processing")
    arg("-f", "--fit", action=st, help="fit outputs to peakshape function")
    arg("-t", "--plot", action=st, help="find optimal parameters & make plots")
    arg("-z", "--zac", action=st, help="optimize ZAC filter")
    arg("-v", "--verbose", action=st, help="set verbose mode")
    args = vars(par.parse_args())
    data_dir = "."
    if args["dir"]: data_dir = args["dir"][0]
    zac = 0
    if args["zac"]: zac = 1
    
    raw_file = f"{data_dir}/tier1/pgt_longtrace_run0117-20200110-105115-calib_raw.lh5"
    
    ds = pu.get_dataset_from_cmdline(args, "meta/runDB.json", "meta/calDB.json")
    #pprint(ds.paths)
    
    d_out = "cage"
    try:
        os.mkdir(d_out)
    except FileExistsError:
        print ("Directory '%s' already exists" % d_out)
    else:
        print ("Directory '%s' created" % d_out)

    f_tier1 = f"{d_out}/cage_optimizer_raw.h5"
    if zac==1:
        print("Run ZAC filter Optimization")
        f_grid = f"{d_out}/zac_optimizer_grid.h5"
        f_opt = f"{d_out}/zac_optimizer_dsp.h5"
    else:
        print("Run Trap filter Optimization")
        f_grid = f"{d_out}/cage_optimizer_grid.h5"
        f_opt = f"{d_out}/cage_optimizer_dsp.h5"
        
    # -- run routines --
    if args["grid"]: set_grid(f_grid,zac)

    # generate a small single-peak file w/ uncalibrated energy to reanalyze
    if args["window"]: window_ds(ds, raw_file, f_tier1)

    # create a file with DataFrames for each set of parameters
    if args["process"]: process_ds(ds, f_grid, f_opt, f_tier1,zac)

    # fit all outputs to the peakshape function and find the best resolution
    if args["fit"]: get_fwhm(f_grid, f_opt, zac,verbose=args["verbose"])

    # show results
    if args["plot"]:
        d_plot = f"{d_out}/plots"
        try:
            os.mkdir(d_plot)
        except FileExistsError:
            print ("Directory '%s' already exists" % d_plot)
        else:
            print ("Directory '%s' created" % d_plot)
        plot_fwhm(f_grid,f_opt,d_plot,zac) 
    
    
def set_grid(f_grid,zac):
    """
    create grid with set of parameters
    """
    if zac==1:
        print("Creation of grid for ZAC optimization")
        #lenghts = np.arange(57, 57.5, 1)
        sigmas = np.arange(10, 51, 10)
        flats = np.arange(1.0, 3.6, 0.5)
        rc_consts = np.arange(160, 161, 5)
        lists = [sigmas, flats, rc_consts]
    else:
        print("Creation of grid for trap optimization")
        # # this is pretty ambitious, but maybe doable -- 3500 entries
        e_rises = np.arange(1, 6, 0.2)
        e_flats = np.arange(0.5, 4, 1)
        rc_consts = np.arange(50, 150, 10)
        
        lists = [e_rises, e_flats, rc_consts]
    
    prod = list(itertools.product(*lists))
    
    if zac==1: df = pd.DataFrame(prod, columns=['sigma', 'flat','decay']) 
    else: df = pd.DataFrame(prod, columns=['rise','flat','rc']) 
    print(df)
    df.to_hdf(f_grid, key="pygama_optimization")
    print("Wrote grid file:", f_grid)
    
    
def window_ds(ds, raw_file, f_tier1):
    """
    Take a DataSet and window it so that the output file only contains 
    events near the calibration peak at 2614.5 keV.
    """
    print("Creating windowed raw file")
    
    #for run in ds.runs:

    conf = ds.config["daq_to_raw"]
    print("conf",conf)
    #read existing raw file
    f = h5py.File(raw_file,'r')
    #create new h5py file
    f_win = h5py.File(f_tier1, 'w')
    print(f_tier1)
    for ged in f.keys():
        try: 
            dset = f[ged]['raw']
            #print("key: ",ged,"Data info: ",dset.keys())
        except:
            print("Not find raw key in:",ged)
            continue
        
        try:
            energies = dset['energy'][()]
            maxe = np.amax(energies)
            h, b, v = ph.get_hist(energies, bins=3500, range=(maxe/4,maxe))
            bin_max = b[np.where(h == h.max())][0]
            min_ene = int(bin_max*0.95)
            max_ene = int(bin_max*1.05)
            hist, bins, var = ph.get_hist(energies, bins=500, range=(min_ene, max_ene))
            print(ged,"Raw energy max",maxe,"histogram max",h.max(),"at",bin_max )
        except:
            print("Maximum not find in:",ged)
            continue

        #create dataset for windowed file
        try:
            bl_win = dset['baseline'][(energies>min_ene) & (energies<max_ene)]
            ene_win = dset['energy'][(energies>min_ene) & (energies<max_ene)]
            ievt_win = dset['ievt'][(energies>min_ene) & (energies<max_ene)]
            ntr_win = dset['numtraces'][(energies>min_ene) & (energies<max_ene)]
            time_win = dset['timestamp'][(energies>min_ene) & (energies<max_ene)]
            wf_max_win = dset['wf_max'][(energies>min_ene) & (energies<max_ene)]
            wf_std_win = dset['wf_std'][(energies>min_ene) & (energies<max_ene)]
            wf_win = dset['waveform']['values'][()][(energies>min_ene) & (energies<max_ene)]
            wf_win_dt = dset['waveform']['dt'][()][(energies>min_ene) & (energies<max_ene)]
        except:
            print("Windowing error in:",ged)
            
        try:
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
            print("Created datasets",ged+"/raw")
        except:
            print("Problem in datasets creation in",ged)
            continue

    f_win.close()        
    print("wrote file:", f_tier1)


def process_ds(ds, f_grid, f_opt, f_tier1, zac):
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

    #t_start = time.time()
    #for i, row in df_grid.iterrows():#loop on parameters
    # estimate remaining time in scan
    #if i == 4:
    #diff = time.time() - t_start
    #tot = diff/5 * len(df_grid) / 60
    #tot -= diff / 60
    #print(f"Estimated remaining time: {tot:.2f} mins")
    #print("")
    #if zac==1:
    #sigma, flat, decay = row
    #print(f"Row {i}/{len(df_grid)},  sigma {sigma}  flat {flat}  decay {decay}")
    #else:
    #rise, flat, rc = row
    #print(f"Row {i}/{len(df_grid)},  rise {rise}  flat {flat}  rc {rc}")
    #df_key = f"opt_{i}"
    for group in f.keys():#loop on detectors
        if group!='g060': continue
        #for group in groups:
        #print("Processing: " + f_tier1 + '/' + group)
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

        #basic processors
        proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
        proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
        proc.add_processor(pole_zero, "wf_blsub", 145*us, "wf_pz")

        for i, row in df_grid.iterrows():#loop on parameters
            if zac==1:
                sigma, flat, decay = row
                proc.add_processor(zac_filter, "wf", sigma*us, flat*us, decay*us, f"wf_zac_{i}(101, f)")
                proc.add_processor(np.amax, f"wf_zac_{i}", 1, f"zacE_{i}", signature='(n),()->()', types=['fi->f'])
            else:
                rise, flat, rc = row
                proc.add_processor(trap_norm, "wf_pz", rise*us, flat*us, f"wf_trap_{i}")
                #proc.add_processor(asymTrapFilter, "wf_pz", rise*us, flat*us, rc*us, "wf_atrap")
                #proc.add_processor(time_point_thresh, "wf_atrap[0:2000]", 0, "tp_0")
                proc.add_processor(np.amax, f"wf_trap_{i}", 1, f"trapEmax_{i}", signature='(n),()->()', types=['fi->f'])
                #proc.add_processor(time_point_thresh, "wf_atrap[0:2000]", 0, "tp_0")
                #proc.add_processor(fixed_time_pickoff, "wf_trap", "tp_0+5*us+9*us", "trapEftp")
                #proc.add_processor(trap_pickoff, "wf_pz", rise*us, flat, "tp_0", "ct_corr")
                        
        # Set up the LH5 output
        lh5_out = lh5.Table(size=proc._buffer_len)
        for i, row in df_grid.iterrows():#loop on parameters
            if zac==1:
                lh5_out.add_field(f"zacE_{i}", lh5.Array(proc.get_output_buffer(f"zacE_{i}"), attrs={"units":"ADC"}))
            else:
                lh5_out.add_field(f"trapEmax_{i}", lh5.Array(proc.get_output_buffer(f"trapEmax_{i}"), attrs={"units":"ADC"}))
                #lh5_out.add_field("trapEftp", lh5.Array(proc.get_output_buffer("trapEftp"), attrs={"units":"ADC"}))
        
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
    print("Optimization groups:",data_opt)
    print("Optimization sub-groups:",data_opt_0)


def get_fwhm(f_grid, f_opt, zac, verbose=False):
    """
    this code fits the 2.6 MeV peak using the peakshape function (same as in
    calibration.py) and writes a new column to df_grid, "fwhm".
    """
    print("Grid file:",f_grid)
    print("DSP file:",f_opt)
    df_grid = pd.read_hdf(f_grid)

    f = h5py.File(f_opt,'r')
    for group in f.keys():
        if group!='g060': continue
        print("Detector:",group)
        data =  f[group]['data']
        
        # declare some new columns for df_grid
        cols = [f"fwhm_{group}", f"fwhmerr_{group}", f"rchi2_{group}"]
        for col in cols:
            df_grid[col] = np.nan
            
        for i, row in df_grid.iterrows():
            try:
                if zac==1: energies = data[f"zacE_{i}"][()]
                else: energies = data[f"trapEmax_{i}"][()]
                mean = np.mean(energies)
                bins = 12000
                hE, xE, vE = ph.get_hist(energies,bins,(mean/2,mean*2))
                #ph.plot_hist(hE,xE,label=group,show_stats=True)
                plt.show()
            except:
                print("Energy not find in",group)
            
            
            # shift the histogram to be roughly centered at 0 and symmetric
            mu0 = xE[np.argmax(hE)]
            #xE -= mu0
            imax = np.argmax(hE)
            hmax = hE[imax]
            idx = np.where(hE > hmax/2) # fwhm
            ilo, ihi = idx[0][0], idx[0][-1]
            sig = (xE[ihi] - xE[ilo]) / 2.355
            idx = np.where(((xE-mu0) > -8 * sig) & ((xE-mu0) < 8 * sig))
            idx0 = np.where(((xE-mu0) > -4.5 * sig) & ((xE-mu0) < 4.5 * sig))
            try:
                ilo, ihi = idx[0][0], idx[0][-1]
                ilo0, ihi0 = idx0[0][0], idx0[0][-1]
                xE = xE[ilo:ihi+1]
                hE, vE = hE[ilo:ihi], vE[ilo:ihi]
            except:
                xE=xE
            
            #ph.plot_hist(hE,xE,label=group,show_stats=True)    
            #plt.plot(xE, hE, ls='steps', c='r', lw=3)
            #plt.show()
                
            # set initial guesses for the peakshape function.  could all be improved
            mu = mu0
            hstep = 0#np.mean(hE[:10])/hmax
            htail = np.mean(hE[:100])/hmax
            tau = np.mean(hE[:10])
            bg0 = 1#np.mean(hE[:10])
            amp = (xE[1]-xE[0])*np.sum(hE[:ihi0-ilo0])
            #x0 = [mu, sig, hstep, htail, tau, bg0, amp]
            x0 = [hmax, mu, sig, bg0, hstep]
            
            #try:
            #xF, xF_cov = pf.fit_hist(pf.radford_peak, hE, xE, var=vE, guess=x0)
            xF, xF_cov = pf.fit_hist(pf.ge_peak, hE, xE, var=vE, guess=x0)
            xF_err = np.sqrt(np.diag(xF_cov))

            # goodness of fit
            chisq = []
            for j, h in enumerate(hE):
                #model = pf.radford_peak(xE[j], *xF)
                model = pf.ge_peak(xE[j], *xF)
                diff = (model - h)**2 / model
                chisq.append(abs(diff))
            print("Fit guess:",x0)
            print("Fit results:",xF)
            # update the master dataframe
            fwhm = xF[2] * 2.355 * 2614.5 / mu0
            fwhmerr = xF_err[2] * 2.355 * 2614.5 / mu0 
            rchi2 = sum(np.array(chisq) / len(hE))
            
            df_grid.at[i, f"fwhm_{group}"] = fwhm
            df_grid.at[i, f"fwhmerr_{group}"] = fwhmerr
            df_grid.at[i, f"rchi2_{group}"] = rchi2
            #if zac==1:
            #sigma, flat, decay = row[:3]
            #label = f"{i} {sigma:.2f} {flat:.2f} {decay:.0f} {fwhm:.2f} {fwhmerr:.2f} {rchi2:.2f}"
            #else:
            #rise, flat, rc = row[:3]
            #label = f"{i} {rise:.2f} {flat:.2f} {rc:.0f} {fwhm:.2f} {fwhmerr:.2f} {rchi2:.2f}"
            #print(label)
            #except:
            #print("Fit not computed")
                
            if verbose:
                #try:
                plt.cla()
                
                # peakshape function
                #plt.plot(xE, pf.radford_peak(xE, *x0), c='orange', label='guess')
                #plt.plot(xE, pf.radford_peak(xE, *xF), c='r', label='peakshape')
                #plt.plot(xE, pf.ge_peak(xE, *x0), c='orange', label='guess')
                plt.plot(xE, pf.ge_peak(xE, *xF), c='r', label='peakshape')
                
                #plt.axvline(mu, c='g')
                
                # plot individual components
                #tail_hi, gaus, bg, step, tail_lo = pf.radford_peak(xE, *xF, components=True)
                gaus, bg, step = pf.ge_peak(xE, *xF, components=True)
                gaus = np.array(gaus)
                step = np.array(step)
                #tail_lo = np.array(tail_lo)
                #plt.plot(xE, gaus*tail_hi, ls="--", lw=2, c='g', label="gauss")
                plt.plot(xE, gaus, ls="--", lw=2, c='g', label="gaus")
                plt.plot(xE, step+bg, ls='--', lw=2, c='m', label='step + bg')
                #plt.plot(xE, tail_lo, ls='--', lw=2, c='k', label='tail_lo')
                
                plt.plot(xE[1:], hE, ls='steps', lw=1, c='b', label=f"data {group}")
                
                plt.xlabel(f"ADC channels", ha='right', x=1)
                plt.ylabel("Counts", ha='right', y=1)
                plt.legend(loc=2, fontsize=10,title=f"FWHM = {fwhm:.2f} $\pm$ {fwhmerr:.2f} keV")
                plt.show()
                #except:
                #   print("")
                
            # write the updated df_grid to the output file.  
            if not verbose:
                df_grid.to_hdf(f_grid, key="/pygama_optimization")
            print("wrote output file")
        print(df_grid)
            
def plot_fwhm(f_grid,f_opt,d_plot,zac):
    """
    """
    print("Grid file:",f_grid)
    df_grid = pd.read_hdf(f_grid)
    f = h5py.File(f_opt,'r')
    for group in f.keys():
        if group!='g060': continue
        print("Detector:",group)
        # find fwhm minimum values
        try:
            df_grid = df_grid.loc[df_grid[f"rchi2_{group}"]<20]
            df_min = df_grid.loc[df_grid[f'fwhm_{group}'].idxmin()]
            print("Best parameters:\n",df_min)
        except: print("")
        if zac==1:
            #try:
            sigma, flat, decay = df_min[:3]
            # 1. vary the sigma cusp
            df_sigma = df_grid.loc[(df_grid.flat==flat)&(df_grid.decay==decay)&(df_grid.decay==decay)]
            x, y, err =  df_sigma['sigma'], df_sigma[f'fwhm_{group}'], df_sigma[f'fwhmerr_{group}']
            plt.errorbar(x,y,err,fmt='o')
            #plt.plot(x,y,'.b')
            plt.xlabel("Sigma Cusp ($\mu$s)", ha='right', x=1)
            plt.ylabel(r"FWHM (keV)", ha='right', y=1)
            plt.savefig(f"{d_plot}/FWHM_vs_Sigma_{group}-zac.pdf")
            plt.cla()
            # 2. vary the flat time
            df_flat = df_grid.loc[(df_grid.sigma==sigma)&(df_grid.decay==decay)]
            x, y, err =  df_flat['flat'], df_flat[f'fwhm_{group}'], df_flat[f'fwhmerr_{group}']
            #plt.plot(x,y,'.b')
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("Flat Top ($\mu$s)", ha='right', x=1)
            plt.ylabel("FWHM (keV)", ha='right', y=1)
            plt.savefig(f"{d_plot}/FWHM_vs_Flat_{group}-zac.pdf")
            plt.cla() 
            # 3. vary the rc constant
            df_decay = df_grid.loc[(df_grid.sigma==sigma)&(df_grid.flat==flat)]
            x, y, err =  df_decay[f'decay'], df_decay[f'fwhm_{group}'], df_decay[f'fwhmerr_{group}']
            #plt.plot(x,y,'.b')
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("Decay constant ($\mu$s)", ha='right', x=1)
            plt.ylabel(r"FWHM (keV)", ha='right', y=1)
            plt.savefig(f"{d_plot}/FWHM_vs_Decay_{group}-zac.pdf")
            plt.cla()
            #except:
            #print("")
        else:
            rise, flat, rc = df_min[:3]
            # 1. vary the rise time
            df_rise = df_grid.loc[(df_grid.flat==flat)&(df_grid.rc==rc)]
            x, y, err =  df_rise['rise'], df_rise[f'fwhm_{group}'], df_rise[f'fwhmerr_{group}']
            #plt.plot(x,y,".b")
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("Ramp time ($\mu$s)", ha='right', x=1)
            plt.ylabel(r"FWHM (kev)", ha='right', y=1)
            # plt.ylabel(r"FWHM", ha='right', y=1)
            plt.savefig(f"{d_plot}/FWHM_vs_Rise_{group}-trap.pdf")
            plt.cla()
            
            # 2. vary the flat time
            df_flat = df_grid.loc[(df_grid.rise==rise)&(df_grid.rc==rc)]
            x, y, err =  df_flat['flat'], df_flat[f'fwhm_{group}'], df_flat[f'fwhmerr_{group}']
            #plt.plot(x,y,'.b')
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("Flat time ($\mu$s)", ha='right', x=1)
            plt.ylabel("FWHM (keV)", ha='right', y=1)
            plt.savefig(f"{d_plot}/FWHM_vs_Flat_{group}-trap.pdf")
            plt.cla() 
            # 3. vary the rc constant
            df_rc = df_grid.loc[(df_grid.rise==rise)&(df_grid.flat==flat)]
            x, y, err =  df_rc['rc'], df_rc[f'fwhm_{group}'], df_rc[f'fwhmerr_{group}']
            #plt.plot(x,y,'.b')
            plt.errorbar(x,y,err,fmt='o')
            plt.xlabel("RC constant ($\mu$s)", ha='right', x=1)
            plt.ylabel(r"FWHM (keV)", ha='right', y=1)
            plt.savefig(f"{d_plot}/FWHM_vs_RC_{group}-trap.pdf")
            plt.cla()
        
                

    
if __name__=="__main__":
    main()

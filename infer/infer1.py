import xarray as xr
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytensor
from pytensor.graph import Apply, Op
import pytensor.tensor as pt
import copy
import queue
import threading
import sys
 

from pyf3d import Fall3DInputFile, YesNo
from source import MeteoSource

from pycompss.api.api import compss_wait_on, compss_barrier, compss_wait_on
from pycompss.api.task import task
from pycompss.api.IO import IO
from pycompss.api.constraint import constraint
from pycompss.api.binary import binary
from pycompss.api.parameter import *



import dask.array as darray
import yaml
import datetime
import re
import os
import glob


import datetime
from pywaf import Api



pytensor.config.blas__ldflags = '-llapack -lblas -lcblas'



def main2(kyou, kyou2, cfg): #, FLAG_NEW_MESSAGE, FLAG_MESSAGE_READ):
    
    print("Hello world",flush=True)

    LUT = cfg['LUT']
    print(cfg)

    print("LUT in main2():", LUT)

    ##### FLUX #####################################################################

    flux_lower = cfg['flux']['priors']['lower']
    flux_upper = cfg['flux']['priors']['upper']

    flux_mu_loc = cfg['flux']['priors']['mu']['loc']
    flux_mu_scale = cfg['flux']['priors']['mu']['scale']

    flux_sigma_loc = cfg['flux']['priors']['sigma']['loc']
    flux_sigma_scale = cfg['flux']['priors']['sigma']['scale']


    ##### HEIGHTS #####################################################################

    height_lower = cfg['height']['priors']['lower']
    height_upper = cfg['height']['priors']['upper']

    height_min = cfg['height']['lut']['min']
    height_max = cfg['height']['lut']['max']
    N_hts =  cfg['height']['lut']['n']

    height_delta = (height_max - height_min)/N_hts


    height_mu_loc =   cfg['height']['priors']['mu']['loc']   #1000.0
    height_mu_scale = cfg['height']['priors']['mu']['scale'] #100.0

    height_sigma_loc = cfg['height']['priors']['sigma']['loc']#200.0
    height_sigma_scale = cfg['height']['priors']['sigma']['scale']#10.0


    ##### PC1 #####################################################################

    pc1_lower = cfg['pc1']['priors']['lower'] #da['model_physics.vertical_turbulence_model'].min().item()
    pc1_upper = cfg['pc1']['priors']['upper'] #da['model_physics.vertical_turbulence_model'].max().item()
    pc1_min = cfg['pc1']['lut']['min'] #da['model_physics.vertical_turbulence_model'].min().item()
    pc1_max = cfg['pc1']['lut']['max'] #da['model_physics.vertical_turbulence_model'].max().item()
    N_pc1 = cfg['pc1']['lut']['n']
    pc1_delta = cfg['pc1']['pc1s'][1] - cfg['pc1']['pc1s'][0] # (pc1_max - pc1_min)/N_pc1 #da['model_physics.vertical_turbulence_model'].values[1] - da['model_physics.vertical_turbulence_model'].values[0]
    pc1_loc = cfg['pc1']['priors']['loc']
    pc1_scale= cfg['pc1']['priors']['scale']

    pc10_lower =( (pc1_lower - pc1_loc)/pc1_scale   )

    pc10_upper = ( (pc1_upper - pc1_loc)/pc1_scale )


    ##### CONC #####################################################################


    sigma_conc_loc = cfg['sigma_conc']['priors']['loc'] #10.0
    sigma_conc_scale = cfg['sigma_conc']['priors']['scale'] #5.0
    sigma_conc_lower= cfg['sigma_conc']['priors']['lower'] #0.0
    sigma_conc0_lower = (sigma_conc_lower - sigma_conc_loc) / sigma_conc_scale
 




    class DynamicLookUpTable:

        def __init__(self, data, kyou, kyou2): #,  FLAG_NEW_MESSAGE, FLAG_MESSAGE_READ):

            self.data = data 

            self.kyou =kyou

            self.kyou2=kyou2

            print("DynamicLookUpTable.__init__ data.shape", data.shape)

        
        def run(self, missing_indices ):
            
            self.kyou.put(missing_indices)

            self.kyou.join() # wait for task_done from main()

           # ... wait for a new message ...
            new_lut = self.kyou2.get(block=True, timeout=None) # block until new message available, with no timeout

            # ... indicate task is done, freeing up main2()
            self.kyou2.task_done()

            return new_lut

    

        def check_and_run_missing(self, i_ps, i_hs, i_ts): 

            #print("check_and_run_missing")
            
            missing_indices = [
                (i_p, i_h,i_t) for 
                (i_p, i_h,i_t) in 
                zip(i_ps, i_hs, i_ts) if 
                np.isnan(self.data[i_p, i_h, i_t]).any()
                ]
            
            if missing_indices:
                
                new_lut = self.run(copy.deepcopy(missing_indices)) # values get popped from this object after passage through the queue!!

                #for (i_p, i_h, i_t), values in zip(missing_indices, missing_values):

                #    self.data[i_p, i_h, i_t, :, :] = values
                self.data = new_lut
                

        def lookup(self, i_ps, i_hs, i_ts):

            self.check_and_run_missing(i_ps, i_hs, i_ts)

            result = self.data[i_ps, i_hs, i_ts,: ,: ]  # [puff, height, turb, stn, time] => [puff, N_i, stn, time]

            return(result)



        




    class fetch_data_for_source2(Op):

        # i_h0, i_h1, i_t0, i_t1
        itypes = [
                pytensor.tensor.dvector, # i_h0
            pytensor.tensor.dvector, # i_h1
            pytensor.tensor.dvector, # i_t0
            pytensor.tensor.dvector, # i_t1
        ]

        otypes = [
            pytensor.tensor.dtensor4,   # data [4, puff, local_id, time]
        ]

        def __init__(self, dynamic_look_up_table):
            
            self.dynamic_look_up_table = dynamic_look_up_table
            #self.counter = 0
            
        
        def perform(self, node, inputs, output_storage):

            i_h0s = inputs[0].astype(int) # N_pufss

            i_h1s = inputs[1].astype(int) # N_puffs

            i_t0s = inputs[2].astype(int) # N_puffs

            i_t1s = inputs[3].astype(int) # N_puffs

            N_puf = len(i_h0s)

            i_hs = np.hstack([i_h0s, i_h0s, i_h1s, i_h1s]) # 4 * N_puffs

            i_ts = np.hstack([i_t0s, i_t1s, i_t0s, i_t1s]) # 4 * N_puffs

            i_ps = np.hstack([range(N_puf)]*4) # 4* N_puffs


            
            result = self.dynamic_look_up_table.lookup(i_ps, i_hs, i_ts) # 4 * N_puffs, N_stns, N_time




            #f00 = fd(ip, i0,j0) 
            #f01 = fd(ip, i0,j1) 
            #f10 = fd(ip, i1,j0) 
            #f11 = fd(ip, i1,j1) 

            f00 = result[ 0*N_puf:1*N_puf, :, :] # [ N_i, N_stns, N_time] => [N_puff, N_stns, N_time]

            f01 = result[ 1*N_puf:2*N_puf, :, :] # [N_i, N_stns, N_time] => [N_puff, N_stns, N_time]
            f10 = result[ 2*N_puf:3*N_puf, :, :] # [N_i, N_stns, N_time] => [N_puff, N_stns, N_time]
            f11 = result[ 3*N_puf:4*N_puf, :, :] # [ N_i, N_stns, N_time] => [N_puff, N_stns, N_time]

            z = output_storage[0]

            #z[0] = np.array([f00, f01, f10, f11]) # 4, N_puff, N_stns, N_time

            z[0] = np.array([f00,f01,f10,f11])

           
            #self.counter = self.counter +1


        def grad(self, inputs, output_grads):
            return [output_grads *0.0]




    dynamic_lookup_table = DynamicLookUpTable(LUT, kyou, kyou2)#, FLAG_NEW_MESSAGE, FLAG_MESSAGE_READ)


    with pm.Model() as grad_model:

        
        fd = fetch_data_for_source2(dynamic_lookup_table)


            
        ###############################################################################
        # HEIGHT NON CENTERED
        ###############################################################################


    
        height_mu0 = pm.Normal(
                        "height_mu0", 
                        mu=0,
                        sigma=1,
                    )

        height_mu = pm.Deterministic(
                        "height_mu", 
                        height_mu0*height_mu_scale + height_mu_loc
                    )


        height_sigma0 = pm.Normal(
                        "height_sigma0", 
                        mu=0,
                        sigma=1,
                    )

        height_sigma = pm.Deterministic(
                        "height_sigma", 
                        height_sigma0*height_sigma_scale + height_sigma_loc
                    )

        height_lower0 = pm.Deterministic(
                        "height_lower0",
                    (height_lower - height_mu)/height_sigma
                    )

        height_upper0 = pm.Deterministic(
                        "height_upper0",
                    (height_upper - height_mu)/height_sigma
                    )  

        height0 = pm.TruncatedNormal(
                        "height0",
                        mu = 0,
                        sigma = 1,
                        lower = height_lower0,
                        upper = height_upper0,
                        shape = cfg['hour']['n']
                    )

        height = pm.Deterministic(
                        "height",
                        height0 * height_sigma + height_mu
                )


            
        ###############################################################################
        # FLUX NON CENTERED
        ###############################################################################


    
        flux_mu0 = pm.Normal(
                        "flux_mu0", 
                        mu=0,
                        sigma=1,
                    )

        flux_mu = pm.Deterministic(
                        "flux_mu", 
                        flux_mu0*flux_mu_scale + flux_mu_loc
                    )


        flux_sigma0 = pm.Normal(
                        "flux_sigma0", 
                        mu=0,
                        sigma=1,
                    )

        flux_sigma = pm.Deterministic(
                        "flux_sigma", 
                        flux_sigma0*flux_sigma_scale + flux_sigma_loc
                    )

        flux_lower0 = pm.Deterministic(
                        "flux_lower0",
                    (flux_lower - flux_mu)/flux_sigma
                    )

        flux_upper0 = pm.Deterministic(
                        "flux_upper0",
                    (flux_upper - flux_mu)/flux_sigma
                    )  

        flux0 = pm.TruncatedNormal(
                        "flux0",
                        mu = 0,
                        sigma = 1,
                        lower = flux_lower0,
                        upper = flux_upper0,
                        shape = cfg['hour']['n']
                    )

        flux = pm.Deterministic(
                        "flux",
                        flux0 * flux_sigma + flux_mu
                )

    

    #    ###############################################################################
    #    # TURBULENCE NON CENTERED
    #    ###############################################################################



        
        pc10 = pm.TruncatedNormal(
                    "pc10", 
                    mu=0, 
                    sigma=1, 
                    lower=pc10_lower, 
                    upper=pc10_upper
                )

        pc1 = pm.Deterministic(
                    "pc1", 
                    pc10*pc1_scale + pc1_loc
                )


        
        ###############################################################################
        # SIGMA_CONC NON CENTERED
        ###############################################################################
            
        sigma_conc0 = pm.TruncatedNormal(
                    "sigma_conc0", 
                    mu=0, 
                    sigma=1, 
                    lower=sigma_conc0_lower, 
                )

        sigma_conc = pm.Deterministic(
                    "sigma_conc", 
                    sigma_conc0*sigma_conc_scale + sigma_conc_loc
                )


        
        ###############################################################################
        # FORWARD MODEL
        ###############################################################################

        #puffs = pm.Data("puffs", range(N_puf))
        
        #surface_conc = np.zeros([N_stn, N_date])


        r_h = (height - height_lower)/height_delta

        d_h = r_h - np.floor(r_h)

        pc1s = pc1 * np.array([1]*cfg['hour']['n'], dtype='float64')

        r_t = (pc1s - pc1_lower)/pc1_delta

        d_t = r_t - np.floor(r_t)

        w11 = (1-d_h)*(1-d_t) 
        w12 = (1-d_h)*d_t
        w21 = d_h*(1-d_t)
        w22 = d_h*d_t
    
        i_h = r_h
        i_t = r_t
    
        i_h0 = i_h + 1-1
        i_h1 = i_h + 2-1
    
        i_t0 = i_t + 1-1
        i_t1 = i_t + 2-1

        result = fd(i_h0, i_h1, i_t0, i_t1) # 4, N_puff, N_stns, N_time

        f00 = result[0] 
        
        f01 = result[1] 
        
        f10 = result[2] 
        
        f11 = result[3]

        interp = (w11[:,None,None]*f00 + 
                 w12[:,None,None]*f01 + 
                 w21[:,None,None]*f10 + 
                w22[:,None,None]*f11 ) # N_puff, N_stns, N_time

        scale= flux[:,None,None] * interp  # N_puff, N_stns, N_time = (N_puff) * (N_puff, N_stns, N_time)

        interp_scale_sum = scale.sum(axis=0)



        sconc = pm.Deterministic("sconc", interp_scale_sum)

        ###############################################################################
        # Likelihood
        ###############################################################################

        #obs_conc = pm.Normal("obs_conc", mu=concs, sigma=sigma_conc, observed=obs)
        


    with grad_model:
        # draw 1000 posterior samples
        idata = pm.sample()#chains=1)#tune=2000, target_accept=0.9)

    print(az.summary(idata, round_to=2))

    idata.to_netcdf("posterior.nc")

    kyou.put(None)
    
    kyou2.put(None)

   



@binary(
    binary = "Fall3d.x",
    args = "ALL {{infile}}",
    working_dir="{{work_dir}}",
    fail_by_exit_value=True
    )
@task(
    infile=FILE_IN
    )
def run_fall3d(work_dir, infile):
    pass




class Fall3DRunGenerator:

    def __init__(self, cfg_file):
        
        print("Loading configuration ...")
        with open(cfg_file) as stream:
            try:
                cfg = yaml.safe_load(stream)
                print("Loaded config file",cfg_file)
                print(cfg)
            except yaml.YAMLError as cfg:
                print(cfg)

        self.cfg = cfg

        self.indices = []

        #print("Initialising config ...")

       # self.initialise_config()

        #print("Initialising paths ...")

       # self.initialise_paths()

        #print("Initialising default Fall3D inoput file ...")

       # self.initialise_fall3d_input_file()

       # print("Initialising meteo source ...")
        
       # self.initialise_meteo_source()

       # print("Standardising and calculating principal components ...")

       # self.initialise_pcs()

       # print("Initialising look up table ...")
        
       # self.initialise_lut()
        
    def initialise_stations(self):

        with open(self.cfg['station']['file']) as f:

            lines = f.readlines()

        lines  = [l.replace("\n","").split(" ")[0] for l in lines]

        self.cfg['station']['n'] = len(lines)

        self.cfg['station']['stations']=lines

        print("---------------------------------------------------------------------------------------")
        print("Stations initialised:")
        print(self.cfg['station'])


    def initialise_dates(self):

        self.cfg['date']['n'] = self.cfg['date']['end_hour'] - self.cfg['date']['start_hour'] + 1 #!NOTE +1 to be inclusive of end hour


        self.cfg['date']['start'] =  datetime.datetime(
                                        year  = self.cfg['date']['year'],
                                        month = self.cfg['date']['month'],
                                        day   = self.cfg['date']['day'],
                                        hour  = self.cfg['date']['start_hour'],
                                    )

        self.cfg['date']['end'] =  self.cfg['date']['start'] +datetime.timedelta(
                                        hours = self.cfg['date']['end_hour'] - self.cfg['date']['start_hour'] 
                                    )

        self.cfg['date']['dates'] = [ self.cfg['date']['start'] + datetime.timedelta(hours=i)  for i in range(self.cfg['date']['n'])]

        print("---------------------------------------------------------------------------------------")
        print("Dates initialised")
        print(self.cfg["date"])


    def initialise_hours(self):

        self.cfg['hour'] = {
                'hours':np.arange(
                                 self.cfg['date']['start_hour'],
                                 self.cfg['date']['end_hour'] + 1   #!NOTE +1 so range is inclusive of the last hour 
                                 )
                            }

        self.cfg['hour']['n'] = len(self.cfg['hour']['hours']) 

        print("---------------------------------------------------------------------------------------")
        print("Hours initialised")
        print(self.cfg['hour'])



    def initialise_heights(self):

        self.cfg['height']['heights'] = np.linspace(
                self.cfg['height']['lut']["min"],
                self.cfg['height']['lut']['max'],
                self.cfg['height']['lut']['n']
            )

        print("---------------------------------------------------------------------------------------")
        print("Height initialised")
        print(self.cfg['height'])



    def initialise_pcs(self):

        ds_all = self.meteo_source.get_meteo_data()

        ds_sub =ds_all.drop_vars(['orog','sr','lsm'])

        ds_sub_mean = ds_sub.mean(['isobaricInhPa','valid_time','latitude','longitude'])

        ds_sub_std = ds_sub.std(['isobaricInhPa','valid_time','latitude','longitude'])

        ds_sub_standardised = (ds_sub - ds_sub_mean)/ds_sub_std

        ds_sub_standardised_stack = ds_sub_standardised.to_stacked_array(
                new_dim="stacked",
                sample_dims=['ensemble']
                )

        ds_pc = copy.deepcopy(ds_sub_standardised_stack.isel(ensemble=0))

        x = darray.array(ds_sub_standardised_stack)

        # u and v are unit vectors
        u, s, v = darray.linalg.svd(x)

        v = v.compute()

        s = s.compute()

        u = u.compute()

        ds_pc.values = v[0]

        ds_pc = (
            ds_pc
                .drop_vars('time')
                .to_unstacked_dataset(dim='stacked')
                .unstack()
            )

        self.ds_pc = xr.merge([
            ds_pc[name]
                .dropna(dim="isobaricInhPa",how='all')
                .squeeze(drop=True)
            for name in ds_pc
            ])

        self.variances = s**2 / (x.shape[0]-1)

        self.explained_variances = self.variances / np.sum(self.variances)

        self.ds_all = ds_all

        self.ds_sub_mean = ds_sub_mean

        self.ds_sub_std = ds_sub_std

        pc1_samples = u.dot(np.diag(s))[:,0]

        pc1_priors = {
                'loc': pc1_samples.mean(),
                'scale': pc1_samples.std()
            }

        self.cfg['pc1']['priors'] = pc1_priors

        self.cfg['pc1']['lut']['max'] = self.cfg['pc1']['priors']['scale']

        self.cfg['pc1']['lut']['min'] = -self.cfg['pc1']['priors']['scale']

        self.cfg['pc1']['priors']['lower'] = self.cfg['pc1']['lut']['min']

        self.cfg['pc1']['priors']['upper'] = self.cfg['pc1']['lut']['max']

        self.cfg['pc1']['pc1s'] = np.linspace(
                self.cfg['pc1']['lut']['min'],
                self.cfg['pc1']['lut']['max'],
                self.cfg['pc1']['lut']['n']
            )

        print("---------------------------------------------------------------------------------------")
        print("PCs initialised")
        print(self.cfg['pc1'])


































    def initialise_config(self):

        self.cfg['date']['n'] = self.cfg['date']['end_hour'] - self.cfg['date']['start_hour'] + 1


        self.cfg['date']['start'] =  datetime.datetime(
                                        year  = self.cfg['date']['year'],
                                        month = self.cfg['date']['month'],
                                        day   = self.cfg['date']['day'],
                                        hour  = self.cfg['date']['start_hour'],
                                    )
                    
        self.cfg['date']['end'] =  self.cfg['date']['start'] +datetime.timedelta(
                                        hours = self.cfg['date']['end_hour'] - self.cfg['date']['start_hour']
                                    )

        self.cfg['date']['dates'] = [ self.cfg['date']['start'] + datetime.timedelta(hours=i)  for i in range(self.cfg['date']['n'])]


        with open(self.cfg['stations']['file']) as f:

            lines = f.readlines()

        lines  = [l.replace("\n","").split(" ")[0] for l in lines]

        self.cfg['stations']['n'] = len(lines)

        self.cfg['stations']['names']=lines

        self.cfg['hours'] = np.arange(
                self.cfg['date']['start_hour'], 
                self.cfg['date']['end_hour']+1
            
                )

        self.cfg['height']['heights'] = np.linspace(
                self.cfg['height']['lut']["min"],
                self.cfg['height']['lut']['max'],
                self.cfg['height']['lut']['n']
            )

        print("#######################################################################################################")
        print("CONFIGURATION SUMMARY")
        print("-------------------------------------------------------------------------------------------------------")
        print("HEIGHTS")
        print(self.cfg['height'])
        print("-------------------------------------------------------------------------------------------------------")
        print("HOURS")
        print(self.cfg['hours'])
        print("-------------------------------------------------------------------------------------------------------")
        print("DATES")
        print(self.cfg['date'])
        print("-------------------------------------------------------------------------------------------------------")
        print("STATIONS")
        print(self.cfg['stations'])
        print("-------------------------------------------------------------------------------------------------------")
        print("")


    def initialise_paths(self):

        self.cfg['base_path'] = os.path.join(
                     self.cfg['base_dir'], 
                     self.cfg['name']
                )

        if not os.path.isdir(self.cfg['base_path']):
        
            os.mkdir(self.cfg['base_path'])



    def initialise_fall3d_input_file(self):

        f3if = Fall3DInputFile.from_file(self.cfg['fall3d']['file'])

        f3if.grid.update({
                'latmax':self.cfg['area']['latmax'],
                'lonmin':self.cfg['area']['lonmin'],
                'latmin':self.cfg['area']['latmin'],
                'lonmax':self.cfg['area']['lonmax']
            })

        f3if.time_utc.update({
                "year":      self.cfg['date']['year'],
                "month":     self.cfg['date']['month'],
                "day":       self.cfg['date']['day'],
                'run_start': self.cfg['date']['start_hour'],
                'run_end':   self.cfg['date']['end_hour']
            })

        f3if.model_output.update({
                'output_track_points_file' : self.cfg['station']['file']
            })

        self.fall3d_input_file = f3if

    def initialise_meteo_source(self):

        self.meteo_source = MeteoSource(self.fall3d_input_file)


    def initialise_pcs_old(self):

        ds_all = self.meteo_source.get_meteo_data()

        ds_sub =ds_all.drop_vars(['orog','sr','lsm'])

        ds_sub_mean = ds_sub.mean(['isobaricInhPa','valid_time','latitude','longitude'])

        ds_sub_std = ds_sub.std(['isobaricInhPa','valid_time','latitude','longitude'])

        ds_sub_standardised = (ds_sub - ds_sub_mean)/ds_sub_std

        ds_sub_standardised_stack = ds_sub_standardised.to_stacked_array(
                new_dim="stacked",
                sample_dims=['ensemble']
                )

        ds_pc = copy.deepcopy(ds_sub_standardised_stack.isel(ensemble=0))

        x = darray.array(ds_sub_standardised_stack)

        # u and v are unit vectors
        u, s, v = darray.linalg.svd(x) 

        v = v.compute()

        s = s.compute()

        u = u.compute()

        ds_pc.values = v[0]

        ds_pc = (
            ds_pc
                .drop_vars('time')
                .to_unstacked_dataset(dim='stacked')
                .unstack()
            )

        self.ds_pc = xr.merge([
            ds_pc[name]
                .dropna(dim="isobaricInhPa",how='all')
                .squeeze(drop=True) 
            for name in ds_pc
            ])

        self.variances = s**2 / (x.shape[0]-1)

        self.explained_variances = self.variances / np.sum(self.variances)

        self.ds_all = ds_all

        self.ds_sub_mean = ds_sub_mean

        self.ds_sub_std = ds_sub_std

        pc1_samples = u.dot(np.diag(s))[:,0]

        pc1_priors = {
                'loc': pc1_samples.mean(),
                'scale': pc1_samples.std()
            }

        self.cfg['pc1']['priors'] = pc1_priors

        self.cfg['pc1']['lut']['max'] = self.cfg['pc1']['priors']['scale']

        self.cfg['pc1']['lut']['min'] = -self.cfg['pc1']['priors']['scale']

        self.cfg['pc1']['priors']['lower'] = self.cfg['pc1']['lut']['min']

        self.cfg['pc1']['priors']['upper'] = self.cfg['pc1']['lut']['max']


        self.cfg['hours'] = np.arange(
                self.cfg['date']['start_hour'],
                self.cfg['date']['end_hour']+1
            )

        self.cfg['N_hrs'] = len(self.cfg['hours'])

        self.cfg['height']['heights'] = np.linspace(
                self.cfg['height']['lut']["min"],
                self.cfg['height']['lut']['max'],
                self.cfg['height']['lut']['n']
            )

        self.cfg['pc1s'] = np.linspace(
                self.cfg['pc1']['lut']['min'],
                self.cfg['pc1']['lut']['max'],
                self.cfg['pc1']['lut']['n']
            )

    def initialise_lut(self):

        print("Initialising look up table")

        self.cfg['lut_path'] = os.path.join(
                self.cfg['base_path'],
                self.cfg['name'] + "_lut.nc"
                )


        if os.path.isfile(self.cfg['lut_path']):

            print("Look up table found, loading ", self.cfg['lut_path'])    

            da_lut = xr.open_dataset(self.cfg['lut_path'])['conc. ground (gm-3)']

        else:

            print("No lookup table found, generating from scratch")
    
            LUT = np.ones((
                        self.cfg['date']['n'], 
                        self.cfg['height']['lut']['n'], 
                        self.cfg['pc1']['lut']['n'], 
                        self.cfg['station']['n'], 
                        self.cfg['date']['n']
                    ))*np.nan

            start_date = datetime.datetime(
                        year =    self.cfg['date']['year'],
                        month =   self.cfg['date']['month'],
                        day =     self.cfg['date']['day'],
                        hour =    self.cfg['date']['start_hour']
                    )

            dates = [start_date + datetime.timedelta(hours=i) for i in range(self.cfg['date']['n'])]


            da_lut = xr.DataArray(
                LUT,
                name =  'conc. ground (gm-3)',
                dims = ['hour', 'height','pc1', 'name', 'date'],
                coords = {
                        'hour':    self.cfg['hour']['hours'],
                        'height':  self.cfg['height']['heights'],
                        'pc1':     self.cfg['pc1']['pc1s'] ,
                        'name':    self.cfg['station']['stations'],
                        'date':    dates
                     })

        self.cfg['N_runs'] =(
               self.cfg['date']['n'] *
               self.cfg['height']['lut']['n'] *
               self.cfg['pc1']['lut']['n'] *
               self.cfg['station']['n'] *
               self.cfg['date']['n']
            )


        
    
        self.cfg['i_run'] = ( ~np.isnan( da_lut.sum(dim=['name','date'], skipna=False) )).sum().item()           
                

        print(
                "Current look up table contains", self.cfg['i_run'], 
                "runs of", self.cfg['N_runs']
                )
       



        da_lut = da_lut.transpose('hour','height','pc1','name','date')

        self.da_lut = da_lut

        da_lut.to_netcdf(self.cfg['lut_path'])

        #print(da_lut)


    def prepare_run(self, i, i_p, i_h, i_pc):


        hour = self.cfg['hour']['hours'][i_p]

        height = self.cfg['height']['heights'][i_h]

        pc1 = self.cfg['pc1']['pc1s'][i_pc]


        ds_meteo =(
                    (pc1 * self.ds_pc) * 
                    self.ds_sub_std.isel(ensemble=0) + 
                    self.ds_sub_mean.isel(ensemble=0)
                )

        ds_meteo = ds_meteo.squeeze(drop=True)

        ds_meteo = xr.merge([
                        ds_meteo,
                        self.ds_all[['orog','sr','lsm']]
                            .isel(valid_time=0)
                            .squeeze(drop=True)
                    ])

        ds_meteo = ds_meteo.drop_vars('time')

        ds_meteo = ds_meteo.rename({"valid_time":"time"})

        
        for name in ds_meteo:
            ds_meteo[name].attrs =  self.ds_all[name].attrs

            if name in ['u','v','u10','v10']:

            #ds_meteo[name].attrs['GRIB_gridType']= 'lambert'
            #ds_meteo[name].attrs['GRIB_gridDefinitionDescription']= 'Lambert conformal '
            #ds_meteo[name].attrs['GRIB_LaDInDegrees'] = 72.0
                ds_meteo[name].attrs['GRIB_LoVInDegrees'] = 324.0
            #ds_meteo[name].attrs['GRIB_DyInMetres']= 2500.0
            #ds_meteo[name].attrs['GRIB_DxInMetres'] = 2500.0
                ds_meteo[name].attrs['GRIB_Latin2InDegrees'] = 72.0
                ds_meteo[name].attrs['GRIB_Latin1InDegrees']= 72.0
           
           
           ##ds[name].attrs['GRIB_latitudeOfSouthernPoleInDegrees'] = 0.0
           # #ds[name].attrs['GRIB_longitudeOfSouthernPoleInDegrees'] = 0.0



        ds_meteo = ds_meteo.transpose('time','isobaricInhPa','latitude','longitude')

        ds_meteo['longitude'] = ds_meteo['longitude'] - 360.0



        run_dir = os.path.join(self.cfg['base_path'], str(i))

        self.indices.append([run_dir, hour, height, pc1])


        if not os.path.exists(run_dir):
            os.mkdir(run_dir)

        f3if_name = str(i) + ".inp"

        meteo_name = str(i) + ".nc"

        surf_conc_name = str(i) + ".res.nc"

        f3if_filepath = os.path.join(run_dir,f3if_name)

        meteo_filepath = os.path.join(run_dir, meteo_name)

        #out_filepath = os.path.join(run_dir, "output.txt")

        surf_conc_filepath = os.path.join(run_dir, surf_conc_name)


        f3if = copy.deepcopy(self.fall3d_input_file)

        f3if.time_utc.update({
                    'run_start':hour,
                })
            
        f3if.source.update({
                    'source_start':str( hour).zfill(2),
                    'source_end': str(hour +1).zfill(2),
                    'height_above_vent' : str(height),
                    'mass_flow_rate': "1000.0" # REMEMBER TO DIVIDE OUTPUT BY 1000!!
                })
            
        f3if.meteo_data.update({
                'meteo_data_file': meteo_filepath,
                'meteo_data_dictionary_file': '/leonardo/home/userexternal/tbarnie0/infer/mnt/aux/CARRA.tbl',
                'meteo_levels_file': '/leonardo/home/userexternal/tbarnie0/infer/mnt/aux/L137_ECMWF.levels',
                'dbs_begin_meteo_data':0,
                'dbs_end_meteo_data': 24
            })

        f3if.model_output.update({
                'output_track_points':YesNo(True),
                'output_track_points_file':"/leonardo/home/userexternal/tbarnie0/infer/mnt/aux/stations2.pts"
            })

        f3if.grid.update({
                'latmax':self.cfg['area']['latmax']-0.03, 
                'lonmin':self.cfg['area']['lonmin']+0.01, 
                'latmin':self.cfg['area']['latmin']+0.01, 
                'lonmax':self.cfg['area']['lonmax']-0.01
            })


        f3if.to_file(f3if_filepath)


        ds_meteo.to_netcdf(meteo_filepath)

        return(run_dir, f3if_filepath)


    def load_files(self, run_dir, hour, height, pc1):

        glob_string = os.path.join(run_dir, "*.SO2.res")

        files = glob.glob(glob_string)

        dfs = []

        for file in files:

            name = re.findall(".*(STA-.*).SO2.res",file)[0]

            df = pd.read_csv(
                file,
                skiprows=7,
                names=[
                    'date',
                    'load ground (kg/m2)',
                    'conc. ground (g/m3)',
                    'conc.PM5 ground (g/m3)',
                    'conc.PM10 ground (g/m3)',
                    'conc.PM20 ground (g/m3)'
                    ],
                sep=r'\s+'
                )

            df['date'] = df['date'].apply(lambda r: datetime.datetime.strptime(r[:-3],r"%d%b%Y_%H"))

            for var in [
                    'load ground (kg/m2)',
                    'conc. ground (g/m3)',
                    'conc.PM5 ground (g/m3)',
                    'conc.PM10 ground (g/m3)',
                    'conc.PM20 ground (g/m3)'
                    ]:
                df[var] = df[var].astype(float)


            df =( df.set_index('date')
                    .reindex(
                        self.cfg['date']['dates'],
                        fill_value=0.0)
                    .reset_index() )

            df['name'] = name

            df['hour'] = hour

            df['height'] = height

            df['pc1'] = pc1

            dfs.append(df)

        ds = pd.concat(dfs)
                                
        
        
        print(ds)        
        
        ds = ds.set_index(['date', 'name','hour', 'height', 'pc1']).to_xarray()

        da = ds[ 'conc. ground (g/m3)']/1000.0

        da = da.rename('conc. ground (gm-3)')

        out_path = os.path.join(run_dir, "conc.nc")

        da.to_netcdf(out_path)
    

        return da


    def process_finished_tasks(self):

        da_runs = [self.da_lut]

        # load data from saved indices
        for ind in self.indices:

            run_dir, hour, height, pc1 = ind
            
            conc_file_path = os.path.join(run_dir, "conc.nc")

            #da_run = xr.open_dataset(conc_file_path)
            da_run = self.load_files(run_dir, hour, height, pc1)

            da_runs.append(da_run)


        print( "Before LUT update",
               (~ np.isnan( self.da_lut.values )).sum(),
               "non NaN values"
                
                )

        # merege with existing LUT and save to disk
        self.da_lut = xr.merge( da_runs )[ 'conc. ground (gm-3)']


       # print("AFTER MERGE",self.da_lut)

        print( "After LUT update",
               (~ np.isnan( self.da_lut.values )).sum(),
               "non NaN values"
                
                )

        self.da_lut.to_netcdf(self.cfg['lut_path'])

        # reset stored indices
        self.indices = []


        #[puff, height, turb, stn, time]
        return self.da_lut.transpose('hour','height','pc1','name','date').values



    def get_cfg(self):

        cfg = copy.deepcopy(self.cfg)

        cfg['LUT'] = copy.deepcopy(self.da_lut.values)

        return(cfg)

    def get_so2_ground_conc(self):

        # initialise the API object ...
        self.api = Api(
                # ... which will store fetched data in the local_storage directory ...
                local_storage="local_storage/"
            )

        # ...the API object will search for data from stations that ...
        self.result = api.get_data(

            # ... lie within this bounding box ...
            minlat = self.cfg['area']['latmin'],
            maxlat = self.cfg['area']['latmax'],
            minlon = self.cfg['area']['lonmin'],
            maxlon = self.cfg['area']['lonmax'],

            # ... and were operational at some point between these two dates ...
            start = self.cfg['date']['start'],
            end = self.cfg['area']['end'],

             # ... for this species:
            species = 'SO2'

             # If the data is present in local_storage, that data will be returned,
             # otherwise, it will fetch the data using the Umhverfisstofnum API
            )

        self.cfg['obs_path'] = os.path.join(
                self.cfg['base_path'],
                self.cfg['name'] + "_obs.nc"
                )



        self.result.to_netcdf( self.cfg['obs_path'] )







def main(cfg_file):
    

 
    fall3d_run_generator = Fall3DRunGenerator(cfg_file) 
    
    fall3d_run_generator.initialise_stations()

    fall3d_run_generator.initialise_dates()

    fall3d_run_generator.initialise_hours()

    fall3d_run_generator.initialise_heights()

    fall3d_run_generator.initialise_fall3d_input_file()

    fall3d_run_generator.initialise_meteo_source()

    fall3d_run_generator.initialise_pcs()

    fall3d_run_generator.initialise_paths()
    
    fall3d_run_generator.initialise_lut()

    #fall3d_run_generator.get_so2_ground_conc()
#    sys.exit()


    kyou = queue.Queue()

    kyou2 = queue.Queue()

    cfg = fall3d_run_generator.get_cfg()

    thread = threading.Thread(target=main2, args=(kyou, kyou2, cfg)) 

    thread.start()

    N_lut = fall3d_run_generator 

    waiting_indices = []
    
    active_tasks = []

    i_run = fall3d_run_generator.cfg['i_run']

    N_runs = fall3d_run_generator.cfg['N_runs']

    for i in range(i_run, N_runs):

        if not active_tasks and not waiting_indices:

            print("---- ", len(waiting_indices), "waiting indices, abnd", len(active_tasks), "active_tasks, waiting for new indices")
            
            waiting_indices = kyou.get(block=True, timeout=None)  

            if waiting_indices is None:

                break


        if waiting_indices:

            print("---- ",len(waiting_indices), "waiting indices left,", len(active_tasks), "active tasks running")

            i_p, i_h, i_pc = waiting_indices.pop()

            run_dir, f3if_filepath = fall3d_run_generator.prepare_run(i, i_p, i_h, i_pc)

            print(run_dir)
            print(f3if_filepath)

            active_tasks.append( run_fall3d( run_dir, f3if_filepath )  )


        if not waiting_indices:

            print("----", len(waiting_indices), "left, waiting for", len(active_tasks), "to finish")

            compss_barrier()

            active_tasks = compss_wait_on(active_tasks)

            new_lut = fall3d_run_generator.process_finished_tasks()

            kyou.task_done()

            kyou2.put(new_lut)

            active_tasks = []

            kyou2.join()


    


if __name__=='__main__':
    
    cfg_file = "/leonardo/home/userexternal/tbarnie0/infer/test_run.yaml"

    main(cfg_file)



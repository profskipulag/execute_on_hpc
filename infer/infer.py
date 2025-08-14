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


def main2(kyou, kyou2, cfg, LUT): #, FLAG_NEW_MESSAGE, FLAG_MESSAGE_READ):

    print("Hello world",flush=True)


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
    pc1_delta =  (pc1_max - pc1_min)/N_pc1 #da['model_physics.vertical_turbulence_model'].values[1] - da['model_physics.vertical_turbulence_model'].values[0]
    pc1_loc = cfg['pc1']['priors']['loc']
    pc1_scale= cfg['pc1']['priors']['scale']

    pc10_lower = (pc1_lower - pc1_loc)/pc1_scale

    pc10_upper =  (pc1_upper - pc1_loc)/pc1_scale


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


        def run(self, missing_indices ):

            self.kyou.put(missing_indices)

            self.kyou.join() # wait for task_done from main()

           # ... wait for a new message ...
            missing_values = self.kyou2.get(block=True, timeout=None) # block until new message available, with no timeout

            # ... indicate task is done, freeing up main2()
            self.kyou2.task_done()

            return missing_values



        def check_and_run_missing(self, i_ps, i_hs, i_ts):

            print("check_and_run_missing")

            missing_indices = [
                (i_p, i_h,i_t) for
                (i_p, i_h,i_t) in
                zip(i_ps, i_hs, i_ts) if
                np.isnan(self.data[i_p, i_h, i_t]).any()
                ]

            if missing_indices:

                missing_values = self.run(copy.deepcopy(missing_indices)) # values get popped from this object after passage through the queue!!

                for (i_p, i_h, i_t), values in zip(missing_indices, missing_values):

                    self.data[i_p, i_h, i_t, :, :] = values



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
                        shape = cfg['N_hrs']
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
                        shape = cfg['N_hrs']
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

        pc1s = pc1 * np.array([1]*cfg['N_hrs'], dtype='float64')

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


    kyou.put(None)

    kyou2.put(None)







@binary(
    binary='/leonardo/home/userexternal/tbarnie0/fall3d/build/bin/Fall3d.x',
    args = "ALL {{infile}}",
    working_dir="{{work_dir}}",
    fail_by_exit_value=True
    )
@task(
    infile=FILE_IN,
    outfile=FILE_OUT
    )
def grepper(work_dir, infile, outfile):
    pass


def load_files(run_dir, hour, height, pc1):

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
            delim_whitespace = True
            )

        df['name'] = name

        df['date'] = df['date'].apply(lambda r: datetime.datetime.strptime(r[:-3],r"%d%b%Y_%H"))

        df['hour'] = hour

        df['height'] = height

        df['pc1'] = pc1

        for var in [
                'load ground (kg/m2)',
                'conc. ground (g/m3)',
                'conc.PM5 ground (g/m3)',
                'conc.PM10 ground (g/m3)',
                'conc.PM20 ground (g/m3)'
                ]:
            df[var] = df[var].astype(float)


        dfs.append(df)

    ds =  pd.concat(dfs).set_index(['date', 'name','hour', 'height', 'pc1']).to_xarray()

    da = ds[ 'conc. ground (g/m3)']

    return da





def main(cfg_file):


    ########### LOAD THE YAML FILE THAT SPECIFIED THE RUN ###############################################

    with open(cfg_file) as stream:
        try:
            cfg = yaml.safe_load(stream)
            print("Loaded config file",cfg_file)
            print(cfg)
        except yaml.YAMLError as cfg:
            print(cfg)



    ##########  LOAD THE FALL3D FILE SPECIFIED BY THE CONFIG ############################################

    f3if = Fall3DInputFile.from_file(cfg['fall3d']['file'])

    f3if.grid.update({
                'latmax':cfg['area']['latmax'],
                'lonmin':cfg['area']['lonmin'],
                'latmin':cfg['area']['latmin'],
                'lonmax':cfg['area']['lonmax']
            })

    f3if.time_utc.update({
                "year":      cfg['date']['year'],
                "month":     cfg['date']['month'],
                "day":       cfg['date']['day'],
                'run_start': cfg['date']['start_hour'],
                'run_end':   cfg['date']['end_hour']
            })
    f3if.model_output.update({
                'output_track_points_file' : cfg['stations']['file']
            })



    ms = MeteoSource(f3if)

    ds_all = ms.get_meteo_data()

    ds_sub =ds_all.drop(['orog','sr','lsm'])

    ds_sub_mean = ds_sub.mean(['isobaricInhPa','valid_time','latitude','longitude'])

    ds_sub_std = ds_sub.std(['isobaricInhPa','valid_time','latitude','longitude'])

    ds_sub_standardised = (ds_sub - ds_sub_mean)/ds_sub_std

    ds_sub_standardised_stack = ds_sub_standardised.to_stacked_array(new_dim="stacked",sample_dims=['ensemble'])

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
        .drop('time')
        .to_unstacked_dataset(dim='stacked')
        .unstack()
    )

    ds_pc = xr.merge([
        ds_pc[name]
        .dropna(dim="isobaricInhPa",how='all')
        .squeeze(drop=True) for name in ds_pc])

    variances = s**2 / (x.shape[0]-1)

    explained_variances = variances / np.sum(variances)

    print("Explained variances (fraction):", explained_variances)


    pc1_samples = u.dot(np.diag(s))[:,0]
    pc1_priors = {
        'loc': pc1_samples.mean(),
        'scale': pc1_samples.std()
    }

    cfg['pc1']['priors'] = pc1_priors


    cfg['pc1']['lut']['max'] = cfg['pc1']['priors']['scale']

    cfg['pc1']['lut']['min'] = -cfg['pc1']['priors']['scale']

    cfg['pc1']['priors']['lower'] = cfg['pc1']['lut']['min']

    cfg['pc1']['priors']['upper'] = cfg['pc1']['lut']['max']

    cfg['hours'] = np.arange(
                cfg['date']['start_hour'],
                cfg['date']['end_hour']
            )

    cfg['N_hrs'] = len(cfg['hours'])

    cfg['height']['heights'] = np.linspace(
                cfg['height']['lut']["min"],
                cfg['height']['lut']['max'],
                cfg['height']['lut']['n']
            )

    cfg['pc1s'] = np.linspace(
                cfg['pc1']['lut']['min'],
                cfg['pc1']['lut']['max'],
                cfg['pc1']['lut']['n']
            )

    with open(cfg['stations']['file']) as f:
        lines = f.readlines()

    lines  = [l.replace("\n","").split(" ")[0] for l in lines]

    stations = lines

    print(stations)

    cfg['date']['n'] = cfg['date']['end_hour'] - cfg['date']['start_hour']

    cfg['stations']['n'] = len(lines)


    LUT = np.ones((
        cfg['date']['n'],
        cfg['height']['lut']['n'],
        cfg['pc1']['lut']['n'],
        cfg['stations']['n'],
        cfg['date']['n']
        ))*np.nan

    start_date = datetime.datetime(
                    year=cfg['date']['year'],
                    month=cfg['date']['month'],
                    day=cfg['date']['day'],
                    hour=cfg['date']['start_hour']
    )

    dates = [start_date + datetime.timedelta(hours=i) for i in range(cfg['N_hrs'])]

    da_LUT = xr.DataArray(
        LUT,
        name =  'conc. ground (g/m3)',
        dims = ['hour', 'height','pc1', 'name', 'date'],
        coords = {
            'hour': cfg['hours'],
            'height':  cfg['height']['heights'],
            'pc1':  cfg['pc1s'] ,
            'name': stations,
            'date': dates
        }
    )

    zeros = np.zeros((
        cfg['stations']['n'],
        cfg['date']['n']
    ))

    da_zeros = xr.DataArray(
            zeros,
            name =  'conc. ground (g/m3)',
            dims = ['name', 'date'] ,
            coords = {
            'name': stations,
            'date': dates
        }
    )



    cfg['N_lut'] =(
        cfg['date']['n'] *
        cfg['height']['lut']['n'] *
        cfg['pc1']['lut']['n'] *
        cfg['stations']['n'] *
        cfg['date']['n']
        )



    kyou = queue.Queue()

    kyou2 = queue.Queue()

    # set main2 running

    t1 = threading.Thread(target=main2, args=(kyou,kyou2, cfg, LUT))

    t1.start()

    current_batch_indices = []

    current_batch_values = []

    current_station_files = []

    total_runs = 0

    base_path = os.path.join("/leonardo/home/userexternal/tbarnie0/infer/mnt/runs", cfg['name'])

    os.mkdir(base_path)


    for i in range(cfg['N_lut']):

        # if we have no forward runs and no indices, then we need more indices

        if not current_batch_values and not current_batch_indices:

            print("Waiting to get next batch of indices")

            current_batch_indices = kyou.get(block=True, timeout=None) # block until new message available, with no timeout

            if current_batch_indices is None:
                break


        # if we still have indices we pop and run them
        if current_batch_indices:


            i_p, i_h, i_pc = current_batch_indices.pop()

            hour = cfg['hours'][i_p]

            height = cfg['height']['heights'][i_h]

            pc1 = cfg['pc1s'][i_pc]


            ds_meteo = (pc1 * ds_pc) * ds_sub_std.isel(ensemble=0) + ds_sub_mean.isel(ensemble=0)

            ds_meteo = ds_meteo.squeeze(drop=True)

            ds_meteo = xr.merge([
                        ds_meteo,
                        ds_all[['orog','sr','lsm']]
                            .isel(valid_time=0)
                            .squeeze(drop=True)
                    ])

            ds_meteo = ds_meteo.drop('time')

            ds_meteo = ds_meteo.rename({"valid_time":"time"})


            for name in ds_meteo:
                ds_meteo[name].attrs = ds_all[name].attrs
                ds_meteo[name].attrs['GRIB_gridType']= 'lambert'
                ds_meteo[name].attrs['GRIB_gridDefinitionDescription']= 'Lambert conformal '
                ds_meteo[name].attrs['GRIB_LaDInDegrees'] = 72.0
                ds_meteo[name].attrs['GRIB_LoVInDegrees'] = 324.0
                ds_meteo[name].attrs['GRIB_DyInMetres']= 2500.0
                ds_meteo[name].attrs['GRIB_DxInMetres'] = 2500.0
                ds_meteo[name].attrs['GRIB_Latin2InDegrees'] = 72.0
                ds_meteo[name].attrs['GRIB_Latin1InDegrees']= 72.0
                #ds[name].attrs['GRIB_latitudeOfSouthernPoleInDegrees'] = 0.0
                #ds[name].attrs['GRIB_longitudeOfSouthernPoleInDegrees'] = 0.0

            ds_meteo = ds_meteo.transpose('time','isobaricInhPa','latitude','longitude')

            ds_meteo['longitude'] = ds_meteo['longitude'] - 360.0

            run_dir = os.path.join(base_path, str(i))

            os.mkdir(run_dir)

            f3if_name = str(i) + ".inp"

            meteo_name = str(i) + ".nc"

            surf_conc_name = str(i) + ".res.nc"

            f3if_filepath = os.path.join(run_dir,f3if_name)

            meteo_filepath = os.path.join(run_dir, meteo_name)

            out_filepath = os.path.join(run_dir, "output.txt")

            surf_conc_filepath = os.path.join(run_dir, surf_conc_name)





            f3if = copy.deepcopy(f3if)

            f3if.time_utc.update({
                    'run_start':hour,
                })

            f3if.source.update({
                    'source_start': str(hour).zfill(2),
                    'source_end': str(hour +1).zfill(2),
                    'height_above_vent' : str(height),
                    'mass_flow_rate': str(1000.0) # REMEMBER TO DIVIDE OUTPUT BY 1000!!
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
                'latmax':cfg['area']['latmax']-0.03,
                'lonmin':cfg['area']['lonmin']+0.01,
                'latmin':cfg['area']['latmin']+0.01,
                'lonmax':cfg['area']['lonmax']-0.01
            })



            f3if.to_file(f3if_filepath)


            ds_meteo.to_netcdf(meteo_filepath)




            print("STARTING TASK",f3if_filepath)
            current_batch_values.append(
               grepper(run_dir, f3if_filepath, surf_conc_filepath)
                )

            current_station_files.append(
                [surf_conc_filepath, run_dir, hour, height, pc1]
            )





        # if we have some forward runs ready and there are no more indices waiting to be processed ...

        if not current_batch_indices:


            print("Iteration number", i,"waiting for", len(current_batch_values), "runs ", total_runs, "/", cfg['N_lut'])

            # ... we wait for all current runs to finish ...

            compss_barrier()

            #current_batch_values = compss_wait_on(current_batch_values)

            current_batch_values = compss_wait_on(current_batch_values)

            print("!!!!!!!!!!!!! FINISHED WAITING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            das = []

            for csf in current_station_files:

                surf_conc_filepath, run_dir, hour, height, pc1 = csf

                da = load_files(run_dir, hour, height, pc1)

                da = da + da_zeros.expand_dims({'hour':[hour],'height':[height], 'pc1':[pc1]})

                das.append(da)

                #print(da)

            das.append(da_LUT)

            da_lut = xr.merge(das, compat='no_conflicts')

            da_lut.to_netcdf(os.path.join(base_path, str(i)+".nc"))

            print(da_lut)

            total_runs = total_runs + len(current_batch_values)


            # ... put

            kyou.task_done()

            #print("Sending current_batch_values back to PyMC Op")
            kyou2.put(current_batch_values)

            current_batch_values = []

            current_station_files = []

            #print("Waiting foir PyMC Op to acknowledge")
            kyou2.join()

    # ...  if for some reason we have run every
    t1.join()

    sys.exit()











        # ... indicate task is done, freeing up main2()










    t1.join()





if __name__=='__main__':
    cfg_file = "/leonardo/home/userexternal/tbarnie0/infer/test_run.yaml"
    main(cfg_file)

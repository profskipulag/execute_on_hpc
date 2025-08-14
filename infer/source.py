from pyf3d import Fall3DInputFile, YesNo, Longitude
import xarray as xr
import os
import re
import glob
import datetime
import numpy as np
import pandas as pd
import cdsapi
import cfgrib
import dask
import copy
import subprocess


class MeteoSource:

    def __init__(self, fall3dinputfile, basedir='/leonardo/home/userexternal/tbarnie0/infer/mnt/archive', num_samples_in_90_degrees=3000):

        self.fall3dinputfile = fall3dinputfile

        self.basedir = basedir

                
        # now we need to specify the resolution
        # this needs to be a number in degrees that is the result of dividing
        # 90 by an integer:
        self.num_samples_in_90_degrees = num_samples_in_90_degrees
        
        self.resolution = 90/self.num_samples_in_90_degrees

        # the subset area for cdsapi is specified like this
        # NOTE! lon is specified -180 - 180 here
        #LONMIN = -23.0
        #   LONMAX = -21.5
        #   LATMIN = 63.7
        #   LATMAX = 64.2
        
        lonmin = fall3dinputfile.grid.lonmin
        
        lonmax = fall3dinputfile.grid.lonmax
        
        latmin = fall3dinputfile.grid.latmin
        
        latmax = fall3dinputfile.grid.latmax
        
        self.area = [latmax, lonmin, latmin, lonmax]

        self.start_year = fall3dinputfile.time_utc.year

        self.start_month = fall3dinputfile.time_utc.month

        self.start_day = fall3dinputfile.time_utc.day

        self.start_hour = fall3dinputfile.time_utc.run_start

        start_date = datetime.datetime(
                        year=self.start_year, 
                        month=self.start_month,
                        day=self.start_day-2,
                    ) + datetime.timedelta(hours = self.start_hour)


        self.end_hour = fall3dinputfile.time_utc.run_end


        end_date = datetime.datetime(
                        year=self.start_year, 
                        month=self.start_month,
                        day=self.start_day,
                    ) + datetime.timedelta(hours = self.end_hour)
        

        
        delta_date = end_date - start_date

        dates = [start_date + datetime.timedelta(days=day) for day in range(delta_date.days+1)]

        self.dates = dates
        
       # self.analysis_files = [
       #     os.path.join(
       #         self.basedir,
       #         d.strftime("%Y%m%d") + 
       #         "__" +
       #         "_".join([str(l) for l in area]) +
       #         "_analysis.nc"
       #     )
       #     
       #     for d in dates ]#

        #self.forecast_files = [
        #    os.path.join(
        #        self.basedir,
        #        d.strftime("%Y%m%d") + 
        #        "__" +
        #        "_".join([str(l) for l in area]) +
        #        "_forecast.nc"
        #    )
        #    
        #    for d in dates ]
     # 


    def get_meteo_data(self):

        dss_analysis = []

        dss_forecast = []

        # check each date to see if we alreday have data for that date and 
        # the current area
        for date  in self.dates:
            print("Fetching meteo for", date)

            # create the base file name ...
            base_file =( 
                        date.strftime("%Y%m%d") + 
                        "__" +
                        "_".join([str(l) for l in self.area])
                        )
            
            # ... the base path is just the storage dir plus the base filename ...
            base_path = os.path.join(self.basedir, base_file)

            # ... with the forecast and analysis files distinguished by a suffix
            # at the end ...
            forecast_file =  base_path + "_forecast.nc"

            analysis_file = base_path + "_analysis.nc"


            # now we check if the forecast file already exists,
            # and if not order the data ...
            if not os.path.isfile(forecast_file):

                print(forecast_file, "not found, ordering from Copernicus Data Store")
                
                self.order_forecast(date, forecast_file)
                
            print(forecast_file, "found")

            ds = xr.open_dataset(forecast_file)
                
            dss_forecast.append(ds)



            # ... and the same for the analysis file - if it doesn't exist, we order it
            if not os.path.isfile(analysis_file):

                print(analysis_file, "not found, ordering from Copernicus Data Store")

                self.order_analysis(date, analysis_file)

            print(analysis_file, "found")

            ds = xr.open_dataset(analysis_file)

            dss_analysis.append(ds)

        ds_analysis = xr.merge(dss_analysis)

        ds_forecast = xr.merge(dss_forecast)

                
        # ... 1st we separate out the constants ...
        ds_constants = ds_analysis[["orog","sr","lsm"]]
        
        ds_analysis = ds_analysis.drop("orog").drop("sr").drop("lsm")
        
        # ... re-arrange ds_forecast so it is indexed on valid_time, and the 
        # start time for the forecast become the ensemble number, for which we 
        # assign an integer:
        
        ds_forecast = xr.concat([
            ds_forecast
                .isel(time=i)
                .isel(step=slice(0,8))
                .swap_dims({"step":"valid_time"})
                .assign_coords(time=[i+1])
                .rename({"time":"ensemble"})
                .drop("step")
                .drop("surface")
            for i in range(len(ds_forecast.time))],
            dim='ensemble')
        
        # ... and we do the same for the analysis ...
        ds_analysis = (
            ds_analysis
                .swap_dims({"time":"valid_time"})
                .expand_dims({"ensemble":[0]})
                .drop("step")
                .drop("surface")
            )
        
        # ... finally we merge them to get an ensemble:
        ds_ensemble = xr.merge([ds_forecast,ds_analysis])
        
        # ... However our enseble consists of a number of forecasts arrange en echelon in
        # ensemble-valid_time space - for any valid_time we only ever have three ensemble values,
        # the rest are nans - we need to drop these nans and compress the three actual values available
        # for any valid_time into three continuous ensembles:
        Valid_times_with_three_ensembles =  [ds_ensemble.isel(valid_time=i).dropna("ensemble") for i in range(len(ds_ensemble.valid_time))]
        
        ds_ensemble = xr.concat([
            t.assign_coords({"ensemble":[0,1,2]}) for t in Valid_times_with_three_ensembles if len(t.ensemble)==3], dim='valid_time')
        
        ds_all = xr.merge([
            ds_ensemble,
            ds_constants.swap_dims({"time":"valid_time"}).drop("step").drop("surface")
        ])

        start_date = datetime.datetime(
                        year=self.start_year, 
                        month=self.start_month,
                        day=self.start_day,
                    ) + datetime.timedelta(hours = self.start_hour)

        end_date = datetime.datetime(
                        year=self.start_year, 
                        month=self.start_month,
                        day=self.start_day,
                    ) + datetime.timedelta(hours = self.end_hour)

        start_date = np.datetime64(start_date)

        end_date = np.datetime64(end_date)
        
        ds_sub = ds_all.sel(valid_time = slice(start_date, end_date))

        return(ds_sub)



    

    def order_forecast(self, date, output_file):

        ds_levels = self.order_forecast_levels(date)

        ds_single = self.order_forecast_single(date)

        ds = xr.merge([ds_levels, ds_single])

        ds.to_netcdf(output_file)

        
        

    def order_analysis(self, date, output_file):

        ds_levels = self.order_analysis_levels(date)

        ds_single = self.order_analysis_single(date)

        ds = xr.merge([ds_levels, ds_single])

        ds.to_netcdf(output_file)
        

    

    def order_analysis_levels(self, date):

        year = date.year

        month = date.month

        day = date.day

        dataset = "reanalysis-carra-pressure-levels"
        
        request = {
            "domain": "west_domain",
            "variable": [
                "geometric_vertical_velocity",
                "geopotential",
                "relative_humidity",
                "u_component_of_wind",
                "v_component_of_wind",
                "temperature"
            ],
            "pressure_level": [
                "600", "700", "750",
                "800", "825", "850",
                "875", "900", "925",
                "950", "1000"
            ],
            "product_type": "analysis",
            "time": [
                "00:00", "03:00", "06:00",
                "09:00", "12:00", "15:00",
                "18:00", "21:00"
            ],
            "year": [year],
            "month": [month],
            "day": [day], 
            'grid':[self.resolution, self.resolution],
            'area': self.area,
            "data_format": "grib"
        }
        
        client = cdsapi.Client()
        
        client.retrieve(dataset, request).download("analysis_levels.grib")

        analysis_levels = xr.merge(cfgrib.open_datasets("analysis_levels.grib"))

        return(analysis_levels)

    def order_forecast_levels(self,date):

        year = date.year

        month = date.month

        day = date.day
        
        dataset = "reanalysis-carra-pressure-levels"
        
        request = {
            "domain": "west_domain",
            "variable": [
                "geometric_vertical_velocity",
                "geopotential",
                "relative_humidity",
                "u_component_of_wind",
                "v_component_of_wind",
                "temperature"
        
            ],
            "pressure_level": [
                "600", "700", "750",
                "800", "825", "850",
                "875", "900", "925",
                "950", "1000"
            ],
            "product_type": "forecast",
            "time": ["00:00", "12:00"],
            "leadtime_hour": [
                "3",
                "6",
                "9",
                "12",
                "15",
                "18",
                "21",
                "24",
                "27",
                "30"
            ],
            "year": [year],
            "month": [month],
            "day": [day],
             'grid':[self.resolution, self.resolution],
            'area': self.area,
            "data_format": "grib"
        }
        
        client = cdsapi.Client()
        
        client.retrieve(dataset, request).download("forecast_levels.grib")

        forecast_levels = xr.merge(cfgrib.open_datasets("forecast_levels.grib"))

        return(forecast_levels)




    def order_analysis_single(self, date):
        
        year = date.year

        month = date.month

        day = date.day
        
        dataset = "reanalysis-carra-single-levels"
        
        request = {
            "domain": "west_domain",
            "level_type": "surface_or_atmosphere",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_relative_humidity",
                "2m_temperature",
                "land_sea_mask",
                "orography",
                "surface_pressure",
                "surface_roughness"
            ],
            "product_type": "analysis",
            "time": [
                "00:00", "03:00", "06:00",
                "09:00", "12:00", "15:00",
                "18:00", "21:00"
            ],
            "year": [year],
            "month": [month],
            "day": [day],
             'grid':[self.resolution,self.resolution],
            'area': self.area,
            "data_format": "grib"
        }
        
        client = cdsapi.Client()
        
        client.retrieve(dataset, request).download("analysis_single.grib")

        analysis_singles = cfgrib.open_datasets("analysis_single.grib")
        
        dss = []
        
        for ds in analysis_singles:
            
            for dim in ["heightAboveGround"]:
                
                if dim  in ds.coords:
                    
                    ds = ds.drop(dim)
        
            dss.append(ds)
        
        analysis_singles = xr.merge(dss)

        return(analysis_singles)


    def order_forecast_single(self, date):
        
        year = date.year

        month = date.month

        day = date.day
                
        dataset = "reanalysis-carra-single-levels"
        
        request = {
            "domain": "west_domain",
            "level_type": "surface_or_atmosphere",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_relative_humidity",
                "2m_temperature",
                "surface_pressure"
            ],
            "product_type": "forecast",
           "time": ["00:00", "12:00"],
            "leadtime_hour": [
                "3",
                "6",
                "9",
                "12",
                "15",
                "18",
                "21",
                "24",
                "27",
                "30"
            ],
            "year": [year],
            "month": [month],
            "day": [day],
            'grid':[self.resolution, self.resolution],
            'area': self.area,
            "data_format": "grib"
        }
        
        client = cdsapi.Client()
        client.retrieve(dataset, request).download("forecast_single.grib")

        forecast_singles = cfgrib.open_datasets("forecast_single.grib")

                
        dss = []
        
        for ds in forecast_singles:
            
            for dim in ["heightAboveGround"]:
                
                if dim  in ds.coords:
                    
                    ds = ds.drop(dim)
        
            dss.append(ds)
        
        forecast_singles = xr.merge(dss)

        return(forecast_singles)
                

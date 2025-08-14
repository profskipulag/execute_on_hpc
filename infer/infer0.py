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





@binary(
    #binary='/leonardo/home/userexternal/tbarnie0/fall3d/build/bin/Fall3d.x',
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






def main():

    current_runs = []


    for i in range(2):

        base_path = "/leonardo/home/userexternal/tbarnie0/infer/mnt/runs"

        run_dir = os.path.join(base_path, str(i))

        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)

        f3if_name = "Example.inp" #str(i) + ".inp"

        #meteo_name = str(i) + ".nc"

        f3if_filepath = os.path.join(run_dir,f3if_name)

        print(f3if_filepath)

        current_runs.append(
                run_fall3d(run_dir, f3if_filepath)
            )

        compss_barrier()

        current_runs = compss_wait_on(current_runs)




if __name__=='__main__':
    main()

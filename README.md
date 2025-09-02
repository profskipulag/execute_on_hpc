To prepare the environment for executing the workflow, we login to Leonardo and install the latest version of the Spack package manager with the latest package files by following the instructions at  https://github.com/spack/spack. Missing Spack package files for e.g. compss (3.3.2), fall3d, cdsapi (0.7.5), pymc (5.22) and pytensor amongast others were created and are provided in the custom dtcv4-repo provided with this DTC. First, activate spack ...

```
. spack/share/spack/setup-env.sh
```
... then clone this repository:

```
git clone https://github.com/profskipulag/execute_on_hpc.git
```
Then set up and install the environment ...

```
cd execute_on_hpc

spack env activate dtcv4-env -p

```
... add the custom repository for packages missing from the default spack install ...

spack repo add dtcv4-repo

```

And finaly, concretize and install

```
spack concretize -f

spack install
```

Then install the steps ...

```
cd ~
git clone https://github.com/profskipulag/ST540102.git
git clone https://github.com/profskipulag/ST540103.git
git clone https://github.com/profskipulag/ST540102.git
```
... and run them like this

```
# Run the first step to fetch the data ...
python ST540102/watch_and_fetch.py
# ... copy the data across for the next step ...
cp  ST540102/dt5402.nc ST540104/dt5402.nc

# ... run the next step ...
ST540103/run.sh
# ... you can visualise the state of the SLURM job with squeue ...
squeue -u tbarnie0 --long # shows the status of the compss job
# ... copy the resulting data to the next step ...
cp ST540103/dt5404.nc ST540104/dt5404.nc

# ... run the next step
python ST540104/infer.py

# ... this last step produces dt5405.nc which can then be displayed using ST5405
```



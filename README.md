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

```
spack repo add dtcv4-repo
```


And finally, concretize and install.

```
spack concretize -f

spack install
```

Now to runj 


./run_infer1.sh
squeue -u tbarnie0 --long

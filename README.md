To prepare the environment for executing the workflow, we login to Leonardo and install the latest version of the Spack package manager with the latest package files by following the instructions at  https://github.com/spack/spack.  Missing Spack package files for compss (3.3.2), fall3d, cdsapi (0.7.5), pymc (5.22) and pytensor were created and are provided in the custom dtcv4-repo provided with this DTC (figure 2). 

. spack/share/spack/setup-env.sh
spack env activate dtcv4-env -p
spack concretize -f
spack install
./run_infer1.sh
squeue -u tbarnie0 --long

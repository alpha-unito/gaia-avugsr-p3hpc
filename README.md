# Gaia AVU-GSR

This repository contains the code for the article "Performance portability via C++ PSTL, SYCL, OpenMP, and HIP: the Gaia AVU-GSR case study".
The project structure is as follows:
- Scripts: This folder contains some *sample* compilation slurm scripts to compile the GaiaSim binary. They use the Makefiles from the directory
 Makefile. examples to compile the source code. These files are merely samples and will not work on other machines. This is the fact that
 each user will need to change the paths of both compilers and SDKs, as well as the framework-specific options, depending on the configuration of their machine.

- Src: This directory contains the source code for the GaiaSim application binary.
- Include: This directory contains common header files for the GaiaSim binary.
- Simulations: This folder contains scipts to submit all jobs three times, for a given CPU-GPU architecture.

## Compile the application.

To compile the code, you may take a look at the `Scripts/<cluster>/comp` folder. In there, you will find compilation sample scripts (that
will references the Makefiles found in Makefile.examples) for each programming framework that has been used on that architecture.

To compile the code, you might take thoose scripts, and change all the paths accordingly, to reflect the local configuration of
your machine. Please refer to this table to match the cluster name to the GPU architecture:

| Cluster name | GPU architecture |
|--------------|------------------|
| CascadeLake  | Nvidia V100s     |
| TeslaT4      | Nvidia T4        |
| Epito        | Nvidia A100      |
| GraceHopper  | Nvidia H100      |
| Setonix      | AMD MI250X       |


## Launch the GAIA AVU-GSR binary
```
./GaiaGsr<...>.x -memGlobal 2 -IDtest 0 -itnlimit 100 
```
Were:
- memGlobal specifies approximately how much memory the system occupies in GB
- IDtest 0 specifies that the test, if run up to convergence, reaches the identity solution
- itnlimit specifies the maximum number of iterations run by LSQR. This number is not reached if confergence is reached before.
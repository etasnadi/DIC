# Differential interference contrast microscopy image reconstruction (OpenCL, cpp)

Based on the paper: Koos et al.: DIC image reconstruction using an energy minimization framework to visualize optical path length distribution (Scientific Reports, 2016)

### Build

Requirements: OpenCV, OpenCL installed on your system. Build it with CMake!

### Run

Usage: ```dicgpu.exe [--str-conf <inline-config>|--file-conf <file-path>, ...]```, where ```inline-config=var1=value-of-var1,...,varn=value-of-varn```

Default values for the variables:

```verbose=1,wAccept=0.025f,wSmooth=0.0125f,direction=135.0f,nIter=20000,locSize=64,kernelSrcPath=dic-rec.cl,input='',output='',platformId=0,deviceId=0```

The ```platformId``` and the ```deviceId``` controls which OpenCL device should be used for the computation. To discover the available devices on your system, run the app with the single parameter ```--get-devinfo``` that gives something like this:

```
0;0;Intel(R) HD Graphics Kabylake Halo GT2 (Intel)
1;0;GeForce GTX 1060 (NVIDIA Corporation)

```

It means that there are 2 devices discovered, the platform id is the first value while the device id is the second one. (The platform id and device id uniquely identifies a computing device.) The third value is the name and the manufacturer of the device, obviously.


```input```: the input file path, ```output```: the output file path, ```kernelSrcPath```: the path of the ```dic-rec.cl``` file.

In the case of the config file: every variable-value pair is in a separate line. Empty lines are allowed while the lines starting with ```#``` are treated as comments and won't be parsed.

You can combine multiple settings. The default settings will be overriden by the ```--str-conf``` or ```--file-conf``` setup.

For example, the following is a valid usage:

```dicrec.exe --file-conf common-settings.conf --str-conf verbose=0 --str-conf nIter=20000,direction=135.0f --str-conf input=examples/bloog.png,output=examples/blood-reconstructed.tif```

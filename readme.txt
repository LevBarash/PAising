
The programs PAisingSSC and PAisingMSC are introduced in the paper:
L.Yu. Barash, M. Weigel, M. Borovsky, W. Janke, L.N. Shchur, GPU accelerated population annealing algorithm
This work is licensed under a Creative Commons Attribution 4.0 International License:
http://creativecommons.org/licenses/by/4.0/

To compile the programs, use Nvidia CUDA compiler (nvcc). For example: 

nvcc -o PAisingSSC PAisingSSC.cu
nvcc -o PAisingMSC PAisingMSC.cu

Optionally, the flags specifying GPU architecture of a particular device can be added. For example:

nvcc -arch=sm_35 -o PAisingSSC PAisingSSC.cu
nvcc -arch=sm_35 -o PAisingMSC PAisingMSC.cu

To make a quick check on your hardware, run the programs with the parameters -s 100
The output files for the PAisingSSC and PAisingMSC programs will be placed in the subdirectories 
"dataSSC_L64_R20000_EqSw100_dB0.005000" and "dataMSC_L64_R20000_EqSw100_dB0.005000" correspondingly.
Compare the output files generated on your hardware with the ones which are saved in the "sample_output" subdirectory.

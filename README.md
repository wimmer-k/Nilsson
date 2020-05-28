# Nilsson model calculations with Python

This code calculates the energy levels in deformed nuclei. The calculations follows the original publication [S. G. Nilsson, Dan. Mat. Fys. Medd. 29 (1955) 1](http://cds.cern.ch/record/212345).
Optionally, the wave function decomposition in to the spherical components, the decoupling parameter, the g-factors, and the spectroscopic factors can also be calculated and displayed.

## Getting started

The program runs with both Python 2.7 and Python 3. It has been developed using Python versions 3.5.2 and 2.7.12. Python 3 is recommended.

### Prerequisites

Required Python packages are [matplotlib](https://matplotlib.org/) and [NumPy](https://numpy.org/) for the display and computing, respectively. Visit these pages to download the packages and for installing instructions. In find to out if the packages are already installed and what version you have try:

```
>>> import matplotlib as mpl
>>> mpl.__version__
>>> import numpy as np 
>>> np.__version__
```

## Running the code

The program only requires one input parameter, N the oscillator shell for which you wish to calculate the Nilsson diagram. For example, to calculate the N=2 shell, use:

```
python nilsson.py -N 2
```

Optionally, you can make the code executable and run it directly

```
chmod +x nilsson.py
./nilsson.py -N 2
```

You will see a matplotlib window opening with the Nilsson level diagram for the chosen shell. Each level is labeled with the asymptotic quantum numbers Omega[N,nz,Lambda] and with a ID reference number.

### Plotting properties

In order to plot the wave function decomposition of a certain oribtal, record its ID number from the diagram and use it with the option "-o" or "--orbital":
```
./nilsson.py -N 2 -o 0
```
will plot the wave function of the 1/2+[220] level as a function of deformation.
The same result is obtained by calling the property choice option "-p" or "--property" and choose "wf" or "wavef"
```
./nilsson.py -N 2 -o 0 -p wf
```

For each of the orbitals you can also plot the decoupling parameter ("a" or "decoup"), the g-factor ("g" or "gfactor"), or the spectroscopic factors ("s" or "sfactor"), for example:
```
./nilsson.py -N 2 -o 0 -p sfactor
```

### Advanced options

The range and number of delta values can be changed with the options "-Nd" and "-r", for example for a range of deformations delta = [-0.5,0.5] with 10 points on each prolate and obalte side use:
```
./nilsson.py -N 2 -Nd 10 -r 0.5
```

As defaul the calulation is performed with standard values for kappa and mu, which are reproducing the usual ordering and spacing of level for delta=0. In order to change the values for kappa and mu use the argmuments "-k" or "--kappa" and "-m" or "--mu".

```
./nilsson.py -N 3 -k 0.04 -m 0.55
```

### Saving the results

The results of the calculation can be saved as a plain text file using the argument "-w" or "--write".
Currently, this only works for the energy diagram

```
./nilsson.py -n 2 -w nilsson_diagram_N2.dat	
```




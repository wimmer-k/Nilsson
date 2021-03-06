# Nilsson model calculations with Python

This code calculates the energy levels in deformed nuclei. The calculations follows the original publication [S. G. Nilsson, Dan. Mat. Fys. Medd. 29 (1955) 1](http://cds.cern.ch/record/212345).
Optionally, the wave function decomposition in to the spherical components, the decoupling parameter, the g-factors, and the spectroscopic factors can also be calculated and displayed.

## Getting started

The program runs with both Python2.7 and Python3. It has been developed using Python versions 3.8.2, 3.5.2 and 2.7.12. Python3 is recommended. The graphical user interface requires Python3, it uses the Tkinter bindings to the Tk GUI toolkit. Tkinter is automatically installed with Python.

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

In order to plot the wave function decomposition of a certain orbital, record its ID number from the diagram and use it with the option "-o" or "--orbital":
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

The range and number of delta values can be changed with the options "-Nd" and "-r", for example for a range of deformations delta = [-0.5,0.5] with 10 points on each prolate and oblate side use:
```
./nilsson.py -N 2 -Nd 10 -r 0.5
```

As default the calculation is performed with standard values for kappa and mu, which are reproducing the usual ordering and spacing of level for delta=0. In order to change the values for kappa and mu use the arguments "-k" or "--kappa" and "-m" or "--mu".

```
./nilsson.py -N 3 -k 0.04 -m 0.55
```

For the calculation of g-factors, the value of gR, typically Z/A, and the quenching factor can be set with "-gR" or "--gfactR" and "-q" or "--gquench", respectively.

If you want to switch off the DeltaN=2 couplings, use the option "-ndN2" or "--no-deltaN2"

### Saving the results

The results of the calculation can be saved as a plain text file using the argument "-w" or "--write".

```
./nilsson.py -N 2 -w nilsson_diagram_N2.dat	
```

For batch calculations, where the graphical output is not desired, the option "-noplot" can be used in combination with "-w" to just write the calculation results to a file.



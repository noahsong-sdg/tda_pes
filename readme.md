# PH x PES

a project to uncover chemical properties of alkanes using persistent homology (PH)

## Installation
PySCF is used for electronic structure calculations. It is only operable on a Linux operating system. If one obtains a potential energy surface via different means, in theory it should be possible to run the persistence module on the dataset. 

## Instructions
the `src` directory contains the relevant scripts:
- 1_pes.py generates a potential energy surface for a given molecule. the example given in this file is of pentane, and it uses the lowest level of theory provided by PySCF. the datapoints are stored in a .dat file.
- 2_ph.py uses the .dat file to compute a sublevel filtration.

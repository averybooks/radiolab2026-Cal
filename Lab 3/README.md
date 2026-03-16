The tracksun file is the control for the antennas and remains generally unchanged from the intial function.

We require the tracksun file to be in the host directory for the colldata script.

The colldata script is the threaded script that houses the data collection alg, quadrupole, power calculation, and averaging functions. It runs tracksun
threaded with the data collections scripts and should be the one that is called in the terminal.

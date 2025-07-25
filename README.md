# PYNUCDET:PYNUCastro Detonation Estimation Tool 
 PYNUCDET does a self-heating nuclear network calculation for either constant
 volume (isochoric) or constant pressure (isobaric) conditions using Mike Zingale's 
 pynucastro for nuclear reaction rates and compositions, scipy.integrate for 
 integration method, and Frank Timmes's Helmholtz equation of state for the speed
 of sound and specific heats. For isobaric conditions, a variation of the Helmholtz
 equation of state (EOS) adapted from Frank Timmes's Torch code is called as a 
 function of pressure and temperature, mean mass number Abar, and mean atomic number 
 Zbar, and returns density. The Helmholtz calls are handled by a custom Fortran
 wrapper around Helmholtz; the stdout from this wrapper contains the key EOS outputs, 
 which are parsed by the function call_helmholtz in the aux.py module, using the Python
 subprocess module.
 
 We furter compute the critical length for distributed nuclear burning
 using Poloudnenko, Gardiner, & Oran's 2011 PRL condition. The essential idea
 for this determination is to use the detonation initiation condition 
 that the burning timescale is less than the sound-crossing time.
 When the burning timescale is less than the sound-crossing time across
 some region of size L, that region can no longer adjust its burning rate
 by increasing or decreasing its flame surface area subsonically. This 
 runaway burning leads to the development of a shock front and a subsequent 
 detonation.

 Mathematically, this condition is equivalent to the statement t_burn < t_cross
 over some length scale L, so that the critical length L > (e_int / eps_nuc) c_s. 
 Here e_int is the specific internal energy, eps_nuc is the specific nuclear energy 
 generation rate, and c_s is the sound speed.

 Optically-thin neutrino cooling is included from pynucastro, and an optional stub (not fully
 implemented) sets a prescribed turbulent dissipation rate.
 
 The code includes a GUI to allow the user to easily select isotopes to be included in the 
 network calculation.

 ![Sample plot of isotope selector.](/_images/isotope_selector.png)
 
 Three sets of figures are produced:  helium_abundances.png for abundances versus
  time, and detonation_lengths.png for detonation initiation length
  versus time, as well as a set of snapshots of reaction flows sampled at intervals
  in the network burn.


![Sample plot of detonation lengths.](/_images/detonation_lengths.png)

![Sample plot of abundances.](/_images/helium_abundances.png)

![Sample plot of reaction flows.](/_images/reaction_flow_0.10.png)

# Compilation and Usage

 To use, one must first compile the Helmholtz Fortran code in the subdirectory
  _helmholtz, eg,
  
 `cd _helmholtz; make`

 The makefile defaults to the use of gfortran; to use another compiler, simply edit the `FC` 
 variable in the makefile.
 
 Then to run the script, simply cd back to the top level and run

 `python3 self_heating.py`

 The command line options can be displayed by running

 `python3 self_heating.py --help`

 This outputs
 
 `usage: self_heat.py [-h] [--isobaric | --isochoric] [-rho RHO] [-T T]`
                   `[-xhe4 XHE4] [-xc12 XC12] [-xo16 XO16] [-tmax TMAX]`

 `Nuclear reaction network script.`

 `options:`
 
   `-h, --help   show this help message and exit`
   
   `--isobaric   Use isobaric conditions, with pressure set by initial density,`
   
                `temperature, and composition.`
                
   `--isochoric  Use isochoric conditions established by initial density.`
   
   `-rho RHO     Initial mass density in g/cm^3.`
   
   `-T T         Initial temperature in Kelvin.`
   
   `-xhe4 XHE4   Initial abundance of He4.`
   
   `-xc12 XC12   Initial abundance of C12.`
   
   `-xo16 XO16   Initial abundance of O16.`
   
   `-tmax TMAX   Simulation evolutionary time (in seconds).`

 # References 
 
 Poloudnenko, Gardiner, & Oran's 2011 PRL: https://arxiv.org/abs/1106.3696
 
 pynucastro: https://pynucastro.github.io/pynucastro/

 Helmholtz: https://cococubed.com/code_pages/eos.shtml
 
 Helmholtz Inversion from: https://cococubed.com/code_pages/burn.shtml

 -rtf120523

 Last update: rtf020924

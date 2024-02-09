# Self Heating Network
 self_heat.py is a script which does a self-heating nuclear network calculation 
 for either isochoric or isobaric conditions using pynucastro for nuclear
 reaction rates and composition, scipy.integrate for integration method,
 and the Helmholtz equation of state for specific heat. A variation of the 
 Helmholtz equation of state is called as a function of pressure and
 temperature, Abar, and Zbar and returns density, and is adapted from Frank 
 Timmmes' Torch code. The Helmholtz calls are handled by a custom Fortran
 wrapper around Helmholtz; the stdout from this wrapper contains the key EOS 
 outputs, which are parsed by the function call_helmholtz in the aux.py
 module, using subprocess.
 
 We furter compute the critical length for distributed nuclear burning
 using Poloudnenko, Gardiner, & Oran's 2011 PRL condition. The essential idea
 for this determination is to use the detonation initiation condition 
 that the burning timescale is less than the sound-crossing time;
 t_burn < t_cross over some length scale L to find the critical length
 L > (e_int / eps_nuc) c_s. Here e_int is the specific internal energy,
 eps_nuc is the specific nuclear energy generation rate, and c_s is the 
 sound speed.

 The code includes a GUI to allow the user to easily select isotopes to be included in the 
 network calculation.

 ![Sample plot of isotope selector.](/_images/isotope_selector.png)
 
 Three sets of figures are produced:  helium_abundances.png for abundances versus
  time, and detonation_lengths.png for detonation initiation length
  versus time, as well as a set of snapshots of reaction flows sampled at intervals
  in the network burn.

![Sample plot of detonation lengths.](/_images/detonation_lengths.png)

![Sample plot of reaction flows.](/_images/reaction_flow_0.10.png)

 To use, one must first compile the Helmholtz Fortran code in the subdirectory
  _helmholtz, eg,
  
 `cd _helmholtz; make`

 The makefile defaults to the use of gfortran; to use another compiler, simply edit the `FC` 
 variable in the makefile.
 
 Then to run the script, simply cd back to the top level and run

 `python3 self_heating.py`
  
 PGO11: https://arxiv.org/abs/1106.3696
 
 pynucastro: https://pynucastro.github.io/pynucastro/

 Helmholtz: https://cococubed.com/code_pages/eos.shtml
 
 Helmholtz Inversion from: https://cococubed.com/code_pages/burn.shtml

 -rtf120523

 Last update: rtf020824

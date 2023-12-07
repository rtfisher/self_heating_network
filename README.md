# Self Heating Network
 self_heat.py is a script which does a self-heating nuclear network calculation 
 for either isochoric or isobaric conditions using pynucastro for nuclear
 reaction rates and composition, scipy.integrate for integration method,
 and the Helmholtz equation of state for specific heat. A version of the 
 Helmholtz equation of state which is called as a function of pressure and
 temperature as well as Abar and Zbar and returns density which is included
 is adapted from Frank Timmmes' Torch code. The Helmholtz calls are handled
 by a custom Fortran wrapper around Helmholtz; the stdout from this wrapper
 contains the key EOS outputs, which are parsed by the function call_helmholtz
 in the aux.py module, using subprocess.
 
 We furter compute the critical length for distributed nuclear burning
 using Poloudnenko, Gardiner, & Oran's 2011 PRL condition. The essential idea
 for this determination is to use the detonation initiation condition 
 that the burning timescale is less than the sond-crossing time;
 t_burn < t_cross over some length scale L to find the critical length
 L > (e_int / eps_nuc) c_s. Here e_int is the specific internal energy,
 eps_nuc is the specific nuclear energy generation rate, and c_s is the 
 sound speed.

 Two figures are produced, helium_abundances.png for abundances versus
  time, and detonation_lengths.png for detonation initiation length
  versus time.

 To use, one must first compile the Helmholtz Fortran code in the subdirectory
  _helmholtz, eg,
  
 `cd _helmholtz`
 
 `gfortran -o helmholtz.exe  helmholtz_wrapper.f90 helmholtz_library.F90 main.F90 invert_helm_pt.f90`

 Then to run the script, simply cd back to the top level and run

 `python3 self_heating.py`
  
 PGO11: https://arxiv.org/abs/1106.3696
 
 pynucastro: https://pynucastro.github.io/pynucastro/

 Helmholtz: https://cococubed.com/code_pages/eos.shtml
 
 Helmholtz Inversion from: https://cococubed.com/code_pages/burn.shtml

 -rtf120523

 Last update: rtf120623

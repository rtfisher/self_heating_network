# self_heating_network
 self_heat.py is a script which does a self-heating nuclear network calculation 
 (curently assuming isochoric conditions) using pynucastro for nuclear
 reaction rates and composition, scipy.integrate for integration method,
 and the Helmholtz equation of state for specific heat. We further
 also compute the critical length for distributed nuclear burning
 using Poloudnenko, Gardiner, & Oran's 2011 PRL condition.
 
 Essential idea is to use the detonation initiation condition 
 t_burn < t_cross over some length scale L to find the critical length
 L < (e_int / eps_nuc) c_s.

 Two figures are produced, helium_abundances.png for abundances versus
  time, and detonation_lengths.png for detonation initiation length
  versus time.

 PGO11: https://arxiv.org/abs/1106.3696
 pynucastro: https://pynucastro.github.io/pynucastro/
 Helmholtz: https://cococubed.com/code_pages/eos.shtml

 -rtf120523

###########################################################################
##
## A script which does a self-heating nuclear network calculation 
##  (curently assuming isochoric conditions) using pynucastro for nuclear
##  reaction rates and composition, scipy.integrate for integration method,
##  and the Helmholtz equation of state for specific heat. We further
##  also compute the critical length for distributed nuclear burning
##  using Poloudnenko, Gardiner, & Oran's 2011 PRL condition.
## 
## Essential idea is to use the detonation initiation condition 
##  t_burn < t_cross over some length scale L to find the critical length
##  L > (e_int / eps_nuc) c_s.
##
## Two figures are produced, helium_abundances.png for abundances versus
##  time, and detonation_lengths.png for detonation initiation length
##  versus time.
##
## PGO11: https://arxiv.org/abs/1106.3696
## pynucastro: https://pynucastro.github.io/pynucastro/
## Helmholtz: https://cococubed.com/code_pages/eos.shtml
##
## -rtf120523
##
############################################################################

import datetime
import pynucastro as pyna
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import aux # auxilliary module for additional code

# Get the initial time
initial_time = datetime.datetime.now()
print("Begin run :", initial_time.strftime("%B %d, %Y, %H:%M:%S"))

library = pyna.ReacLibLibrary()
# Define a minimal set of isotopes for a rapid He burn
sub = library.linking_nuclei(["p", "n", "he4", "c12", "n13", "n14", "o16",
                              "ne20", "ne21", "na23", "mg23", "mg24", "al27",
                              "si27", "si28"])
rc = pyna.RateCollection(libraries=sub) # Using Reaclib
pynet = pyna.PythonNetwork (libraries=sub)

# Write out the network and import it back in
pynet.write_network ("helium_network.py")
import helium_network as helium_network

# Mapping of isotopes to indices in the helium_network object
isotope_map = {
    'p': helium_network.jp,
    'n': helium_network.jn,
    'he4': helium_network.jhe4,
    'c12': helium_network.jc12,
    'n13': helium_network.jn13,
    'n14': helium_network.jn14,
    'o16': helium_network.jo16,
    'ne20': helium_network.jne20,
    'ne21': helium_network.jne21,
    'na23': helium_network.jna23,
    'mg23': helium_network.jmg23,
    'mg24': helium_network.jmg24,
    'al27': helium_network.jal27,
    'si27': helium_network.jsi27,
    'si28': helium_network.jsi28
}

# Defne initial density and temperature. For isochoric conditions, rho = const.
rho = 1.e5 # g/cm^3
T = 1.e9   # K

# Define initial abundances and initialize pyna Composition object
xhe4_init = 1.0
xc12_init = 0.0

comp = pyna.Composition(rc.get_nuclei())
comp.set_all (0.)
comp.set_nuc ("he4", xhe4_init)
comp.set_nuc ("c12", xc12_init)

# Also define a numpy array of mass and molar initial abundances X0 and Y0
X0 = np.zeros (helium_network.nnuc)
X0 [helium_network.jhe4] = 1.0
X0 [helium_network.jc12] = 0.
Y0 = X0 / helium_network.A

# Set limits of time integration and initial timestep dt
t    = 0.
dt   = 1.e-3
tmax = 10.

# Initialize lists for data storage
times = []
solutions = []
energies = []
critical_lengths = []

while t < tmax:
    print ("t = ", t, " dt = ", dt, " T = ", T)
    # Save the initial state in case we need to redo the step
    Y0_initial = Y0.copy()
    T_initial = T

    # Integrate the ODE system forward  to t + dt
    sol = solve_ivp(helium_network.rhs, [t, t + dt], Y0, method="BDF",
                    jac=helium_network.jacobian, dense_output=True,
                    args=(rho, T), rtol=1.e-6, atol=1.e-6)

    # Append the latest timestep data to solutions
    solutions.append(sol.y[:, -1])
    times.append (t)

    # Compute the energy increase
    n = sol.y.shape[1] - 1
    dY = sol.y[:, n] - Y0_initial [:]   # Y difference over timestep
    de_nuc = helium_network.energy_release(dY)

# Make sure to include neutrino losses here

    # Update mass composition X = Y / A
    for isotope, index in isotope_map.items():
       comp.set_nuc(isotope, sol.y [index, n] / helium_network.A [index])

    # Update abar and zbar
    abar = comp.eval_abar()
    zbar = comp.eval_zbar()

    # Call to the EOS
    pres, eint, gammac, gammae, h, cs, cp, cv = aux.call_helmholtz(rho, T, abar, zbar)

    # Calculate temperature increment for isochoric network using cv
    dT = de_nuc / cv
    T += dT

    # Calculate time derivative of molar abundances dYdt
    dYdt = dY / dt

    # Calculate critical length and append to list
    critical_lengths.append (eint / helium_network.energy_release(dYdt) * cs)
    energies.append ( helium_network.energy_release(dYdt) )

    # Check if temperature increment is too large
    if abs(dT / T_initial) > 0.01:
        # Halve the time step and redo the step
        dt /= 2.0
        Y0 = Y0_initial
        T = T_initial
        continue # skip rest of loop, return to while

    # Update time and initial condition for the next iteration
    t = sol.t[-1]
    Y0 = sol.y[:, -1]

    dt *= 1.01 #increase timestep

    # Check if the next step exceeds tmax
    if t + dt > tmax:
        dt = tmax - t
# end while

########################
# Abundance plot
########################

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(1.e-8, 1.0)
ax.set_xlabel("t (s)")
ax.set_ylabel("X")

# Isotopes to plot
species = [helium_network.jp, helium_network.jhe4, helium_network.jc12, helium_network.jo16, helium_network.jne20, helium_network.jne21, helium_network.jna23, helium_network.jmg24, helium_network.jal27, helium_network.jsi27, helium_network.jsi28]

solutions_array = np.array(solutions).T  # Transpose to match the expected shape
times_array = np.array  (times)
energies_array = np.array  (energies)

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.set_xlim(1.e10, 1.e20)
ax.set_ylim(1.e-8, 1.0)

# Iterate over isotopes, converting betweeen Y and X by multiplying by A
for i in species:
    ax.loglog(times_array, solutions_array [i,:] * helium_network.A[i], label=f"X({helium_network.names[i].capitalize()})")

# Set legend location. Other options include 'upper right', 'lower left', 'best', etc.
ax.legend(loc='upper right')

fig.savefig ("helium_abundances.png")
fig.clf()

############################################################
# Plot critical lengths and proton abundance, eps_nuc inset
#############################################################

fig, ax_left = plt.subplots()

critical_lengths_array = np.array (critical_lengths)

ax_left.loglog(times_array, critical_lengths_array)
# Label the axes
ax_left.set_ylim()
ax_left.set_xlabel('Time (s)')
ax_left.set_ylabel('Critical length (cm)')

# Create a second y-axis that shares the same x-axis
ax_right = ax_left.twinx()

# Plotting proton abundances on the right y-axis
ax_right.plot(times, solutions_array[helium_network.jp, :], 'r')  # 'r' for red line, change as needed
ax_right.set_yscale('log')
ax_right.set_ylim(1.e-8, 1.e-4)
ax_right.set_ylabel('X (p)')

plt.title(r"Isochoric Self-Heating Network with $\rho = 10^5 \, \mathrm{g\, cm}^{-3}$, $T_0 = 10^9\ $ K")

# Add text box in the upper left of the figure
text_str = r'$X(^4\mathrm{He}) = 1.0$, $X(^{12}\mathrm{C}) = 0$'
plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))

# Create the inset axes, adjusting the position
inset_ax = inset_axes(plt.gca(), width="25%", height="25%", loc='lower left',
                      bbox_to_anchor=(0.25, 0.15, 1, 1), bbox_transform=plt.gca().transAxes)

# Plotting on the inset axes
inset_ax.loglog(times_array, np.log10(energies_array), 'r-')  # 'r-' for red line, change as needed

# Make the inset plot partially transparent
inset_ax.patch.set_alpha(0.5)  # Adjust alpha for transparency, 0 is fully transparent, 1 is opaque

# labels for the inset plot
inset_ax.set_xlabel('Time (s)')
inset_ax.set_ylabel(r'log ($\epsilon_{\rm nuc}$)')

fig.savefig("detonation_lengths.png")

# Get the final time
final_time = datetime.datetime.now()
time_difference = final_time - initial_time
print("End run:", final_time.strftime("%B %d, %Y, %H:%M:%S"))
print("Time difference in seconds:", time_difference.total_seconds())

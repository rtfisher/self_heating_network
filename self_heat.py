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
## The algorithm proceeds by initializing the network in pynucastro,
##  setting initial abundances, density, and temperature, and integrating
##  these forward in time to obtain the nuclear energy release as well
##  as the optically-thin neutrino cooling. The temperature is then updated
##  using these heating and cooling rates, as well as C_V or C_P calculated
##  from the Helmholtz EOS assuming isochoric or isobaric thermodynamic
##  conditions, respectively. For isobaric conditions, the density is updated
##  with another call to Helmholtz in an inverse mode, inputing pressure and
##  temperature (and composition) and returning density.
##
## Two figures are produced, helium_abundances.png for abundances versus
##  time, and detonation_lengths.png for detonation initiation length
##  versus time. The figures are adaptively generated and labeled.
##
## PGO11: https://arxiv.org/abs/1106.3696
## pynucastro: https://pynucastro.github.io/pynucastro/
## Helmholtz: https://cococubed.com/code_pages/eos.shtml
##
## -rtf120523
##
############################################################################

import datetime, argparse, math
import pynucastro as pyna
from pynucastro.neutrino_cooling import sneut5
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import aux # auxilliary module for additional code
import sys
import self_heat_gui as gui # GUI module for isotope selection

parser = argparse.ArgumentParser(description="Self-heating nuclear reaction network script.")

 # Add mutually exclusive group for --isobaric and --isochoric to avoid both being accidentally set
group = parser.add_mutually_exclusive_group()
group.add_argument('--isobaric', action='store_true', help='Use isobaric conditions, with pressure set by initial density, temperature, and composition.')
group.add_argument('--isochoric', action='store_true', help='Use isochoric conditions established by initial density.')

# Add arguments for initial density and temperature
parser.add_argument('-rho', type=float, default=1.e5, help='Initial mass density in g/cm^3.')
parser.add_argument('-T', type=float, default=1.e9, help='Initial temperature in Kelvin.')

# Add arguments for initial abundances
parser.add_argument('-xp', type=float, default=0.0, help='Initial abundance of protons.')
parser.add_argument('-xhe4', type=float, default=0.8, help='Initial abundance of He4.')
parser.add_argument('-xc12', type=float, default=0.1, help='Initial abundance of C12.')
parser.add_argument('-xo16', type=float, default=0.1, help='Initial abundance of O16.')

#Add argument for run duration.
parser.add_argument('-tmax', type=float, default=0.1, help='Simulation evolutionary time (in seconds).')

#Add argument to allow for optional output of C++ network. Python is automatically generated.
parser.add_argument("--cppnet", action="store_true", help="Generate a C++ network")

# Add flag for GUI isotope selection
parser.add_argument("--gui", action="store_true", help="Select isotopes using a GUI.")

# Parse the arguments
args = parser.parse_args()

# Set the invert variable based on the arguments
if args.isobaric:
    invert = True
elif args.isochoric:
    invert = False
else:
    # Default behavior if neither isobaric nor isochoric is specified
    invert = False

print ("<PYNUCDet:PYNUCastro Detonation Estimation Tool>  Copyright (C) 2024, Robert T. Fisher. This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it under certain conditions: see the GNU General Public License for details.") 

if invert:
  print("Isobaric run")
else:
  print("Isochoric run")

# Defne initial density and temperature. For isochoric conditions, rho = const.
rho = args.rho # g/cm^3
T_init = T = args.T  # K

print ("Initial density = ", rho, " g/cm^3", " Initial temperature = ", T, " K")

# Define initial abundances and initialize pyna Composition object
norm = args.xp + args.xhe4 + args.xc12 + args.xo16

tolerance = 1e-15  # Adjust the tolerance value as needed

if abs(norm - 1) < tolerance:
  xp_init  = args.xp
  xhe4_init = args.xhe4
  xc12_init = args.xc12
  xo16_init = args.xo16
  print (f"Initial abundances: xp = {xp_init}, xhe4 = {xhe4_init}, xc12 = {xc12_init}, xo16 = {xo16_init}")
else:
  print (f"Error: initial abundances ('{args.xp}' + '{args.xhe4}' + '{args.xc12}' + '{args.xo16}' = '{norm}') must add to unity.")
  exit (1)

# Initialize the pynucastro reaction library
library = pyna.ReacLibLibrary()

# Define the isotopes to be included in the network using a GUI
if args.gui:
  app = gui.QApplication(sys.argv)
  ex = gui.IsotopeSelector()
  ex.show()
  result = app.exec_()
  isotope_list = ex.selected_isotopes  # Access the selected isotopes after the window is closed
else: # use the full network by default
  isotope_list = [
            "p", "n", "he4", "b11", "c12", "c13", "n13", "n14",
            "n15", "o15", "o16", "o17", "f18", "ne19",
            "ne20", "ne21", "ne22", "na22", "na23", "mg23", "mg24",
            "mg25", "mg26", "al25", "si26", "al26", "al27", "si28", "si29",
            "si30", "p29", "p30", "s30", "p31", "s31",
            "s32", "s33", "ar34", "cl33", "cl34", "cl35", "ar36", "ar37",
            "ar38", "ar39", "k39", "ca40", "sc43", "ti44", "v47",
            "cr48", "mn51", "fe52", "fe55", "co55", "ni56",
            "ni58", "ni59"
           ]

print("Included isotopes = ", isotope_list)

# Start the timer for integration, and get the initial time
initial_time = datetime.datetime.now()
print("Begin run :", initial_time.strftime("%B %d, %Y, %H:%M:%S"))

linking_isotopes = library.linking_nuclei( isotope_list)

# Generate the Python rate network
rc = pyna.RateCollection(libraries=linking_isotopes) # Using Reaclib
pynet = pyna.PythonNetwork (libraries=linking_isotopes)

# Also output a C++ network if requested
if args.cppnet:
     cppnet = pyna.SimpleCxxNetwork (libraries=linking_isotopes)
     cppnet.write_network ("cpp_nuclear_network_directory")

# Write out the network and import it back in
pynet.write_network ("nuclear_network.py")
import nuclear_network as nuclear_network

# Auo-generate a map of isotope names to indices in the network
isotope_map = {}
for isotope in isotope_list:
    attribute_name = 'j' + isotope
    if hasattr(nuclear_network, attribute_name):
        isotope_map[isotope] = getattr(nuclear_network, attribute_name)
    else:
        print(f"Attribute {attribute_name} not found in nuclear_network")

# Set composition.
comp = pyna.Composition(rc.get_nuclei())
comp.set_all (0.)
comp.set_nuc ("p", xp_init)
comp.set_nuc ("he4", xhe4_init)
comp.set_nuc ("c12", xc12_init)
comp.set_nuc ("o16", xo16_init)

# Also define a numpy array of mass and number initial abundances X0 and Y0
X0 = np.zeros (nuclear_network.nnuc)
X0 [nuclear_network.jp] = xp_init
X0 [nuclear_network.jhe4] = xhe4_init
X0 [nuclear_network.jc12] = xc12_init
X0 [nuclear_network.jo16] = xo16_init
Y0 = X0 / nuclear_network.A

# Calculate abar and zbar
abar = comp.abar
zbar = comp.zbar

# For isobaric conditions, define a constant pressure by an initial call to Helmholtz
#  using the initial density and temperature and composition.
pres = 0. # initialize
if (invert):
    first_invert = False
    dens, pres, eint, gammac, gammae, h, cs, cp, cv = aux.call_helmholtz (first_invert, rho, T, abar, zbar, pres)
#pres =  8.85359E+021 # dyne/cm^2 for pure He at rho5 = 1, T9 = 1

# Set limits of time integration and initial timestep dt
t    = 0.
accumulated_time = 0  # Initialize an accumulated time counter
tmax = args.tmax  
dt   = 1.e-3 * tmax # Initial timestep, scaled by tmax
dt_plot = tmax / 10  # Plot network flows  every dt_plot, scaled by tmax

# Initialize lists for data storage; the lists will be appended to at each timestep so that they always have a length appropriate for the number of timesteps.
times = []
solutions = []
energies = []
critical_lengths = []

while t < tmax:
    print ("t (s) = ", t, " dt (s) = ", dt, " T (K) = ", T)
    # Save the initial state in case we need to redo the step
    Y0_initial = Y0.copy()
    T_initial = T

    # Integrate the ODE system forward  to t + dt
    sol = solve_ivp (nuclear_network.rhs, [t, t + dt], Y0, method="BDF",
                     jac=nuclear_network.jacobian, dense_output=True,
                     args=(rho, T), rtol=1.e-6, atol=1.e-6)

    # Append the latest timestep data to solutions
    solutions.append(sol.y[:, -1])
    times.append (t)

    # Compute the specific nuclear energy increase
    n = sol.y.shape[1] - 1
    dY = sol.y[:, n] - Y0_initial [:]   # Y difference over timestep
    de_nuc = nuclear_network.energy_release(dY)

    # Update mass composition X = Y * A
    for isotope, index in isotope_map.items():
        comp.set_nuc(isotope, sol.y [index, n] * nuclear_network.A [index])

    # Update abar and zbar
    abar = comp.abar
    zbar = comp.zbar

# Calculate the specific neutrino loss rate
    snu = sneut5 (rho, T, comp) # erg / g / s

    # Call to the EOS, include T/F flag invert to determine whether we call the EOS
    #  as rho/T mode or T/P mode as an inversion, respectively. For isobaric calls,
    #  the rho value is taken as an initial guess
    dens, pres, eint, gammac, gammae, h, cs, cp, cv = aux.call_helmholtz (invert, rho, T, abar, zbar, pres)

    # include a turbulent heating term
#    eturb = 5.e16 # erg / g/ s
    eturb = 0.

    # Check if the accumulated time has reached or exceeded dt_plot
    if accumulated_time >=  dt_plot:
        print(f"Plotting at time {t}")
        ydotmax = np.max (np.abs (dYdt) )
        fig = rc.plot(rho=dens, T=T, comp=comp, ydot_cutoff_value=1.e-2 * ydotmax, curved_edges=False, rotated=False, node_size=800, node_font_size=14, size=(3200, 2400), hide_xalpha=True)
        formatted_time = "{:.2f}".format(t)  # Format to 2 decimal places
        fig.savefig (f"reaction_flow_{formatted_time}.png", dpi=300)
        fig.clf()
        # Reset the accumulated time
#        accumulated_time -= dt_plot
        accumulated_time = 0.

    if not (invert):
    # Calculate temp. increment for isochoric network using specifc heat cv
      dT = (de_nuc + (eturb - snu) * dt) / cv
    else:
      dT = (de_nuc + (eturb - snu) * dt) / cp # isobaric with specific heat cp

    T += dT

    # Calculate time derivative of number abundances dYdt
    dYdt = dY / dt

    # Calculate critical length and append to list
    critical_lengths.append (eint / nuclear_network.energy_release(dYdt) * cs)
    energies.append ( nuclear_network.energy_release(dYdt) )

    # Check if temperature increment is too large
    if abs(dT / T_initial) > 0.01:
        # Halve the time step and redo the step
        print ("Halving timestep due to large temperature increment")
        dt /= 2.0 # Need to check that dt is not too small
        Y0 = Y0_initial
        T = T_initial
        continue # skip rest of loop, return to while; isobaric retains rho

    # Solution is ok; update the density as well for isobaric conditions
    if (invert):
        rho = dens       #update density from EOS call

    # Update time and initial condition for the next iteration
    t = sol.t[-1]
    Y0 = sol.y[:, -1]

    dt *= 1.01 #increase timestep

# Modify dt to ensure the time interval between plots is exactly dt_plot
    if accumulated_time + dt > dt_plot:
        dt = dt_plot - accumulated_time

    # Check if the next step exceeds tmax
    if t + dt > tmax:
        dt = tmax - t

    accumulated_time += dt

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
#species = [nuclear_network.jp, nuclear_network.jhe4, nuclear_network.jc12, nuclear_network.jo16, nuclear_network.jne20, nuclear_network.jne21, nuclear_network.jna23, nuclear_network.jmg24, nuclear_network.jal27,  nuclear_network.jsi28, nuclear_network.js32,  nuclear_network.jar36,  nuclear_network.jca40]

# Plot alpha chain (+ p) isotopes only 
species = [nuclear_network.jp, nuclear_network.jhe4, nuclear_network.jc12, nuclear_network.jo16, nuclear_network.jne20, nuclear_network.jmg24,  nuclear_network.jsi28, nuclear_network.js32,  nuclear_network.jar36,  nuclear_network.jca40]

solutions_array = np.array(solutions).T  # Transpose to match the expected shape
times_array = np.array  (times)
energies_array = np.array  (energies)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(1.e-8, 1.0)

# Iterate over isotopes, converting betweeen Y and X by multiplying by A
for i in species:
    ax.loglog(times_array, solutions_array [i,:] * nuclear_network.A[i], label=f"X({nuclear_network.names[i].capitalize()})")

if (invert):
    plt.title(fr"Isobaric Self-Heating Network, $P = {aux.float_to_latex_scientific(pres)}\, \mathrm{{dyne}}\ \mathrm{{cm}}^{{-2}}$, $T_0 = {aux.float_to_latex_scientific(T_init)}\ $ K") # using f strings
else:
    plt.title(fr"Isochoric Self-Heating Network, $\rho = {aux.float_to_latex_scientific(rho)}\, \mathrm{{g}}\ \mathrm{{cm}}^{{-3}}$, $T_0 = {aux.float_to_latex_scientific(T_init)}\ $ K")

# Set legend location. Other options include 'upper right', 'lower left', 'best', etc.
ax.legend(loc='upper right')

fig.savefig ("abundances.png", dpi=300)
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
ax_right.plot(times, solutions_array[nuclear_network.jp, :], 'r')  # 'r' for red line, change as needed
#ax_right.set_yscale('log')
#ax_right.set_ylim(1.e-8, 1.e-4)
# Dynamically adjust the second y-axis limits based on the maximum and minimum values of the proton abundance; switch from log to linear scale if the dynamic range is too small
xpmax = np.max(solutions_array[nuclear_network.jp, :])
xpmin = np.min(solutions_array[nuclear_network.jp, :])
if (xpmin < 1.e-2 * xpmax):
    roundxpmax = 10**(round (math.log10 (xpmax) ) )
    ax_right.set_ylim(1.e-4 * roundxpmax, roundxpmax)
    ax_right.set_yscale('log')
else:
    ax_right.set_ylim(0, 1.0)
    ax_right.set_yscale('linear')
#roundxpmax = 10**(round (math.log10 (xpmax) ) )
#ax_right.set_ylim(1.e-4 * roundxpmax, roundxpmax)
ax_right.set_ylabel('X (p)')

if (invert):
    plt.title(fr"Isobaric Self-Heating with $P = {aux.float_to_latex_scientific(pres)}\, \mathrm{{dyne}}\ \mathrm{{cm}}^{{-2}}$, $T_0 = {aux.float_to_latex_scientific(T_init)}\ $ K") # using f strings
else:
    plt.title(fr"Isochoric Self-Heating with $\rho = {aux.float_to_latex_scientific(rho)}\, \mathrm{{g}}\ \mathrm{{cm}}^{{-3}}$, $T_0 = {aux.float_to_latex_scientific(T_init)}\ $ K") 

# Add text box in the upper left of the figure
#text_str = r'$X(^4\mathrm{He}) = 1.0$, $X(^{12}\mathrm{C}) = 0$'
text_str = fr'$X(\mathrm{{p}}) = {aux.float_to_latex_scientific(xp_init)}$, $X(^4\mathrm{{He}}) = {aux.float_to_latex_scientific(xhe4_init)}$, $X(^{{12}}\mathrm{{C}}) = {aux.float_to_latex_scientific(xc12_init)}$, $X(^{{16}}\mathrm{{O}}) = {aux.float_to_latex_scientific(xo16_init)}$'

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

fig.savefig("detonation_lengths.png", dpi=300)

# Get the final time
final_time = datetime.datetime.now()
time_difference = final_time - initial_time
print("End run:", final_time.strftime("%B %d, %Y, %H:%M:%S"))
print("Total run time (seconds):", time_difference.total_seconds())

# Store the detonation length data in a file
# Assuming you have the "critical_lengths_array" defined somewhere in your code
min_critical_length = np.min(critical_lengths_array)

# Create the output string
output = f"{args.xhe4} {args.xc12} {args.xo16} {args.rho} {args.T} {min_critical_length}\n"

# Append the output string to the file
with open("detonation_lengths.dat", "a") as file:
    file.write(output)

# Terminate with normal execution
sys.exit (0)

import pynucastro as pyna
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import subprocess

def call_helmholtz(dens, temp, abar, zbar):
    # Construct the command to run the Fortran executable with arguments
    command = ['./helmholtz.exe', str(dens), str(temp), str(abar), str(zbar)]

    # Run the command and capture output
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)

    # Split the output by spaces to get the individual values
    output_values = result.stdout.split()

    # Convert the string values to float and unpack them
    pres, eint, gammac, gammae, h, cs, cp = map(float, output_values)

    return pres, eint, gammac, gammae, h, cs, cp

# A short script to plot the rates and do some calcualtions on a simplified
#  network for a helium burn.

library = pyna.ReacLibLibrary()
sub = library.linking_nuclei(["p", "n", "he4", "c12", "n13", "n14", "o16",
                              "ne20", "ne21", "na23", "mg23", "mg24", "al27",
                              "si27", "si28"])
rc = pyna.RateCollection(libraries=sub)
pynet = pyna.PythonNetwork (libraries=sub)

comp = pyna.Composition(rc.get_nuclei())
comp.set_all (0.)
comp.set_nuc ("he4", 0.8)
comp.set_nuc ("c12", 0.2)
comp.set_nuc ("o16", 1.e-3)
comp.set_nuc ("ne20", 1.e-3)

rc.plot(rho=1.e5, T=1.e9, comp=comp, outfile="helium_flow.png", always_show_p = True, curved_edges = True, rotated = True, ydot_cutoff_value = 1.e-100)

c12c12p = rc.get_rate_by_name("c12(c12,p)na23")
ne20ap  = rc.get_rate_by_name("ne20(a,p)na23")
mg24ap  = rc.get_rate_by_name("mg24(a,p)al27")

c12ag = rc.get_rate_by_name("c12(a,g)o16")
triplea = rc.get_rate_by_name ("a(aa,g)c12") 

rates = [c12c12p, c12ag, ne20ap, mg24ap, triplea]

for rate in rates:
  print (rate, "rate exponent = ", rate.get_rate_exponent (1.e9))

fig = c12c12p.plot()
fig.savefig ("c12c12p.png")

# Temperature range (in Kelvin)
temperatures = np.linspace(5e8, 2e9, 100)

# Initialize lists to store ratio values
ratio_c12c12p_triplea = []
ratio_ne20ap_triplea = []
ratio_mg24ap_triplea = []
ratio_c12ag_triplea = []

# Calculate the ratios for each temperature
for T in temperatures:
    rate_c12c12p = c12c12p.eval(T)
    rate_ne20ap  = ne20ap.eval (T)
    rate_mg24ap  = mg24ap.eval (T)
    rate_c12ag = c12ag.eval(T)
    rate_triplea = triplea.eval(T)

    ratio_c12c12p_triplea.append(rate_c12c12p / rate_triplea)
    ratio_ne20ap_triplea.append(rate_ne20ap / rate_triplea)
    ratio_mg24ap_triplea.append(rate_mg24ap / rate_triplea)
    ratio_c12ag_triplea.append(rate_c12ag / rate_triplea)

# Convert ratios to logarithmic scale (base 10)
log_ratio_c12c12p_triplea = np.log10(ratio_c12c12p_triplea)
log_ratio_ne20ap_triplea = np.log10(ratio_ne20ap_triplea)
log_ratio_mg24ap_triplea = np.log10(ratio_mg24ap_triplea)
log_ratio_c12ag_triplea = np.log10(ratio_c12ag_triplea)

# Plotting
plt.figure(figsize=(10, 6))
#plt.plot(temperatures, log_ratio_c12ag_triplea, label=r'$^{12}$C($\alpha$, $\gamma$)$^{16}$O/3$\alpha$', color='black',  linewidth=2)
plt.plot(temperatures, log_ratio_c12c12p_triplea, label=r'$^{12}$C($^{12}$C,p)$^{23}$Na/3$\alpha$', color='black', linestyle = 'dashed', linewidth=2)
plt.plot(temperatures, log_ratio_ne20ap_triplea, label=r'$^{20}$Ne($\alpha$, p)$^{23}$Na/3$\alpha$', color='black', linestyle = 'dotted', linewidth=2)
plt.plot(temperatures, log_ratio_mg24ap_triplea, label=r'$^{24}$Mg($\alpha$,p)$^{27}$Al/3$\alpha$', color='black', linestyle = 'solid',  linewidth=2)
plt.xlabel('Temperature (K)')
plt.ylabel('Log$_{10}$ of Rate Ratio')
plt.title(r'Log$_{10}$ of Rate Ratios vs Temperature at $\rho = 10^5$ g cm$^{-3}$')
plt.legend()
plt.grid(True)
plt.savefig ("rate_ratio.png")

# Clear plots
plt.clf()

#pynet = pyna.PythonNetwork (rates= rc)
pynet.write_network ("helium_test_integrate.py")

import helium_test_integrate as helium_network

# Integrate an initial abundance
rho = 1.e5
T = 1.e9
X0 = np.zeros (helium_network.nnuc)
X0 [helium_network.jhe4] = 0.8
X0 [helium_network.jc12] = 0.2

Y0 = X0 / helium_network.A

tmax = 0.5

sol = solve_ivp(helium_network.rhs, [0, tmax], Y0, method="BDF", jac=helium_network.jacobian,
                dense_output=True, args=(rho, T), rtol=1.e-6, atol=1.e-6)

fig = plt.figure()
ax = fig.add_subplot(111)

species = [helium_network.jp, helium_network.jhe4, helium_network.jc12, helium_network.jo16, helium_network.jne20, helium_network.jna23, helium_network.jmg24]

#for i in range(helium_network.nnuc):
for i in species:
    ax.loglog(sol.t, sol.y[i,:] * helium_network.A[i], label=f"X({helium_network.names[i].capitalize()})")

#ax.set_xlim(1.e10, 1.e20)
ax.set_ylim(1.e-8, 1.0)
ax.legend(fontsize="small")
ax.set_xlabel("t (s)")
ax.set_ylabel("X")

fig.set_size_inches((8, 6))
fig.savefig ("helium_integration.png")

fig.clf()

# Energy generation plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Get the number of columns
num_columns = sol.y.shape[1]

# Create an array of column indices
column_indices = np.arange(num_columns)

times = []
energies = []
critical_lengths = []

abar = comp.eval_abar()
zbar = comp.eval_zbar()

temp = T / 1.e8
dens = rho / 1.e7

pres, eint, gammac, gammae, h, cs, cp = call_helmholtz(dens, temp, abar, zbar)

for index in column_indices[1:]:
    dY = sol.y[:, index] - sol.y[:, index - 1]
    dt = sol.t[index] - sol.t[index - 1]
    dYdt = dY / dt

    # Accumulate the time and energy values
    times.append(sol.t[index])
    energies.append(helium_network.energy_release(dYdt))
    critical_lengths.append (eint / helium_network.energy_release(dYdt) * cs )

# Plotting outside the loop
ax.loglog(times, critical_lengths)
# Label the axes
ax.set_xlabel('Time (s)')
ax.set_ylabel('Critical length (cm)')
plt.title ("Detonation Length versus Time")
fig.savefig("critical_lengths.png")

###############################################################################
##
## This aux module contains several worker methods.
## 
##  -rtf120723
##
##############################################################################

import subprocess

# This wrapper method calls the Helmholtz using the Unix subprocess module.
#  We parse the stdout of the modified Helmholtz routine to retrieve the 
#  key outputs of the EOS, and return these.

def call_helmholtz(invert, dens, temp, abar, zbar, pres):
    # Construct the command to run the Fortran executable with arguments
    if not (invert):
      command = ['_helmholtz/helmholtz.exe', str (invert), str(dens), str(temp), str(abar), str(zbar)] # isochoric
    else:
      command = ['_helmholtz/helmholtz.exe', str (invert), str(dens), str(temp), str(abar), str(zbar), str (pres)] # isobaric

    # Run the command and capture output
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
   
    # Split the output by spaces to get the individual values
    output_values = result.stdout.split()
    
    # Convert the string values to float and unpack them
    dens, pres, eint, gammac, gammae, h, cs, cp, cv = map(float, output_values)
    
    return dens, pres, eint, gammac, gammae, h, cs, cp, cv

# This method converts a floating point number to a LaTeX string for use in figures.
#  precision is the number of decimal digits to include.

def float_to_latex_scientific(val, precision=1):
    """
    Convert a floating point number to a LaTeX formatted string in scientific notation.
    For values between 0.01 and 99.99, display the number without scientific notation.
    For zero, display it simply as "0".

    :param val: Floating point number.
    :param precision: Number of decimal places.
    :return: String representing the number in LaTeX scientific notation, plain number, or "0".
    """
    # Check for zero
    if val == 0:
        return "0"

    # Check if the number is within the specified range to display without scientific notation
    if 0.01 <= abs(val) <= 99.99:
        return f"{val:.{precision}f}"
    else:
        # Format the number in scientific notation with the specified precision
        formatted_number = f"{val:.{precision}e}"

        # Split the formatted number into its base and exponent
        base, exponent = formatted_number.split("e")
        exponent = int(exponent)  # Convert exponent to an integer

        # Create the LaTeX formatted string
        return f"{base} \\times 10^{{{exponent}}}"


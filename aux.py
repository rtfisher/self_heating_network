import subprocess

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

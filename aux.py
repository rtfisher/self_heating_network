import subprocess

def call_helmholtz(dens, temp, abar, zbar):
    # Construct the command to run the Fortran executable with arguments
    command = ['_helmholtz/helmholtz.exe', str(dens), str(temp), str(abar), str(zbar)]
    
    # Run the command and capture output
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
   
    # Split the output by spaces to get the individual values
    output_values = result.stdout.split()
    
    # Convert the string values to float and unpack them
    pres, eint, gammac, gammae, h, cs, cp, cv = map(float, output_values)
    
    return pres, eint, gammac, gammae, h, cs, cp, cv

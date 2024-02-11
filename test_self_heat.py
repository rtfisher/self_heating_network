import numpy as np
import aux

# Test functions for PYNUCDET code (self-heat)

# Test the call_helmholtz function call
def test_helmholtz_call():

##################################################################################
##
# First, test that the inverted Helmholtz call on a Helmholtz call returns the original values for density and temperature
##
#################################################################################

# Call Helmholrz with desity, temperature, abar, zbar 
    rho = 1.0e6; T = 1.0e9; abar = 16.0; zbar = 8.0; pres = 0. # pressure not needed
    invert = False # Forward EOS Call
    dens, pres, eint, gammac, gammae, h, cs, cp, cv = aux.call_helmholtz (invert, rho, T, abar, zbar, pres)

# Then call Helmholrz with the same density nd  pressure returned from the first call
    invert = True # Inverted EOS Call
    dens_inv, pres_inv, eint_inv, gammac_inv, gammae_inv, h_inv, cs_inv, cp_inv, cv_inv = \
            aux.call_helmholtz (invert, dens, T, abar, zbar, pres)

    np.testing.assert_allclose(dens_inv, rho, rtol=1e-5, atol=1e-8)

##################################################################################
##
# Next, test that Helmholtz returns the expected values for a set of test data
##
#################################################################################

# Read the input and output test data
    input_file_path  = 'helm_input_test_data.txt'
    output_file_path = 'helm_output_test_data.txt'

    with open(input_file_path, 'r') as file1, open(output_file_path, 'r') as file2:
        # Iterate over each line in the files
        for line1, line2 in zip(file1, file2):
            # Strip newline characters and split the lines by spaces to parse columns
            columns1 = line1.strip().split()
            columns2 = line2.strip().split()
            print (columns2)
            # Convert each string in the columns list to a float and thence to a numpy array
            expected_outputs = np.array ([float(column2) for column2 in columns2])

            # Directly convert the first column to a Boolean for file1
            invert = columns1[0] == "True"  # Convert first column directly to Boolean
            rho, T, abar, zbar, pres = [float(column1) for column1 in columns1[1:]]

            # Call Helmholtz and return the results for testing
            dens, pres, eint, gammac, gammae, h, cs, cp, cv = aux.call_helmholtz (invert, rho, T, abar, zbar, pres)
            computed_outputs = np.array ([dens, pres, eint, gammac, gammae, h, cs, cp, cv])

            np.testing.assert_allclose(computed_outputs, expected_outputs, rtol=1e-5, atol=1e-8)

test_helmholtz_call()

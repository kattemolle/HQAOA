"""
Run QAOA from the command line. 

Example
-------
Run QAOA using graph_input.txt located in 'path', using 'd' cycles, with error probability p and error time-correlation ep.

$ python3 QAOA.py path d p ep
"""
import _HQAOA
import qem
from time import time
import os
import chainer as ch
import pickle
from datetime import datetime
import numpy


class Name:
    pass


# Get command line input
cmd_args = _HQAOA.get_command_line_input()
run_args = Name()
try:  # Use GPU if CuPy installation is available.
    import cupy as xp

    run_args.GPU = True
except ImportError:
    import numpy as xp

    run_args.GPU = False

# Make timestamp in UTC of start
run_args.date_start = str(datetime.utcnow())

# Load the ansatz from graph_input.txt
with open(cmd_args.path + "/graph_input.txt", "r") as file:
    run_args.con = eval(file.readline())
    listq = type(run_args.con) == list
    rowq = all(type(row) == list for row in run_args.con)
    entq = all(type(ent) == int for row in run_args.con for ent in row)
    run_args.con = numpy.array(run_args.con, dtype=int)  # Convert to numpy.array
    twodq = len(run_args.con.shape) == 2
    squareq = run_args.con.shape[0] == run_args.con.shape[1]
    orthq = run_args.con.transpose() == run_args.con
    assert all([listq, rowq, entq, twodq, squareq]), "Input file improperly formatted."

# The number of qubits is equal to the number of rows in con.
run_args.n = run_args.con.shape[0]

# Print info about current run to stdout.
print("\n<===")
print("Started basinhopping of", cmd_args.path, "with:")
print("Command line arguments:\n", vars(cmd_args), "\n")
print("Current runtime variables:\n", vars(run_args))
print("<===\n")

###### RUN QAOA #####
run_args.start = time()
qaoa_out = _HQAOA.run_QAOA(cmd_args, run_args)  # Returns Namespace instance.
if run_args.GPU == True:
    qem.sync()
run_args.end = time()
run_args.wall_clock = (
    (run_args.end - run_args.start) / 60 / 60
)  # Wall-clock time of QAOA run, in hours.
####################

# Load the true ground state and ground state energy into memory for computation of infidelities.
with open(cmd_args.path + "/ground_state.txt", "r") as file:
    run_args.ground_space = file.readlines()
    run_args.ground_space = [eval(x.strip()) for x in run_args.ground_space]

# Get the infidelity of the final state compared to the real ground state.
_opt_par = ch.Variable(xp.array(qaoa_out.opt_parameters))
if cmd_args.direction == "temporal":
    run_args.inf_qaoa = (
        _HQAOA.infidelity_from_parameters_swap_temporal_quantum_classical(
            run_args.con,
            run_args.n,
            cmd_args.d,
            cmd_args.p,
            cmd_args.ep,
            _opt_par,
            run_args.ground_space,
        )
    )
elif cmd_args.direction == "spatial":
    run_args.inf_qaoa = _HQAOA.infidelity_from_parameters_swap_spatial(
        run_args.con,
        run_args.n,
        cmd_args.d,
        cmd_args.p,
        cmd_args.ep,
        _opt_par,
        run_args.ground_space,
    )

del _opt_par

# End time in UTC
run_args.date_end = str(datetime.utcnow())

# Write input, runtime arguemtents and results to disk. If no former output exists, print a line explaining the data in the output file.
run_args.con = run_args.con.tolist()
output = str([vars(cmd_args), vars(run_args), vars(qaoa_out)])
if not os.path.exists(cmd_args.path + "/output.txt"):
    f = open(cmd_args.path + "/output.txt", "w")
    f.write(
        "### Output of the QAOA runs which have the folder this txt file is in as 'path' argument.\n"
    )
with open(cmd_args.path + "/output.txt", "a") as f:
    f.write(output + "\n\n")

# Update plot of datapoints
# try:
#    _HQAOA.plot_QAOA_data(cmd_args.path)
#    run_args.plot='success'
# except Exception as e:
#    run_args.plot='fail'
#    print('Plotting error:\n',e)
#    print('! Data generated succesfully but plotting failed.')

# Write input and results to stdout
print(" ")
print(
    "====================================================================================>"
)
print("Finished basinhopping of ", cmd_args.path, "with:")
print("Command line arguments:")
print(vars(cmd_args), "\n")
print("Runtume arguments:")
print(vars(run_args), "\n")
print("Optimization output arguments:")
print(vars(qaoa_out))
print(
    "====================================================================================>"
)

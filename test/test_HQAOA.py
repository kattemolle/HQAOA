import sys

sys.path.insert(1, "../")
import os

assert os.path.isfile("test_HQAOA.py"), "Tests must be run from within the test folder"
from itertools import product
import qem
import _HQAOA
from importlib import reload
from functools import reduce
from matplotlib import pyplot as plt
import chainer as ch
import numpy
from time import time

try:  # Use GPU if CuPy installation is available.
    import cupy as xp

    GPU = True
except ImportError:
    import numpy as xp

    GPU = False


class Name:  # Simple namespace class
    pass


def test_ansatz_phys_swap_sc_spatial():
    n = 4
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    pars = ch.Variable(xp.array([1.0, 2.0, 3.0, 4.0]))
    psc = _HQAOA.ansatz_phys_swap_sc_spatial(con, n, 2, 0.1, 0.2, pars)
    # psc.print_channels() # Checked by hand...

    n = 8
    d = 4
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    pars = ch.Variable(xp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    psc = _HQAOA.ansatz_phys_swap_sc_spatial(con, n, d, 0.1, 0.2, pars)

    RXgates = [
        action.gate
        for channel in psc.chans
        for cir in channel.cirs
        for layer in cir.layers
        for action in layer.actions
        if type(action.gate) == qem.RX
    ]
    assert len(RXgates) == d * n  # Check the number of RX gates.
    assert (
        RXgates[-1].angle.data == pars[-1].data
    )  # Check if last RX gate gets last parameter

    RIsinggates = [
        action.gate
        for channel in psc.chans
        for cir in channel.cirs
        for layer in cir.layers
        for action in layer.actions
        if type(action.gate) == qem.RIsingSWAP
    ]
    assert len(RIsinggates) == d * ((n - 1) ** 2 / 2 + (n - 1) / 2)
    assert RIsinggates[-1].angle.data == pars[-2].data

    # Check transitions are only done on the aux qubit.
    transitions = [channel for channel in psc.chans if channel.name == "Transition"]
    qubits = [trans.cirs[0].layers[0].actions[0].qubits for trans in transitions]
    assert len(transitions) == d * ((n - 1) ** 2 / 2 + (n - 1) / 2) * 2 + d * n
    assert all(qubit[0] == n for qubit in qubits)

    # Check the control is always on the aux qubit
    qubits = [
        action.qubits
        for channel in psc.chans
        for cir in channel.cirs
        for layer in cir.layers
        for action in layer.actions
        if type(action.gate) == qem.CY
    ]
    assert len(qubits) == d * ((n - 1) ** 2 / 2 + (n - 1) / 2) * 2 + d * n
    controls = [qubit[0] for qubit in qubits]
    controls = xp.array(qubits)
    assert all(qubit[0] == n for qubit in qubits)

    # Check if at d odd, qord=reversed(qord_init), and if d is even, qord=qord_init.
    n = 8
    d = 3
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    pars = ch.Variable(xp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    psc = _HQAOA.ansatz_phys_swap_sc_spatial(con, n, d, 0.1, 0.2, pars)
    assert all(psc.qord[i] == list(reversed(range(8)))[i] for i in range(8))

    n = 8
    d = 20
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    pars = ch.Variable(xp.linspace(0, 10, 40))
    psc = _HQAOA.ansatz_phys_swap_sc_temporal(con, n, d, 0.1, 0.2, pars)
    assert all(psc.qord[i] == i for i in range(8))

    # Test for when error_loc is specified.
    # First check if we add the right number of CYs
    n = 8
    d = 20
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    p = 1
    ep = xp.random.rand()
    pars = ch.Variable(xp.random.rand(2 * d))
    error_loc = list(set(xp.random.randint(0, 40, 20).tolist()))
    sc = _HQAOA.ansatz_phys_swap_sc_spatial(con, n, d, p, ep, pars, error_loc=error_loc)
    CYs = [
        channel
        for channel in sc.chans
        if len(channel.cirs) == 1
        and channel.cirs[0].layers[0].actions[0].gate.name == "CY"
    ]
    assert len(error_loc) == len(CYs)

    # Check that the number of CY interaction is what I think it should be in the case error_loc=='all'
    n = xp.random.randint(1, 10) * 2
    d = xp.random.randint(1, 40)
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    p = xp.random.rand()
    ep = xp.random.rand()
    pars = ch.Variable(xp.random.rand(2 * d))
    sc = _HQAOA.ansatz_phys_swap_sc_spatial(con, n, d, p, ep, pars)
    CYs = [
        channel
        for channel in sc.chans
        if len(channel.cirs) == 1
        and channel.cirs[0].layers[0].actions[0].gate.name == "CY"
    ]
    assert len(CYs) == (n // 2 * n + n // 2 * (n - 2) + n) * d

    # Check that if I do 'temporally cor errors' (I do not need the temporal sc if p=1 and I insert error manually), all errors are indeed on the same qubit.


test_ansatz_phys_swap_sc_spatial()
print("test_ansatz_phys_swap_sc_spatial passed")


def parameter_negation():  # Check that negating all variational parameters is a symmetry of the cost function.
    def rand_con(n):
        con = xp.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                con[i, j] = xp.random.randint(2) * 2 - 1
                con[j, i] = con[i, j]
        return con

    p = xp.random.rand() / 2
    ep = xp.random.rand()

    def test(fn, n, d):
        con = rand_con(n)
        gammas = xp.random.rand(d) * 100 - 50
        betas = xp.random.rand(d) * 100 - 50
        pars = xp.transpose([gammas, betas]).flatten()
        pars = ch.Variable(pars)
        ref_cost = fn(con, n, d, p, ep, pars)
        pars = -pars
        cost = fn(con, n, d, p, ep, pars)
        qem.round_assert(cost.data, ref_cost.data)

    test(_HQAOA.cost_from_parameters_swap_temporal_quantum_classical, 6, 4)
    test(_HQAOA.cost_from_parameters_swap_temporal, 4, 6)
    test(_HQAOA.cost_from_parameters_swap_spatial, 4, 5)


parameter_negation()
print("parameter_negation passed")


def test_wrap_parameters():  # This test the fn wrap_parameters, which wraps parameters according to symmetries in such a way that the euclid length of the gammas is minimized. Testing this function crosschecks all cost_form_parameters functions. It checks if they have the right symmetries.
    def rand_con(n):
        con = xp.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                con[i, j] = xp.random.randint(2) * 2 - 1
                con[j, i] = con[i, j]
        return con

    p = xp.random.rand() / 2
    ep = xp.random.rand()

    def test(fn, n, d):
        con = rand_con(n)
        gammas = xp.random.rand(d) * 0.001 - 50
        betas = xp.random.rand(d) * 100 - 50
        pars = xp.transpose([gammas, betas]).flatten()
        pars = ch.Variable(pars)
        ref_cost = fn(con, n, d, p, ep, pars)
        pars = ch.Variable(_HQAOA.wrap_parameters(pars))
        cost = fn(con, n, d, p, ep, pars)
        qem.round_assert(cost.data, ref_cost.data)

    test(_HQAOA.cost_from_parameters_swap_temporal_quantum_classical, 6, 4)
    test(_HQAOA.cost_from_parameters_swap_temporal, 4, 6)
    test(_HQAOA.cost_from_parameters_swap_spatial, 4, 5)


test_wrap_parameters()
print("test_wrap_parameters passed")


def test_infidelity_from_parameters_swap_temporal_quantum_classical():
    # Do QAOA on 4 qubits, strore the opt parameters, use those parameters to compute the infidel later.
    cmd_args = Name()
    cmd_args.d = 4
    cmd_args.p = 0.0
    cmd_args.ep = 0.0
    cmd_args.n_iter = 5

    run_args = Name()
    run_args.complete_graph = [
        (0, 1, [-1.0]),
        (2, 3, [1.0]),
        (0, 3, [1.0]),
        (1, 2, [-1.0]),
        (0, 2, [1.0]),
        (1, 3, [1.0]),
    ]
    run_args.con = xp.zeros((4, 4))
    for edge in run_args.complete_graph:
        run_args.con[edge[0]][edge[1]] = edge[2][0]
    for i in range(4):
        for j in range(i + 1, 4):
            run_args.con[j][i] = run_args.con[i][j]
    run_args.n = 4
    run_args.GPU = GPU

    # out=_HQAOA.run_QAOA(cmd_args,run_args)
    # print(out.opt_parameters)

    ## Check that infidel goes to zero using previously obtained parameters.
    opt_par = [
        -0.33069965950444813,
        0.5719554502311639,
        -0.3713398682996954,
        0.3099784254474594,
        -0.40869805477216725,
        0.8114189671163156,
        -1.0647727262828002,
        -0.46144475436704413,
    ]
    opt_par = ch.Variable(xp.array(opt_par))
    grdsp = _HQAOA.Ising_ground_state(run_args.con)
    infidel = _HQAOA.infidelity_from_parameters_swap_temporal_quantum_classical(
        run_args.con, run_args.n, cmd_args.d, cmd_args.p, cmd_args.ep, opt_par, grdsp
    )
    qem.round_assert(infidel, 0.0)

    # Check against old implementation
    cmd_args = Name()
    cmd_args.d = 4
    cmd_args.p = 0.001
    cmd_args.ep = 0.1
    run_args = Name()
    run_args.con = xp.random.rand(4, 4)
    run_args.con = run_args.con + xp.transpose(run_args.con)
    run_args.n = 4
    for j in range(run_args.n):
        run_args.con[j][j] = 0.0
    run_args.GPU = GPU
    pars = (xp.random.rand(8) - 1 / 2) * 5
    pars = ch.Variable(pars)
    grdsp = _HQAOA.Ising_ground_state(run_args.con)
    new = _HQAOA.infidelity_from_parameters_swap_temporal_quantum_classical(
        run_args.con, run_args.n, cmd_args.d, cmd_args.p, cmd_args.ep, pars, grdsp
    )
    old = _HQAOA.infidelity_from_parameters_swap_temporal(
        run_args.con, run_args.n, cmd_args.d, cmd_args.p, cmd_args.ep, pars, grdsp
    )
    qem.round_assert(new, old)


test_infidelity_from_parameters_swap_temporal_quantum_classical()
print("test_infidelity_from_parameters_swap_temporal_quantum_classical passed")


def test_ansatz_phys_swap_sc_temporal_quantum_classical():
    # Do opgt
    n = 4
    d = 2
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    pars = ch.Variable(xp.array([1.0, 2.0, 3.0, 4.0]))
    psc = _HQAOA.ansatz_phys_swap_sc_temporal_quantum_classical(
        con, n, 2, 0.1, 0.2, pars
    )
    # psc.print_channels() # Checked by hand...

    # Same but with opgi
    n = 4
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    hang1 = ch.Variable(xp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    mixang1 = ch.Variable(xp.array([7.0, 8.0, 9.0, 10.0]))
    hang2 = ch.Variable(xp.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0]))
    mixang2 = ch.Variable(xp.array([17.0, 18.0, 19.0, 20.0]))

    pars = [hang1, mixang1, hang2, mixang2]
    psc = _HQAOA.ansatz_phys_swap_sc_temporal_quantum_classical(
        con, n, 2, 0.1, 0.2, pars
    )
    # psc.print_channels() # Checked by hand...

    # Completely flatten pars
    pars = [ent for par in pars for ent in par]
    RXgates = [
        action.gate
        for channel in psc.chans
        for cir in channel.cirs
        for layer in cir.layers
        for action in layer.actions
        if type(action.gate) == qem.RX
    ]
    assert len(RXgates) == d * n  # Check the number of RX gates.
    assert (
        RXgates[-1].angle.data == pars[-1].data
    )  # Check if last RX gate gets last parameter

    RIsinggates = [
        action.gate
        for channel in psc.chans
        for cir in channel.cirs
        for layer in cir.layers
        for action in layer.actions
        if type(action.gate) == qem.RIsingSWAP
    ]
    assert len(RIsinggates) == d * ((n - 1) ** 2 / 2 + (n - 1) / 2)
    assert (
        RIsinggates[-1].angle.data == pars[-n - 1].data
    )  # Last RIsingSWAP gate gets not the last angle, but n angles before that.


test_ansatz_phys_swap_sc_temporal_quantum_classical()
print("test_ansatz_phys_swap_sc_temporal_quantum_classical passed")


def test_cost_from_parameters_swap_temporal_quantum_classical():
    # Test cost against a bunch of costs obtained in direct Mathematica method.
    # Note I only test the ep=0 and ep=1 cases, and only for two qubits, and only the one parameter per layer case.
    # One parameter per gate case is tested against a one parameter per gate type case using repeated parameters.
    #
    #
    #
    # n=2 qubits, with weight 1, d=1,ep=0
    n = 2
    d = 1

    p = 0.1
    w = [1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    beta = 0.1
    gamma = 0.1
    ep = 0
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
        con, n, d, p, ep, pars
    )
    cost.backward()
    qem.round_assert(cost.data, 0.0202084)
    qem.round_assert(pars.grad.data, [0.199382, 0.199382])

    p = 0.99
    w = [-1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 0
    beta = -3 * xp.pi
    gamma = 0.23
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
        con, n, d, p, ep, pars
    )
    cost.backward()
    qem.round_assert(cost.data, 0.0)
    qem.round_assert(pars.grad.data, [0.0, -0.835681])

    p = 0.4
    w = [1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 1
    beta = 0.9
    gamma = -10.0
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
        con, n, d, p, ep, pars
    )
    cost.backward()
    qem.round_assert(cost.data, -0.177814)
    qem.round_assert(pars.grad.data, [0.158964, 0.0829692])

    p = 0.7
    w = [1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 1
    beta = -0.01
    gamma = -0.1 * xp.pi
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
        con, n, d, p, ep, pars
    )
    cost.backward()
    qem.round_assert(cost.data, -0.00470197)
    qem.round_assert(pars.grad.data, [0.0129434, 0.470134])

    p = 0.7
    w = [-0.3]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 1
    beta = -0.01
    gamma = -0.1 * xp.pi
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
        con, n, d, p, ep, pars
    )
    cost.backward()
    qem.round_assert(cost.data, -0.000449685)
    qem.round_assert(pars.grad.data, [0.0014144, 0.0449625])

    # Check against previous implementation, and check speed dif.
    n = 4
    d = 2
    p = xp.random.rand()
    pars1 = xp.random.rand(2 * d)
    pars2 = pars1.copy()
    pars1 = ch.Variable(pars1)
    pars2 = ch.Variable(pars2)
    rand = xp.random.rand(n, n)
    con = rand + rand.transpose()
    for i in range(n):
        con[i, i] = 0

    start2 = time()
    cost2 = _HQAOA.cost_from_parameters_swap_temporal(con, n, d, p, ep, pars2)
    cost2.backward()
    end2 = time()

    start1 = time()
    cost1 = _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
        con, n, d, p, ep, pars1
    )
    cost1.backward()
    end1 = time()

    print("old implementation took", end2 - start2, "seconds")
    print("new implementation took", end1 - start1, "seconds")

    qem.round_assert(cost1.data, cost2.data)
    qem.round_assert(pars1.grad.data, pars2.grad.data)

    n = 2
    d = 1
    p = 0.7
    w = [-0.3]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 1
    beta = -0.01
    gamma = -0.1 * xp.pi
    pars = ch.Variable(xp.array([gamma, beta, beta]))
    pars_ = [pars[0], pars[1:]]
    cost = _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
        con, n, d, p, ep, pars_
    )
    cost.backward()
    qem.round_assert(cost.data, -0.000449685)
    qem.round_assert(pars.grad.data, [0.0014144, 0.0449625 / 2, 0.0449625 / 2])


test_cost_from_parameters_swap_temporal_quantum_classical()
print("test_cost_from_parameters_swap_temporal_quantum_classical passed")


def test_HQAOA():  # test HQAOA.py as a script. First make a temporary graph_input.txt and ground_state.txt, then run HQAOA. See to it that the infidelity goes to zero for an error-free run, and that plots are created. Remove all files created in the process.
    # Prepare
    n = 2
    d = 2
    con = numpy.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.randint(2) * 2 - 1
            con[j, i] = con[i, j]

    if os.path.isfile("graph_input.txt"):
        raise FileExistsError(
            "a graph_input.txt already exists in the current folder. Please remove it to continue testing."
        )

    with open("graph_input.txt", "w") as f:
        f.write(str(con.tolist()))

    if os.path.isfile("ground_state.txt"):
        raise FileExistsError(
            "a ground_state.txt already exists in the current folder. Please remove it to continue testing."
        )

    if os.path.isfile("data_plot_d={}.pdf".format(d)):
        raise FileExistsError(
            "a",
            "data_plot_d={}.pdf".format(d),
            "already exists in the current folder. Please remove it to continue testing.",
        )

    if os.path.isfile("output.txt"):
        raise FileExistsError(
            "an output.txt already exists in the current folder. Please remove it to continue testing."
        )

    if os.path.isfile("wall_clock.pdf"):
        raise FileExistsError(
            "an wall_clock.pdf already exists in the current folder. Please remove it to continue testing."
        )

    _HQAOA.export_Ising_ground_state(".")

    # Run HQAOA.py for temporal noise, get the infidelity from file, and assert infidelity is small.
    os.system("python3 ../HQAOA.py . 2 0. 0. temporal --n_iter=5")

    with open("output.txt", "r") as f:
        f.readline()
        data = f.readlines()
        data = [line for line in data if line != "\n"]
        data = [eval(line.strip()) for line in data]
        inf_qaoa_list = [line[1]["inf_qaoa"] for line in data]
        inf_qaoa = inf_qaoa_list[0]

    qem.round_assert(inf_qaoa, 0.0)
    os.remove("output.txt")

    # Same for spatial noise correlations.
    os.system("python3 ../HQAOA.py . 2 0. 0. spatial --n_iter=5")
    with open("output.txt", "r") as f:
        f.readline()
        data = f.readlines()
        data = [line for line in data if line != "\n"]
        data = [eval(line.strip()) for line in data]
        inf_qaoa_list = [line[1]["inf_qaoa"] for line in data]
        inf_qaoa = inf_qaoa_list[0]

    qem.round_assert(inf_qaoa, 0.0)
    os.remove("output.txt")
    os.remove("ground_state.txt")
    os.remove("graph_input.txt")


test_HQAOA()
print("test_HQAOA passed")


def test_export_Ising_ground_state():
    # Create a random con matrix, compute the gs directly, save it is grsp.
    # Save the con matrix to a file, run export_Ising_ground_state on this file, import the file again, see if the restult is equal to grdsp.

    # Create random con.
    n = 5
    con = numpy.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.randint(2) * 2 - 1
            con[j, i] = con[i, j]

    # Direct method.
    grdsp = _HQAOA.Ising_ground_state(con)

    # Indirect method
    # import os.path
    if os.path.isfile("graph_input.txt"):
        raise FileExistsError(
            "a graph_input.txt already exists in the current folder. Please remove it to continue testing."
        )

    with open("graph_input.txt", "w") as f:
        f.write(str(con.tolist()))

    if os.path.isfile("ground_state.txt"):
        raise FileExistsError(
            "a ground_state.txt already exists in the current folder. Please remove it to continue testing."
        )

    _HQAOA.export_Ising_ground_state(".")

    with open("ground_state.txt", "r") as file:
        export_grdsp = file.readlines()
        export_grdsp = [eval(x.strip()) for x in export_grdsp]

    os.remove("ground_state.txt")
    os.remove("graph_input.txt")

    assert grdsp == export_grdsp


test_export_Ising_ground_state()
print("test_export_Ising_ground_state passed")


def test_run_QAOA():
    randw = xp.random.rand()
    cmd_args = Name()
    cmd_args.d = 2
    cmd_args.p = 0.001
    cmd_args.ep = 1
    cmd_args.n_iter = 5
    cmd_args.direction = "spatial"
    run_args = Name()
    run_args.con = xp.array([[0, randw], [randw, 0]])
    run_args.n = 2
    run_args.GPU = GPU
    out = _HQAOA.run_QAOA(cmd_args, run_args)
    assert numpy.isnan(out.cost_qaoa) == False

    randw = xp.random.rand()
    cmd_args = Name()
    cmd_args.d = 2
    cmd_args.p = 0.001
    cmd_args.ep = 1
    cmd_args.n_iter = 5
    cmd_args.direction = "temporal"
    run_args = Name()
    run_args.con = xp.array([[0, randw], [randw, 0]])
    run_args.n = 2
    run_args.GPU = GPU
    out = _HQAOA.run_QAOA(cmd_args, run_args)
    assert numpy.isnan(out.cost_qaoa) == False

    randw = xp.random.rand()
    cmd_args = Name()
    cmd_args.d = 2
    cmd_args.p = 0
    cmd_args.ep = 1
    cmd_args.n_iter = 5
    cmd_args.direction = "spatial"
    run_args = Name()
    run_args.con = xp.array([[0, randw], [randw, 0]])
    run_args.n = 2
    run_args.GPU = GPU
    out = _HQAOA.run_QAOA(cmd_args, run_args)
    qem.round_assert(out.cost_qaoa, -randw)

    randw = xp.random.rand()
    cmd_args = Name()
    cmd_args.d = 2
    cmd_args.p = 0.0
    cmd_args.ep = 1
    cmd_args.n_iter = 5
    cmd_args.direction = "temporal"
    run_args = Name()
    run_args.con = xp.array([[0, randw], [randw, 0]])
    run_args.n = 2
    run_args.GPU = GPU
    out = _HQAOA.run_QAOA(cmd_args, run_args)
    qem.round_assert(out.cost_qaoa, -randw)

    randw = xp.random.rand()
    cmd_args = Name()
    cmd_args.d = 2
    cmd_args.p = 0
    cmd_args.ep = 0
    cmd_args.n_iter = 5
    cmd_args.direction = "temporal"
    run_args = Name()
    run_args.con = xp.array([[0, randw], [randw, 0]])
    run_args.n = 2
    run_args.GPU = GPU
    out = _HQAOA.run_QAOA(cmd_args, run_args)
    qem.round_assert(out.cost_qaoa, -randw)

    randw = xp.random.rand()
    cmd_args = Name()
    cmd_args.d = 2
    cmd_args.p = 0
    cmd_args.ep = 0
    cmd_args.n_iter = 5
    cmd_args.direction = "spatial"
    run_args = Name()
    run_args.con = xp.array([[0, randw], [randw, 0]])
    run_args.n = 2
    run_args.GPU = GPU
    out = _HQAOA.run_QAOA(cmd_args, run_args)
    qem.round_assert(out.cost_qaoa, -randw)


test_run_QAOA()
print("test_run_QAOA passed")


def test_Ising_ground_state():
    complete_graph = [
        (0, 1, [1]),
        (1, 2, [-1]),
        (2, 3, [1]),
        (0, 3, [-1]),
        (0, 2, [1]),
        (1, 3, [1]),
    ]
    con = xp.zeros((4, 4))
    for edge in complete_graph:
        con[edge[0]][edge[1]] = edge[2][0]
    out = _HQAOA.Ising_ground_state(con)
    outn = []
    for el in out:
        outn.append(el[0])
        for ell in el[1]:
            outn.append(ell)

    try:
        qem.round_assert(outn, [-6.0, 1, -1, -1, 1, -6.0, -1, 1, 1, -1])
    except AssertionError:
        qem.round_assert(outn, [-6.0, -1, 1, 1, -1, -6.0, 1, -1, -1, 1])

    complete_graph = [
        (0, 1, [-0.997916]),
        (1, 2, [-0.209602]),
        (2, 3, [0.612989]),
        (0, 3, [-0.193729]),
        (0, 2, [1.71222]),
        (1, 3, [-0.632934]),
        (0, 4, [0.514167]),
        (1, 4, [-0.997916]),
        (2, 4, [1.71222]),
        (3, 4, [0.451942]),
    ]
    con = xp.zeros((5, 5))
    for edge in complete_graph:
        con[edge[0]][edge[1]] = edge[2][0]
    for i in range(5):
        for j in range(i + 1, 5):
            con[j][i] = con[i][j]
    out = _HQAOA.Ising_ground_state(con)
    outn = []
    for el in out:
        outn.append(el[0])
        for ell in el[1]:
            outn.append(ell)
    try:
        qem.round_assert(
            outn, [-5.6842130, -1, -1, 1, -1, -1, -5.6842130, 1, 1, -1, 1, 1]
        )
    except AssertionError:
        qem.round_assert(
            outn, [-5.6842130, 1, 1, -1, 1, 1, -5.6842130, -1, -1, 1, -1, -1]
        )


test_Ising_ground_state()
print("test_Ising_ground_state passed")


def test_infidelity_from_parameters_swap_temporal():
    ## First run QAOA
    randw = xp.random.rand()

    cmd_args = Name()
    cmd_args.d = 2
    cmd_args.p = 0
    cmd_args.ep = 0
    cmd_args.n_iter = 4
    cmd_args.direction = "temporal"

    run_args = Name()
    run_args.con = xp.array([[0, randw], [randw, 0]])
    run_args.n = 2
    run_args.GPU = GPU

    out = _HQAOA.run_QAOA(cmd_args, run_args)

    ## Check that infidel goes to zero.
    opt_pars = out.opt_parameters
    opt_pars = ch.Variable(xp.array(opt_pars))
    grdsp = _HQAOA.Ising_ground_state(run_args.con)
    infidel = _HQAOA.infidelity_from_parameters_swap_temporal(
        run_args.con, run_args.n, cmd_args.d, cmd_args.p, cmd_args.ep, opt_pars, grdsp
    )

    qem.round_assert(infidel, 0.0)

    # Do QAOA on 4 qubits, strore the opt parameters, use those parameters to compute the infidel later.
    cmd_args = Name()
    cmd_args.d = 4
    cmd_args.p = 0
    cmd_args.ep = 0
    cmd_args.n_iter = 5

    run_args = Name()
    run_args.layers = [
        [(0, 1, [-1]), (2, 3, [1])],
        [(0, 3, [1]), (1, 2, [-1])],
        [(0, 2, [1]), (1, 3, [1])],
    ]
    run_args.complete_graph = [
        (0, 1, [-1]),
        (2, 3, [1]),
        (0, 3, [1]),
        (1, 2, [-1]),
        (0, 2, [1]),
        (1, 3, [1]),
    ]
    run_args.con = xp.zeros((4, 4))
    for edge in run_args.complete_graph:
        run_args.con[edge[0]][edge[1]] = edge[2][0]
    for i in range(4):
        for j in range(i + 1, 4):
            run_args.con[j][i] = run_args.con[i][j]
    run_args.n = 4
    run_args.GPU = GPU

    # out=_HQAOA.run_QAOA(cmd_args,run_args)
    # print(out.opt_parameters)

    ## Check that infidel goes to zero using previously obtained parameters.
    opt_par = [
        -0.33069965950444813,
        0.5719554502311639,
        -0.3713398682996954,
        0.3099784254474594,
        -0.40869805477216725,
        0.8114189671163156,
        -1.0647727262828002,
        -0.46144475436704413,
    ]
    opt_par = ch.Variable(xp.array(opt_par))
    grdsp = _HQAOA.Ising_ground_state(run_args.con)
    infidel = _HQAOA.infidelity_from_parameters_swap_temporal(
        run_args.con, run_args.n, cmd_args.d, cmd_args.p, cmd_args.ep, opt_par, grdsp
    )
    qem.round_assert(infidel, 0.0)


test_infidelity_from_parameters_swap_temporal()
print("test_infidelity_from_parameters_swap_temporal passed")


def test_infidelity_from_parameters_swap_spatial():
    ## First run QAOA
    randw = xp.random.rand()

    cmd_args = Name()
    cmd_args.d = 2
    cmd_args.p = 0
    cmd_args.ep = 0
    cmd_args.n_iter = 3
    cmd_args.direction = "spatial"

    run_args = Name()
    run_args.con = xp.array([[0, randw], [randw, 0]])
    run_args.n = 2
    run_args.GPU = GPU

    out = _HQAOA.run_QAOA(cmd_args, run_args)

    ## Check that infidel goes to zero.
    opt_pars = out.opt_parameters
    opt_pars = ch.Variable(xp.array(opt_pars))
    grdsp = _HQAOA.Ising_ground_state(run_args.con)
    infidel = _HQAOA.infidelity_from_parameters_swap_spatial(
        run_args.con, run_args.n, cmd_args.d, cmd_args.p, cmd_args.ep, opt_pars, grdsp
    )

    qem.round_assert(infidel, 0.0)

    # Do QAOA on 4 qubits, store the opt parameters, use those parameters to compute the infidel later.
    cmd_args = Name()
    cmd_args.d = 4
    cmd_args.p = 0
    cmd_args.ep = 0
    cmd_args.n_iter = 5

    run_args = Name()
    run_args.layers = [
        [(0, 1, [-1]), (2, 3, [1])],
        [(0, 3, [1]), (1, 2, [-1])],
        [(0, 2, [1]), (1, 3, [1])],
    ]
    run_args.complete_graph = [
        (0, 1, [-1]),
        (2, 3, [1]),
        (0, 3, [1]),
        (1, 2, [-1]),
        (0, 2, [1]),
        (1, 3, [1]),
    ]
    run_args.con = xp.zeros((4, 4))
    for edge in run_args.complete_graph:
        run_args.con[edge[0]][edge[1]] = edge[2][0]
    for i in range(4):
        for j in range(i + 1, 4):
            run_args.con[j][i] = run_args.con[i][j]
    run_args.n = 4
    run_args.GPU = GPU

    # out=_HQAOA.run_QAOA(cmd_args,run_args)
    # print(out.opt_parameters)

    ## Check that infidel goes to zero using previously obtained parameters.
    opt_par = [
        -0.33069965950444813,
        0.5719554502311639,
        -0.3713398682996954,
        0.3099784254474594,
        -0.40869805477216725,
        0.8114189671163156,
        -1.0647727262828002,
        -0.46144475436704413,
    ]
    opt_par = ch.Variable(xp.array(opt_par))
    grdsp = _HQAOA.Ising_ground_state(run_args.con)
    infidel = _HQAOA.infidelity_from_parameters_swap_spatial(
        run_args.con, run_args.n, cmd_args.d, cmd_args.p, cmd_args.ep, opt_par, grdsp
    )
    qem.round_assert(infidel, 0.0)


test_infidelity_from_parameters_swap_spatial()
print("test_infidelity_from_parameters_swap_spatial passed")


def test_cost_from_parameters_swap_spatial():
    # Test cost against a bunch of costs obtained in direct Mathematica method.
    # n=2 qubits, with weight 1, d=1,ep=0
    n = 2
    d = 1

    p = 0.1
    w = [1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    beta = 0.1
    gamma = 0.1
    ep = 0
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_spatial(con, n, d, p, ep, pars)
    cost.backward()
    qem.round_assert(cost.data, 0.0202084)
    qem.round_assert(pars.grad.data, [0.199382, 0.199382])

    p = 0.99
    w = [-1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 0
    beta = -3 * xp.pi
    gamma = 0.23
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_spatial(con, n, d, p, ep, pars)
    cost.backward()
    qem.round_assert(cost.data, 0.0)
    qem.round_assert(pars.grad.data, [0.0, -0.835681])

    p = 0.4
    w = [1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 1
    beta = 0.9
    gamma = -10.0
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_spatial(con, n, d, p, ep, pars)
    cost.backward()
    qem.round_assert(cost.data, -0.177814)
    qem.round_assert(pars.grad.data, [0.158964, 0.0829692])

    p = 0.7
    w = [1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 1
    beta = -0.01
    gamma = -0.1 * xp.pi
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_spatial(con, n, d, p, ep, pars)
    cost.backward()
    qem.round_assert(cost.data, -0.00470197)
    qem.round_assert(pars.grad.data, [0.0129434, 0.470134])

    p = 0.7
    w = [-0.3]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 1
    beta = -0.01
    gamma = -0.1 * xp.pi
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_spatial(con, n, d, p, ep, pars)
    cost.backward()
    qem.round_assert(cost.data, -0.000449685)
    qem.round_assert(pars.grad.data, [0.0014144, 0.0449625])


test_cost_from_parameters_swap_spatial()
print("test_cost_from_parameters_swap_spatial passed")


def test_cost_con():  # Make a bunch of graphs and states, and check the cost.
    ## Graph: Fully connected graph on 4 qubits.
    ## State: (|1 0 1 1> + 1j*|0 0 0 1>)/sqrt(2)
    ## Cost should be: 0.
    ## But we interchange qubits 0 and 1.

    n = 4
    con = xp.array(
        [
            [0, 0.1, -0.5, -0.4],
            [0.1, 0, 0.2, -0.6],
            [-0.5, 0.2, 0, 0.3],
            [-0.4, -0.6, 0.3, 0],
        ]
    )
    qord = [0, 1, 2, 3]
    # complete_graph=[(0,1,[.1]),(1,2,[.2]),(3,2,[.3]),(3,0,[-.4]),(2,0,[-.5]),(1,3,[-.6])]
    reg = qem.MixReg(4)
    ket0 = [[1.0], [0.0]]
    ket1 = [[0.0], [1.0]]
    initket = (
        reduce(xp.kron, [ket1, ket0, ket1, ket1])
        + 1j * reduce(xp.kron, [ket0, ket0, ket0, ket1])
    ) / xp.sqrt(2)
    initrho = xp.kron(initket, initket.conjugate().transpose()).reshape((2,) * 8)
    reg.psi = qem.Array(initrho.real, initrho.imag)
    # Now swap around some qubits, and check if energy remains invariant.
    costref = (-0.1 - 0.2 + 0.3 - 0.4 - 0.5 + 0.6) / 2 + (
        0.1 + 0.2 - 0.3 + 0.4 - 0.5 + 0.6
    ) / 2
    for t in range(20):
        i = xp.random.randint(4)
        j = (i + xp.random.randint(1, 4)) % 4
        cir = qem.Cir()
        cir.append_action(qem.Action((i, j), qem.SWAP()))
        qem.apply_channel(qem.Chan([cir]), reg)
        qord[i], qord[j] = qord[j], qord[i]
        qem.round_assert(_HQAOA.cost_con(con, qord, n, reg).data, costref)


test_cost_con()
print("test_cost_con passed")


def test_cost_from_parameters_swap_temporal():
    # Test cost against a bunch of costs obtained in direct Mathematica method.
    # Note I only test the ep=0 and ep=1 cases, and only for two qubits.
    # n=2 qubits, with weight 1, d=1,ep=0
    n = 2
    d = 1

    p = 0.1
    w = [1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    beta = 0.1
    gamma = 0.1
    ep = 0
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_temporal(con, n, d, p, ep, pars)
    cost.backward()
    qem.round_assert(cost.data, 0.0202084)
    qem.round_assert(pars.grad.data, [0.199382, 0.199382])

    p = 0.99
    w = [-1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 0
    beta = -3 * xp.pi
    gamma = 0.23
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_temporal(con, n, d, p, ep, pars)
    cost.backward()
    qem.round_assert(cost.data, 0.0)
    qem.round_assert(pars.grad.data, [0.0, -0.835681])

    p = 0.4
    w = [1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 1
    beta = 0.9
    gamma = -10.0
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_temporal(con, n, d, p, ep, pars)
    cost.backward()
    qem.round_assert(cost.data, -0.177814)
    qem.round_assert(pars.grad.data, [0.158964, 0.0829692])

    p = 0.7
    w = [1.0]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 1
    beta = -0.01
    gamma = -0.1 * xp.pi
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_temporal(con, n, d, p, ep, pars)
    cost.backward()
    qem.round_assert(cost.data, -0.00470197)
    qem.round_assert(pars.grad.data, [0.0129434, 0.470134])

    p = 0.7
    w = [-0.3]
    con = xp.array([[0.0, w[0]], [w[0], 0.0]])
    ep = 1
    beta = -0.01
    gamma = -0.1 * xp.pi
    pars = ch.Variable(xp.array([gamma, beta]))
    cost = _HQAOA.cost_from_parameters_swap_temporal(con, n, d, p, ep, pars)
    cost.backward()
    qem.round_assert(cost.data, -0.000449685)
    qem.round_assert(pars.grad.data, [0.0014144, 0.0449625])


test_cost_from_parameters_swap_temporal()
print("test_cost_from_parameters_swap_temporal passed")


def test_ansatz_phys_swap_sc_temporal():
    n = 4
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    pars = ch.Variable(xp.array([1.0, 2.0, 3.0, 4.0]))
    psc = _HQAOA.ansatz_phys_swap_sc_temporal(con, n, 2, 0.1, 0.2, pars)
    # psc.print_channels() # Checked by hand...

    n = 8
    d = 4
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    pars = ch.Variable(xp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    psc = _HQAOA.ansatz_phys_swap_sc_temporal(con, n, d, 0.1, 0.2, pars)

    RXgates = [
        action.gate
        for channel in psc.chans
        for cir in channel.cirs
        for layer in cir.layers
        for action in layer.actions
        if type(action.gate) == qem.RX
    ]
    assert len(RXgates) == d * n  # Check the number of RX gates.
    assert (
        RXgates[-1].angle.data == pars[-1].data
    )  # Check if last RX gate gets last parameter

    RIsinggates = [
        action.gate
        for channel in psc.chans
        for cir in channel.cirs
        for layer in cir.layers
        for action in layer.actions
        if type(action.gate) == qem.RIsingSWAP
    ]
    assert len(RIsinggates) == d * ((n - 1) ** 2 / 2 + (n - 1) / 2)
    assert RIsinggates[-1].angle.data == pars[-2].data

    # Check transitions are only done on aux qubits.
    transitions = [channel for channel in psc.chans if channel.name == "Transition"]
    qubits = [trans.cirs[0].layers[0].actions[0].qubits for trans in transitions]
    assert len(transitions) == d * ((n - 1) ** 2 / 2 + (n - 1) / 2) * 2 + d * n
    qubits = xp.array(qubits)
    qubits %= 2
    assert all(qubits)

    # Check the control is always on the aux qubit
    qubits = [
        action.qubits
        for channel in psc.chans
        for cir in channel.cirs
        for layer in cir.layers
        for action in layer.actions
        if type(action.gate) == qem.CY
    ]
    assert len(qubits) == d * ((n - 1) ** 2 / 2 + (n - 1) / 2) * 2 + d * n
    controls = [qubit[0] for qubit in qubits]
    controls = xp.array(qubits)
    controls %= 2
    assert all(qubits)

    # Check if at d odd, qord=reversed(qord_init), and if d is even, qord=qord_init.
    n = 8
    d = 3
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    pars = ch.Variable(xp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    psc = _HQAOA.ansatz_phys_swap_sc_temporal(con, n, d, 0.1, 0.2, pars)
    assert all(psc.qord[i] == list(reversed(range(8)))[i] for i in range(8))

    n = 8
    d = 20
    con = numpy.zeros((n, n))  # numpy.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            con[i, j] = numpy.random.rand()  # numpy.random.randint(2)*2-1
            con[j, i] = con[i, j]

    pars = ch.Variable(xp.linspace(0, 10, 40))
    psc = _HQAOA.ansatz_phys_swap_sc_temporal(con, n, d, 0.1, 0.2, pars)
    assert all(psc.qord[i] == i for i in range(8))


test_ansatz_phys_swap_sc_temporal()
print("test_ansatz_phys_swap_sc_temporal passed")


def test_generate_random_SK_model_SWAP_network_graph_input():
    # Check by hand
    # _HQAOA.generate_random_SK_model_SWAP_network_graph_input('data',4)
    pass


test_generate_random_SK_model_SWAP_network_graph_input()
print("generate_random_SK_model_SWAP_network_graph_input passed")


def test_transition_channel():
    print("test_transition_channel, progress:")

    # Check that application of the chan gives right outcome

    p = xp.random.rand() / 2
    ep = xp.random.rand()

    reg = qem.MixReg(1)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data, [[1 - p + ep * p, 0], [0, p - ep * p]]
    )

    reg = qem.MixReg(1)
    qem.apply_channel(qem.X_channel(0), reg)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data,
        [[1 - p - ep + ep * p, 0], [0, ep + p - ep * p]],
    )

    p = 0
    ep = 1

    reg = qem.MixReg(1)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data, [[1 - p + ep * p, 0], [0, p - ep * p]]
    )

    reg = qem.MixReg(1)
    qem.apply_channel(qem.X_channel(0), reg)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data,
        [[1 - p - ep + ep * p, 0], [0, ep + p - ep * p]],
    )

    p = 0.5
    ep = 0

    reg = qem.MixReg(1)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data, [[1 - p + ep * p, 0], [0, p - ep * p]]
    )

    reg = qem.MixReg(1)
    qem.apply_channel(qem.X_channel(0), reg)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data,
        [[1 - p - ep + ep * p, 0], [0, ep + p - ep * p]],
    )

    p = xp.random.rand() / 2
    ep = xp.random.rand()

    reg = qem.MixReg(1)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data, [[1 - p + ep * p, 0], [0, p - ep * p]]
    )

    reg = qem.MixReg(1)
    qem.apply_channel(qem.X_channel(0), reg)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data,
        [[1 - p - ep + ep * p, 0], [0, ep + p - ep * p]],
    )

    p = 0.5
    ep = 1

    reg = qem.MixReg(1)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data, [[1 - p + ep * p, 0], [0, p - ep * p]]
    )

    reg = qem.MixReg(1)
    qem.apply_channel(qem.X_channel(0), reg)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data,
        [[1 - p - ep + ep * p, 0], [0, ep + p - ep * p]],
    )

    p = 0
    ep = 0

    reg = qem.MixReg(1)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data, [[1 - p + ep * p, 0], [0, p - ep * p]]
    )

    reg = qem.MixReg(1)
    qem.apply_channel(qem.X_channel(0), reg)
    qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
    qem.round_assert(
        reg.psi.re.data + 1j * reg.psi.im.data,
        [[1 - p - ep + ep * p, 0], [0, ep + p - ep * p]],
    )

    # Check that average state is close to steady state.
    p = 0.1 + 0.8 * xp.random.rand()
    ep = (
        0.5 * xp.random.rand()
    )  # Note more correlation makes it way harder for the correlation strength to converge.
    tmax = 2**8

    reg = qem.random_mixreg(1)

    for i in range(tmax):
        qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)

    afinal = reg.psi.re.data + 1j * reg.psi.im.data
    qem.round_assert(afinal, [[1 - p, 0], [0, p]])

    # Check that cor. strength. is 1+4 p (ep + p - ep p -1). Make single rel. Very time consuming because estimator may converge slowly.
    p = 3 / 8 + xp.random.rand() / 8
    ep = (
        0.2 * xp.random.rand()
    )  # Note more correlation makes it harder for the correlation strength to converge.
    exponent = 5
    tmax = 10**exponent
    reg = qem.MixReg(1)
    x = xp.random.rand()
    if x < 0.5:
        qem.apply_channel(qem.X_channel(0), reg)

    lst = []

    for i in range(tmax):
        sys.stdout.write("i=%d%%\r" % (i / tmax * 100))
        qem.apply_channel(_HQAOA.transition_channel(p, ep, 0), reg)
        x = xp.array(xp.random.rand())
        if x < reg.psi.re.data[0][0]:
            reg.psi.re = xp.array([[1.0, 0.0], [0.0, 0.0]])
            lst.append(1)
        else:
            reg.psi.re = xp.array([[0.0, 0.0], [0.0, 1.0]])
            lst.append(-1)

    su = 0.0
    for i in range(0, len(lst), 2):
        su += lst[i] * lst[i + 1]
        # if i!=0: print(su/(i))
    cor = su / (len(lst) / 2)
    cor2 = 1 + 4 * p * (ep + p - ep * p - 1)
    print("estimated correlation:", cor, ", analytical correlation:", cor2)
    assert xp.round(cor, exponent - 4) == xp.round(cor2, exponent - 4)


# This test is not run because it takes very long, especially with large tmax, which is needed for convergence.
# test_transition_channel()
# print('test_transition_channel passed')

#####################################
print("All tests passed succesfully")
#####################################

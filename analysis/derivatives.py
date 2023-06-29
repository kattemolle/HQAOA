"""
Compute derivatives of the AR, using the derivative theorem of the paper, and use them to compute a extrapolated AR. Output is stored in devivatives.out by redirecting stdout externally.
"""

import sys

sys.path.insert(1, "../")
import numpy as np
import _HQAOA
import chainer as ch
from tabulate import tabulate


def max_error_locs(n, d):
    assert type(n) == int
    assert type(d) == int
    return (n // 2 * n + n // 2 * (n - 2) + n) * d


def error_locs_uncor(n, d):
    assert n % 2 == 0
    l = max_error_locs(n, d)
    return np.array(range(l)).reshape(l, 1).tolist()


def error_locs_spatial(n, d):
    assert n % 2 == 0
    ls = []
    i = 0
    count = 0
    while i < (n + 1) * d:
        if (i % (n + 1)) % 2 == 0:
            ls.append(list(range(count, count + n)))
            i += 1
            count += n
        else:
            ls.append(list(range(count, count + (n - 2))))
            i += 1
            count += n - 2

    assert len(ls) == (n + 1) * d
    flattened = [ent for line in ls for ent in line]
    assert flattened == list(range(max_error_locs(n, d)))

    return ls


def error_locs_temporal(
    n, d
):  # Take spatial noise locs, pad the lines with lower length with None to make regular array, transpose the array, remove the Nones.
    lsspat = error_locs_spatial(n, d)
    lstemp = []
    for ent in lsspat:
        new_ent = ent
        assert len(new_ent) == n or len(new_ent) == (n - 2)
        if len(new_ent) == n - 2:
            new_ent.insert(0, None)
            new_ent.append(None)
        lstemp.append(new_ent)

    lstemp = np.array(lstemp).transpose().tolist()

    def remove_none(list):
        return [ent for ent in list if ent != None]

    lstemp = [remove_none(line) for line in lstemp]

    return lstemp


def calc_linearized_AR(
    case,
):  # Case should specify: case.ipath_noiseless,case.ipath_noise,n,case.load_p,case.load_ep,case.d,case.error_locs_fn,case.noise_aware_p,case.noise_aware_ep
    # Load optimal noiseless data from case.ipath_noiseless
    out = _HQAOA.Name()
    load_p = 0
    load_ep = 0
    opt_line = _HQAOA.load_optimal_line(case.ipath_noiseless, load_p, load_ep, case.d)
    con = opt_line[1]["con"]
    pars_noiseless = opt_line[2]["opt_parameters"]
    pars_noiseless = ch.Variable(np.array(pars_noiseless))
    out.noiseless_cost = opt_line[2]["cost_qaoa"]
    E0 = _HQAOA.get_instance_E0(case.ipath_noiseless)

    inj_p = 1  # The error probability p when noise is injected manually.
    compute_ep = 0  # Results should not depend on compute_ep, because inj_p=1, it should, however, be set nontheless.

    out.av = 0
    for error_loc in case.error_locs_fn(case.n, case.d):
        term = _HQAOA.cost_from_parameters_swap_spatial(
            con, case.n, case.d, inj_p, compute_ep, pars_noiseless, error_loc
        ).data  # Note the spatial cost_from_parameters is used in all cases. It's ok because errors are inserted manually at compute_ep=0 with p=1 anyway.
        out.av += term

    if case.error_locs_fn == error_locs_uncor:
        l = max_error_locs(case.n, case.d)
    elif case.error_locs_fn == error_locs_spatial:
        l = (case.n + 1) * case.d
    elif case.error_locs_fn == error_locs_temporal:
        l = case.n

    out.noise_aware_cost = _HQAOA.load_optimal_line(
        case.ipath_noise, case.noise_aware_p, case.noise_aware_ep, case.d
    )[2]["cost_qaoa"]
    out.av = out.av / l
    out.dif = out.av - out.noiseless_cost
    out.extrap_cost = out.noiseless_cost + l * out.dif * case.noise_aware_p
    out.dARdp = l * out.dif / E0
    out.extrap_AR = out.noiseless_cost / E0 + out.dARdp * case.noise_aware_p

    out.noise_aware_AR = out.noise_aware_cost / E0
    out.rel_error_with_aware = (
        str((out.noise_aware_AR - out.extrap_AR) / out.noise_aware_AR * 100) + "%"
    )

    # Also compute noise unwarare AR for comparison
    if "spatial" in case.ipath_noise:
        out.noise_unaware_AR = _HQAOA.cost_from_parameters_swap_spatial(
            con, case.n, case.d, case.noise_aware_p, case.noise_aware_ep, pars_noiseless
        ).data
    elif "temporal" in case.ipath_noise:
        out.noise_unaware_AR = (
            _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
                con,
                case.n,
                case.d,
                case.noise_aware_p,
                case.noise_aware_ep,
                pars_noiseless,
            ).data
        )
    out.noise_unaware_AR = out.noise_unaware_AR / E0
    out.rel_error_with_unaware = (
        str((out.noise_unaware_AR - out.extrap_AR) / out.noise_unaware_AR * 100) + "%"
    )

    # Print results
    print()
    print("Printing data for the case:")
    print("ipath_noiseless")
    print(case.ipath_noiseless)
    print("ipath_noise")
    print(case.ipath_noise)
    del case.ipath_noiseless
    del case.ipath_noise
    print(tabulate([vars(case).keys(), vars(case).values()]))
    print("result:")
    num = len(vars(out)) // 2
    split = [list(vars(out).keys()), list(vars(out).values())]
    split = split + split
    split = [split[0][:num], split[1][:num], split[2][num:], split[3][num:]]
    print(tabulate(split))
    print()


if __name__ == "__main__":
    # Temporal
    temporal = _HQAOA.Name()
    temporal.ipath_noiseless = "../data/SK/SWAP-network/6/spatial/010010100111110"
    temporal.ipath_noise = "../data/SK/SWAP-network/6/temporal/010010100111110"
    temporal.n = 6
    temporal.d = 3
    temporal.error_locs_fn = error_locs_temporal
    temporal.noise_aware_p = 0.001
    temporal.noise_aware_ep = 1

    # Spatial
    spatial = _HQAOA.Name()
    spatial.ipath_noiseless = "../data/SK/SWAP-network/6/spatial/010010100111110"
    spatial.ipath_noise = "../data/SK/SWAP-network/6/spatial/010010100111110"
    spatial.n = 6
    spatial.d = 3
    spatial.error_locs_fn = error_locs_spatial
    spatial.noise_aware_p = 0.001
    spatial.noise_aware_ep = 1

    # Uncor
    uncor = _HQAOA.Name()
    uncor.ipath_noiseless = "../data/SK/SWAP-network/6/spatial/010010100111110"
    uncor.ipath_noise = "../data/SK/SWAP-network/6/spatial/010010100111110"
    uncor.n = 6
    uncor.d = 3
    uncor.error_locs_fn = error_locs_uncor
    uncor.noise_aware_p = 0.001
    uncor.noise_aware_ep = 0

    # Add predictions on noise unaware extrapolation to p=0.01
    # Temporal
    temporal2 = _HQAOA.Name()
    temporal2.ipath_noiseless = "../data/SK/SWAP-network/6/spatial/010010100111110"
    temporal2.ipath_noise = "../data/SK/SWAP-network/6/temporal/010010100111110"
    temporal2.n = 6
    temporal2.d = 3
    temporal2.error_locs_fn = error_locs_temporal
    temporal2.noise_aware_p = 0.01  # Dif with above
    temporal2.noise_aware_ep = 1

    # Spatial
    spatial2 = _HQAOA.Name()
    spatial2.ipath_noiseless = "../data/SK/SWAP-network/6/spatial/010010100111110"
    spatial2.ipath_noise = "../data/SK/SWAP-network/6/spatial/010010100111110"
    spatial2.n = 6
    spatial2.d = 3
    spatial2.error_locs_fn = error_locs_spatial
    spatial2.noise_aware_p = 0.01  # Dif with above
    spatial2.noise_aware_ep = 1

    # Uncor
    uncor2 = _HQAOA.Name()
    uncor2.ipath_noiseless = "../data/SK/SWAP-network/6/spatial/010010100111110"
    uncor2.ipath_noise = "../data/SK/SWAP-network/6/spatial/010010100111110"
    uncor2.n = 6
    uncor2.d = 3
    uncor2.error_locs_fn = error_locs_uncor
    uncor2.noise_aware_p = 0.01  # Dif with above
    uncor2.noise_aware_ep = 0

    # Run
    list(
        map(calc_linearized_AR, [temporal, spatial, uncor, temporal2, spatial2, uncor2])
    )

    # Output is stored in output file 'derivatives.out' by shell redirection.

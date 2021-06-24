from typing import Union, Sequence, List, Tuple, Dict
import numpy as np
from scipy.stats import binom, binom_test, poisson, skellam, norm
from logzero import logger
from ..type import STATES


def assign_init(intvls, profile, depths):
    asgn = ''
    for b, e in intvls:
        dmin = 32768
        smin = None
        for s in STATES:   # TODO: binary search
            d = abs(depths[s] - max(profile[b], profile[e - 1]))
            if d < dmin:
                smin = s
            dmin = d
        asgn += smin
    return asgn


def asgn_to_states(asgn, intvls):
    return ''.join([s * (e - b) for (b, e), s in zip(intvls, asgn)])


def assign_update(intvls,
                  asgn,
                  profile,
                  pread,
                  depths,
                  max_n_iter: int = 15,
                  return_all_history: bool = False,
                  verbose: bool = False) -> Union[str, List[str]]:
    asgns = [asgn]   # all history of assignments
    changed = True
    count = 0
    while changed and count < max_n_iter:
        if verbose:
            print(f"########## Round {count + 1} ##########")
        else:
            print(".", end='')
        asgn = list(asgns[-1])
        changed = False
        # NOTE: 1. left to right and right to left
        # for i in (list(range(len(asgn))) + list(reversed(range(len(asgn))))):
        # NOTE: 2. upper to lower and lower to upper (more robust?)
        count_index_list = [(min(profile[s], profile[t - 1]), i)
                            for i, (s, t) in enumerate(intvls)]
        for _, i in sorted(count_index_list, reverse=True) + sorted(count_index_list):
            new_s = update_state(i, intvls, asgn, profile,
                                 pread, depths, verbose)
            if new_s != asgn[i]:
                changed = True
                asgn[i] = new_s
        count += 1
        asgns.append(''.join(asgn))
    if count == max_n_iter:
        logger.warn(f"Read {pread.id}: Not converged")
    return asgns if return_all_history else asgns[-1]


def update_state(i, intvls, asgn, profile, pread, DEPTHS, verbose=False):
    if not isinstance(asgn, list):
        asgn = list(asgn)
    max_logp = -np.inf
    max_s = None
    if verbose:
        print(
            f"Interval {i}: range={intvls[i]}, boundary counts=({profile[intvls[i][0]]}, {profile[intvls[i][1] - 1]})")
    for s in STATES:
        logp = (logp_e(i, intvls, asgn, profile, pread, DEPTHS, verbose) if s == 'E'
                else logp_r(i, intvls, asgn, profile, DEPTHS, verbose) if s == 'R'
                else logp_h(i, intvls, asgn, profile, pread, DEPTHS, verbose) if s == 'H'
                else logp_d(i, intvls, asgn, profile, pread, DEPTHS, verbose))
        if logp > max_logp:
            max_logp = logp
            max_s = s
        if verbose:
            print(f"    >>> logp(s_{i}={s}) = {logp:.3f}\n")
    if max_s != asgn[i]:
        if verbose:
            print(
                f"*** State updated: s_{i} (range={intvls[i]}) = {asgn[i]} -> {max_s}\n")
    return max_s


def nn_intvl(i, s, d, asgn):
    # find nearest neighbor count from interval `i` whose state is in `s`
    if not isinstance(s, set):
        s = set(s if isinstance(s, list) else [s])
    assert d in ('l', 'r', 'b')
    if d in ('l', 'b'):
        p = i - 1
        while p >= 0 and asgn[p] not in s:
            p -= 1
    if d in ('r', 'b'):
        n = i + 1
        while n < len(asgn) and asgn[n] not in s:
            n += 1
    if d == 'l':
        return p
    elif d == 'r':
        return n
    else:
        return p, n


def estimate_true_counts(i, s, d, intvls, asgn, profile,
                         lestimate=1000, lestimate_min=1):
    # use max count `l` bp around interval `i` as an estimate of the true counts
    if not isinstance(s, set):
        s = set(s if isinstance(s, list) else [s])
    assert d in ('l', 'r', 'b')
    if d in ('l', 'b'):
        _l = lestimate
        _profile = []
        p = i - 1
        while p >= 0 and _l > 0:
            if asgn[p] in s:
                pb, pe = intvls[p]
                __profile = profile[pb:pe][-_l:]
                _l -= len(__profile)
                _profile += __profile
            p -= 1
        if p >= 0:
            assert len(_profile) == lestimate
        # print(_profile)
        l_cmean = - \
            1 if len(_profile) < lestimate_min else sum(
                _profile) // len(_profile)
    if d in ('r', 'b'):
        _l = lestimate
        _profile = []
        n = i + 1
        while n < len(asgn) and _l > 0:
            if asgn[n] in s:
                nb, ne = intvls[n]
                __profile = profile[nb:ne][:_l]
                _l -= len(__profile)
                _profile += __profile
            n += 1
        if n < len(asgn):
            assert len(_profile) == lestimate
        # print(_profile)
        r_cmean = - \
            1 if len(_profile) < lestimate_min else sum(
                _profile) // len(_profile)
    if d == 'l':
        return l_cmean
    elif d == 'r':
        return r_cmean
    else:
        #print(f"@{i}({s}): l_cmean={l_cmean}, r_cmean={r_cmean}")
        return (l_cmean, r_cmean)


def estimate_true_counts_intvl(i, s, d, intvls, asgn, profile,
                               lestimate=5, lestimate_min=1):
    # use max count `l` bp around interval `i` as an estimate of the true counts
    if not isinstance(s, set):
        s = set(s if isinstance(s, list) else [s])
    assert d in ('l', 'r', 'b')
    if d in ('l', 'b'):
        _intvls = [intvls[j] for j in range(i) if asgn[j] in s]
        if len(_intvls) < lestimate_min:
            l_cmean = -1
        else:
            l_cmean = (sum([(e - b) * (profile[b] + profile[e - 1]) / 2 for b, e in _intvls[-lestimate:]])
                       // sum([(e - b) for b, e in _intvls[-lestimate:]]))
    if d in ('r', 'b'):
        _intvls = [intvls[j]
                   for j in range(i + 1, len(intvls)) if asgn[j] in s]
        if len(_intvls) < lestimate_min:
            r_cmean = -1
        else:
            r_cmean = (sum([(e - b) * (profile[b] + profile[e - 1]) / 2 for b, e in _intvls[:lestimate]])
                       // sum([(e - b) for b, e in _intvls[:lestimate]]))
    if d == 'l':
        return l_cmean
    elif d == 'r':
        return r_cmean
    else:
        #print(f"@{i}({s}): l_cmean={l_cmean}, r_cmean={r_cmean}")
        return (l_cmean, r_cmean)


def logp_e(i, intvls, asgn, profile, pread, DEPTHS, verbose):
    # return -np.inf
    if verbose:
        print("    ERROR:")
    ib, ie = intvls[i]
    if ib == 0:
        logp_l = poisson.logpmf(profile[ib], DEPTHS['E'])
        if verbose:
            print(f"        [no prev count]")
    else:
        logp_po = poisson.logpmf(profile[ib], DEPTHS['E'])
        logp_er = np.log(pread.pe["self"]["drop"][ib])
        if verbose:
            print(
                f"        [B ERROR] {profile[ib - 1]} @ {ib - 1} -> {profile[ib]} @ {ib}")
            print(
                f"          logp(PO)={logp_po:.3f} {'***' if logp_po >= logp_er else ''}")
            print(
                f"          logp(ER)={logp_er:.3f} {'***' if logp_er >= logp_po else ''}")
        #logp_l = logp_er
        logp_l = max(logp_po, logp_er)
        #logp_l = min(logp_po, logp_er)
    if ie == len(profile):
        logp_r = poisson.logpmf(profile[ie - 1], DEPTHS['E'])
        if verbose:
            print(f"        [no next count]")
    else:
        logp_po = poisson.logpmf(profile[ie - 1], DEPTHS['E'])
        logp_er = np.log(pread.pe["self"]["gain"][ie])
        if verbose:
            print(
                f"        [E ERROR] {profile[ie - 1]} @ {ie - 1} -> {profile[ie]} @ {ie}")
            print(
                f"          logp(PO)={logp_po:.3f} {'***' if logp_po >= logp_er else ''}")
            print(
                f"          logp(ER)={logp_er:.3f} {'***' if logp_er >= logp_po else ''}")
        #logp_r = logp_er
        logp_r = max(logp_po, logp_er)
        #logp_r = min(logp_po, logp_er)
    """
    if ib == 0 and ie == len(profile):
        return 
    elif ib == 0:
        logp_l = logp_r
    elif ie == len(profile):
        logp_r = logp_l
    """
    return logp_l + logp_r
    # return min(logp_l, logp_r)


def logp_r(i, intvls, asgn, profile, DEPTHS, verbose, n_sigma=3, lread=20000):
    if verbose:
        print("    REPEAT:")
    ib, ie = intvls[i]
    ibc, iec = profile[ib], profile[ie - 1]
    if max(profile[ib], profile[ie - 1]) >= DEPTHS['R']:
        if verbose:
            print(
                f"        Counts are larger than R-depth (={DEPTHS['R']}). Set R.")
        return 0.
    computed_from = "nearest D"
    # NOTE: 1. 5 nearest D-intvls
    #pc, nc = estimate_true_counts_intvl(i, 'D', 'b', intvls, asgn, profile)
    # NOTE: 2. nearest 1k D-mers
    #pc, nc = estimate_true_counts(i, 'D', 'b', intvls, asgn, profile)
    # NOTE: 3. nearest D-mer
    p, n = nn_intvl(i, 'D', 'b', asgn)
    if p >= 0:
        pe = intvls[p][1]
        pc = profile[pe - 1]
    else:
        pe, pc = -1, -1
    if n < len(intvls):
        nb = intvls[n][0]
        nc = profile[nb]
    else:
        nb, nc = -1, -1
    if pc == -1 and nc == -1:
        #computed_from = "nearest H"
        #pc, nc = estimate_true_counts_intvl(i, 'H', 'b', intvls, asgn, profile)
        #pc, nc = estimate_true_counts(i, 'H', 'b', intvls, asgn, profile)
        #p, n = nn_intvl(i, 'H', 'b', asgn)
        #pc = profile[intvls[p][1] - 1] if p >= 0 else -1
        #nc = profile[intvls[n][0]] if n < len(intvls) else -1
        #if pc == -1 and nc == -1:
        computed_from = "initial D-depth"
        pc, nc = DEPTHS['D'], DEPTHS['D']
        #elif pc == -1:
        #    pc = nc
        #elif nc == -1:
        #    nc = pc
    elif pc == -1:
        pc = nc
    elif nc == -1:
        nc = pc

    """
    # D < R must hold
    if ibc < pc or iec < nc:
        if verbose:
            print(
                f"        Counts are smaller than D-depth. No R.")
        return -np.inf
    """

    dr_ratio = 1 + n_sigma * (1 / np.sqrt(DEPTHS['D']))   # X-sigma interval
    pc, nc = pc * dr_ratio, nc * dr_ratio
    if verbose:
        print(
            f"            [L] Imaginary R-depth (= {n_sigma}-sigma of {computed_from}) = {pc:.1f} (-> {ibc})")
    #if pc <= ibc:
    #    if verbose:
    #        print("NG")
    #    return 0.
    #else:
    #    if verbose:
    #        print("OK")
    sf_lambda = pc * (ib - (pe - 1)) / lread
    logp_sf = skellam.logpmf(int(ibc - pc), sf_lambda, sf_lambda)
    logp_er = binom.logpmf(ibc, pc, 1 - 0.01) if ibc < pc else -np.inf   # TODO: need to be binomial test?
    logp_l = max(logp_sf, logp_er)
    if verbose:
        print(
            f"                logp(SF)={logp_sf:.3f} {'*' if logp_sf >= logp_er else ''}")
        print(
            f"                logp(ER)={logp_er:.3f} {'*' if logp_er >= logp_sf else ''}")
    if verbose:
        print(
            f"            [R] Imaginary R-depth (= {n_sigma}-sigma of {computed_from}) = {nc:.1f} (<- {iec})  ")
    #if nc <= iec:
    #    if verbose:
    #        print("NG")
    #    return 0.
    #else:
    #    if verbose:
    #        print("OK")
    sf_lambda = nc * (nb - (ie - 1)) / lread
    logp_sf = skellam.logpmf(int(nc - iec), sf_lambda, sf_lambda)
    logp_er = binom.logpmf(iec, nc, 1 - 0.01) if iec < nc else -np.inf   # TODO: need to be binomial test?
    logp_r = max(logp_sf, logp_er)
    if verbose:
        print(
            f"                logp(SF)={logp_sf:.3f} {'*' if logp_sf >= logp_er else ''}")
        print(
            f"                logp(ER)={logp_er:.3f} {'*' if logp_er >= logp_sf else ''}")
    return max(logp_l, logp_r)


def logp_hd(state, i, intvls, asgn, profile, pread, DEPTHS, verbose, lread=20000):
    ib, ie = intvls[i]
    p, n = nn_intvl(i, state, 'b', asgn)
    if p < 0:
        if verbose:
            print(f"        [no prev {state} intvl]")
    else:
        pb, pe = intvls[p]
        sf_lambda = DEPTHS[state] * (ib - (pe - 1)) / lread
        logp_sf = skellam.logpmf(profile[ib] - profile[pe - 1],
                                 sf_lambda, sf_lambda)
        if pe == ib:
            logp_er = np.log(min(
                pread.corrected_pe["others"]["drop"][ib], pread.corrected_pe["others"]["gain"][ib]))
        else:
            logp_er = -np.inf
        logp_l = max(logp_sf, logp_er)
        if verbose:
            print(
                f"        [B {'DROP' if profile[pe - 1] >= profile[ib] else 'GAIN'}] {profile[pe - 1]} @ {pe - 1} -> {profile[ib]} @ {ib}")
            print(
                f"          logp(SF)={logp_sf:.3f} {'***' if logp_sf >= logp_er else ''}")
            print(
                f"          logp(ER)={logp_er:.3f} {'***' if logp_er >= logp_sf else ''}")
    if n >= len(asgn):
        if verbose:
            print(f"        [no next {state} intvl]")
    else:
        nb, ne = intvls[n]
        sf_lambda = DEPTHS[state] * (nb - (ie - 1)) / lread
        logp_sf = skellam.logpmf(profile[nb] - profile[ie - 1],
                                 sf_lambda, sf_lambda)
        if ie == nb:
            logp_er = np.log(min(
                pread.corrected_pe["others"]["drop"][ie], pread.corrected_pe["others"]["gain"][ie]))
        else:
            logp_er = -np.inf
        logp_r = max(logp_sf, logp_er)
        if verbose:
            print(
                f"        [E {'DROP' if profile[ie - 1] >= profile[nb] else 'GAIN'}] {profile[ie - 1]} @ {ie - 1} -> {profile[nb]} @ {nb}")
            print(
                f"          logp(SF)={logp_sf:.3f} {'***' if logp_sf >= logp_er else ''}")
            print(
                f"          logp(ER)={logp_er:.3f} {'***' if logp_er >= logp_sf else ''}")
    if p < 0 and n >= len(asgn):
        logp_l = poisson.logpmf(profile[ib], DEPTHS[state])
        logp_r = poisson.logpmf(profile[ie - 1], DEPTHS[state])
        if verbose:
            print(f"          logp(L POISSON)={logp_l:.3f}")
            print(f"          logp(R POISSON)={logp_r:.3f}")
    elif p < 0:
        logp_l = logp_r
    elif n >= len(asgn):
        logp_r = logp_l
    # return logp_l + logp_r
    return min(logp_l, logp_r)
    #return max(logp_l, logp_r)


def logp_h(i, intvls, asgn, profile, pread, DEPTHS, verbose, lread=20000, n_sigma=2):
    if verbose:
        print("    HAPLO:")
    # H < nearest D must hold
    if verbose:
        print(f"        [1. Requirment: H < nearest D] ")
    ib, ie = intvls[i]
    ibc, iec = profile[ib], profile[ie - 1]
    p, n = nn_intvl(i, 'D', 'b', asgn)
    if p >= 0:
        pe = intvls[p][1]
        pc = profile[pe - 1]
        if verbose:
            print(
                f"            [L] nearest D count = {pc} @ {pe} (-> {ibc})  ", end='')
        if pc < ibc:
            if verbose:
                print("NG")
            return -np.inf
        else:
            if verbose:
                print("OK")
    if n < len(intvls):
        nb = intvls[n][0]
        nc = profile[nb]
        if verbose:
            print(
                f"            [R] nearest D count = {nc} @ {nb} (<- {iec})  ", end='')
        if nc < iec:
            if verbose:
                print("NG")
            return -np.inf
        else:
            if verbose:
                print("OK")

    # H < (average of nearest 5 D-intvls)/1.25 must hold
    if verbose:
        print(f"        [2. Requirment: H < (mean of 5 D-intvls)/1.25] ")
    ib, ie = intvls[i]
    ibc, iec = profile[ib], profile[ie - 1]
    pc, nc = estimate_true_counts_intvl(i, 'D', 'b', intvls, asgn, profile)
    #if pc == -1 and nc == -1:
    #    pc, nc = DEPTHS['D'], DEPTHS['D']
    #elif pc == -1:
    #    pc = nc
    #elif nc == -1:
    #    nc = pc
    pc, nc = pc / 1.25, nc / 1.25   # TODO: change to X-sigma-interval
    if pc > 0:
        if verbose:
            print(
                f"            [L] nearest 5 D-intvls average count / 1.25 = {pc:.1f} (-> {ibc})  ", end='')
        if pc <= ibc:
            if verbose:
                print("NG")
            return -np.inf
        else:
            if verbose:
                print("OK")
    if nc > 0:
        if verbose:
            print(
                f"            [R] nearest 5 D-intvls average count / 1.25 = {nc:.1f} (<- {iec})  ", end='')
        if nc <= iec:
            if verbose:
                print("NG")
            return -np.inf
        else:
            if verbose:
                print("OK")

    """
    # H >> nearest H must not hold
    p, n = nn_intvl(i, 'H', 'b', asgn)
    pc, nc = profile[intvls[p][1] - 1] if p >= 0 else - \
        1, profile[intvls[n][0]] if n < len(intvls) else -1
    if pc == -1 and nc == -1:
        pc, nc = DEPTHS['H'], DEPTHS['H']
    elif pc == -1:
        pc = nc
    elif nc == -1:
        nc = pc
    hd_ratio = 1 + n_sigma * (1 / np.sqrt(DEPTHS['H']))   # X-sigma interval
    pc, nc = pc * hd_ratio, nc * hd_ratio
    if verbose:
        print(
            f"        [LEFT] RH_est={pc:.3f}, {profile[ib]} ~ [RIGHT] RH_est={nc:.3f}, {profile[ie - 1]}")
    if profile[ib] >= pc or profile[ie - 1] >= nc:
        return -np.inf
    """

    return logp_hd('H', i, intvls, asgn, profile, pread, DEPTHS, verbose, lread)


def logp_d(i, intvls, asgn, profile, pread, DEPTHS, verbose, lread=20000):
    if verbose:
        print("    DIPLO:")
    # H < D
    ib, ie = intvls[i]
    p, n = nn_intvl(i, 'H', 'b', asgn)
    if p >= 0 and n < len(asgn):
        pb, pe = intvls[p]
        nb, ne = intvls[n]
        #if profile[pe - 1] > profile[ib] and profile[nb] > profile[ie - 1]:
        #    return -np.inf

    # TODO: use sigma etc.
    pc, nc = estimate_true_counts_intvl(i, 'H', 'b', intvls, asgn, profile)
    if pc == -1 and nc == -1:
        pc, nc = DEPTHS['H'], DEPTHS['H']
    elif pc == -1:
        pc = nc
    elif nc == -1:
        nc = pc
    pc, nc = pc * 1.25, nc * 1.25
    if verbose:
        print(
            f"[LEFT] H_est={pc}, {profile[ib]} ~ [RIGHT] H_est={nc}, {profile[ie - 1]}")
    if profile[ib] < pc and profile[ie - 1] < nc:
        return -np.inf

    return logp_hd('D', i, intvls, asgn, profile, pread, DEPTHS, verbose, lread)

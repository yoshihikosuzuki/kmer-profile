from typing import Sequence, List, Tuple, Dict
import numpy as np
from scipy.stats import binom, binom_test, poisson, skellam, norm
from logzero import logger
from ..type import STATES


def assign_init(intvls, profile, DEPTHS):
    asgn = ''
    for b, e in intvls:
        dmin = 32768
        smin = None
        for s in STATES:   # TODO: binary search
            d = abs(DEPTHS[s] - max(profile[b], profile[e - 1]))
            if d < dmin:
                smin = s
            dmin = d
        asgn += smin
    return asgn


def asgn_to_states(asgn, intvls):
    return ''.join([s * (e - b) for (b, e), s in zip(intvls, asgn)])


def assign_update(intvls, asgn, profile, pread, DEPTHS, max_n_iter=15, verbose=False):
    asgn = list(asgn)
    changed = True
    count = 0
    while changed and count < max_n_iter:
        print(".", end='')
        changed = False
        for i in (list(range(len(asgn))) + list(reversed(range(len(asgn))))):
            new_s = update_state(i, intvls, asgn, profile, pread, DEPTHS, verbose)
            if new_s != asgn[i]:
                changed = True
                asgn[i] = new_s
        count += 1
        # print(''.join(assignments))
    if count == max_n_iter:
        print("Not converged")
    return ''.join(asgn)


def update_state(i, intvls, asgn, profile, pread, DEPTHS, verbose=False):
    if not isinstance(asgn, list):
        asgn = list(asgn)
    max_logp = -np.inf
    max_s = None
    for s in STATES:
        logp = (logp_e(i, intvls, asgn, profile, pread, DEPTHS, verbose) if s == 'E'
                else logp_r(i, intvls, asgn, profile, DEPTHS, verbose) if s == 'R'
                else logp_h(i, intvls, asgn, profile, pread, DEPTHS, verbose) if s == 'H'
                else logp_d(i, intvls, asgn, profile, pread, DEPTHS, verbose))
        if verbose:
            print(f"intvl {intvls[i]}: logp state {s} = {logp}")
        if logp > max_logp:
            max_logp = logp
            max_s = s
    if max_s != asgn[i]:
        if verbose:
            print(f"State updated @{i}: {asgn[i]} -> {max_s}")
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
        _intvls = [intvls[j] for j in range(i + 1, len(intvls)) if asgn[j] in s]
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
    #return -np.inf
    if verbose:
        print("### ERROR ###")
    ib, ie = intvls[i]
    if ib == 0:
        logp_l = poisson.logpmf(profile[ib], DEPTHS['E'])
        if verbose:
            print(f"[no prev count]")
    else:
        logp_po = poisson.logpmf(profile[ib], DEPTHS['E'])
        logp_er = np.log(pread.pe["self"]["drop"][ib])
        if verbose:
            print(
                f"[B ERROR] {profile[ib - 1]} @ {ib - 1} -> {profile[ib]} @ {ib}")
            print(
                f"  logp(PO)={logp_po} {'***' if logp_po >= logp_er else ''}")
            print(
                f"  logp(ER)={logp_er} {'***' if logp_er >= logp_po else ''}")
        #logp_l = logp_er
        logp_l = max(logp_po, logp_er)
        #logp_l = min(logp_po, logp_er)
    if ie == len(profile):
        logp_r = poisson.logpmf(profile[ie - 1], DEPTHS['E'])
        if verbose:
            print(f"[no next count]")
    else:
        logp_po = poisson.logpmf(profile[ie - 1], DEPTHS['E'])
        logp_er = np.log(pread.pe["self"]["gain"][ie])
        if verbose:
            print(
                f"[E ERROR] {profile[ie - 1]} @ {ie - 1} -> {profile[ie]} @ {ie}")
            print(
                f"  logp(PO)={logp_po} {'***' if logp_po >= logp_er else ''}")
            print(
                f"  logp(ER)={logp_er} {'***' if logp_er >= logp_po else ''}")
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
    #return min(logp_l, logp_r)


def logp_r(i, intvls, asgn, profile, DEPTHS, verbose):
    if verbose:
        print("### REPEAT ###")
    ib, ie = intvls[i]
    if max(profile[ib], profile[ie - 1]) >= DEPTHS['R']:
        return 0.
    pc, nc = estimate_true_counts_intvl(i, 'D', 'b', intvls, asgn, profile)
    if pc == -1 and nc == -1:
        pc, nc = estimate_true_counts_intvl(i, 'H', 'b', intvls, asgn, profile)
        if pc == -1 and nc == -1:
            pc, nc = DEPTHS['D'], DEPTHS['D']
        elif pc == -1:
            pc = nc
        elif nc == -1:
            nc = pc
    elif pc == -1:
        pc = nc
    elif nc == -1:
        nc = pc
    pc, nc = pc * 1.25, nc * 1.25
    if verbose:
        print(f"[LEFT] R_est={pc}, {profile[ib]} ~ [RIGHT] R_est={nc}, {profile[ie - 1]}")
    if profile[ib] >= pc or profile[ie - 1] >= nc:
        return 0.
    else:
        return binom.logpmf(profile[ib], pc, 1 - 0.01) + binom.logpmf(profile[ie - 1], nc, 1 - 0.01)
    """
    logp_max = max(logp_e(i, intvls, asgn, profile, verbose=False),
                   logp_h(i, intvls, asgn, profile, verbose=False),
                   logp_d(i, intvls, asgn, profile, verbose=False))
    if verbose:
        print(f"logp_max={logp_max}")
    if logp_max <= -20:
        return 0.
    else:
        return -np.inf
    """


def logp_hd(state, i, intvls, asgn, profile, pread, DEPTHS, verbose, lread=20000):
    ib, ie = intvls[i]
    p, n = nn_intvl(i, state, 'b', asgn)
    if p < 0:
        if verbose:
            print(f"[no prev {state} intvl]")
    else:
        pb, pe = intvls[p]
        sf_lambda = DEPTHS[state] * (ib - (pe - 1)) / lread
        logp_sf = skellam.logpmf(profile[ib] - profile[pe - 1],
                                 sf_lambda, sf_lambda)
        if pe == ib:
            logp_er = np.log(min(pread.corrected_pe["others"]["drop"][ib], pread.corrected_pe["others"]["gain"][ib]))
        else:
            logp_er = -np.inf
        logp_l = max(logp_sf, logp_er)
        if verbose:
            print(
                f"[B {'DROP' if profile[pe - 1] >= profile[ib] else 'GAIN'}] {profile[pe - 1]} @ {pe - 1} -> {profile[ib]} @ {ib}")
            print(
                f"  logp(SF)={logp_sf} {'***' if logp_sf >= logp_er else ''}")
            print(
                f"  logp(ER)={logp_er} {'***' if logp_er >= logp_sf else ''}")
    if n >= len(asgn):
        if verbose:
            print(f"[no next {state} intvl]")
    else:
        nb, ne = intvls[n]
        sf_lambda = DEPTHS[state] * (nb - (ie - 1)) / lread
        logp_sf = skellam.logpmf(profile[nb] - profile[ie - 1],
                                 sf_lambda, sf_lambda)
        if ie == nb:
            logp_er = np.log(min(pread.corrected_pe["others"]["drop"][ie], pread.corrected_pe["others"]["gain"][ie]))
        else:
            logp_er = -np.inf
        logp_r = max(logp_sf, logp_er)
        if verbose:
            print(
                f"[E {'DROP' if profile[ie - 1] >= profile[nb] else 'GAIN'}] {profile[ie - 1]} @ {ie - 1} -> {profile[nb]} @ {nb}")
            print(
                f"  logp(SF)={logp_sf} {'***' if logp_sf >= logp_er else ''}")
            print(
                f"  logp(ER)={logp_er} {'***' if logp_er >= logp_sf else ''}")
    if p < 0 and n >= len(asgn):
        logp_l = poisson.logpmf(profile[ib], DEPTHS[state])
        logp_r = poisson.logpmf(profile[ie - 1], DEPTHS[state])
        if verbose:
            print(f"  logp(L POISSON)={logp_l}")
            print(f"  logp(R POISSON)={logp_r}")
    elif p < 0:
        logp_l = logp_r
    elif n >= len(asgn):
        logp_r = logp_l
    #return logp_l + logp_r
    return min(logp_l, logp_r)
    #return max(logp_l, logp_r)


def logp_h(i, intvls, asgn, profile, pread, DEPTHS, verbose, lread=20000):
    if verbose:
        print("### HAPLO ###")
    # H < D
    ib, ie = intvls[i]
    p, n = nn_intvl(i, 'D', 'b', asgn)
    if p >= 0 and n < len(asgn):
        pb, pe = intvls[p]
        nb, ne = intvls[n]
        if profile[pe - 1] < profile[ib] and profile[nb] < profile[ie - 1]:
            return -np.inf
    
    pc, nc = estimate_true_counts_intvl(i, 'D', 'b', intvls, asgn, profile)
    if pc == -1 and nc == -1:
        pc, nc = DEPTHS['D'], DEPTHS['D']
    elif pc == -1:
        pc = nc
    elif nc == -1:
        nc = pc
    pc, nc = pc / 1.25, nc / 1.25
    if verbose:
        print(f"[LEFT] H_est={pc}, {profile[ib]} ~ [RIGHT] H_est={nc}, {profile[ie - 1]}")
    if profile[ib] > pc and profile[ie - 1] > nc:
        return -np.inf

    return logp_hd('H', i, intvls, asgn, profile, pread, DEPTHS, verbose, lread)


def logp_d(i, intvls, asgn, profile, pread, DEPTHS, verbose, lread=20000):
    if verbose:
        print("### DIPLO ###")
    # H < D
    ib, ie = intvls[i]
    p, n = nn_intvl(i, 'H', 'b', asgn)
    if p >= 0 and n < len(asgn):
        pb, pe = intvls[p]
        nb, ne = intvls[n]
        if profile[pe - 1] > profile[ib] and profile[nb] > profile[ie - 1]:
            return -np.inf

    pc, nc = estimate_true_counts_intvl(i, 'H', 'b', intvls, asgn, profile)
    if pc == -1 and nc == -1:
        pc, nc = DEPTHS['H'], DEPTHS['H']
    elif pc == -1:
        pc = nc
    elif nc == -1:
        nc = pc
    pc, nc = pc * 1.25, nc * 1.25
    if verbose:
        print(f"[LEFT] H_est={pc}, {profile[ib]} ~ [RIGHT] H_est={nc}, {profile[ie - 1]}")
    if profile[ib] < pc and profile[ie - 1] < nc:
        return -np.inf

    return logp_hd('D', i, intvls, asgn, profile, pread, DEPTHS, verbose, lread)
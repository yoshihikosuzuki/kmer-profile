
from typing import Sequence, List, Tuple, Dict
import numpy as np
from scipy.stats import binom, binom_test, poisson, skellam, norm
from logzero import logger
from ..type import STATES
from .classifiy_reliable import nn_intvl, estimate_true_counts_intvl, logp_e


def logp_r_short(i, intvls, asgn, profile, DEPTHS, verbose):
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
    pc, nc = pc * 1.5, nc * 1.5
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
    

def logp_hd_short2(state, i, intvls, asgn, profile, pread, DEPTHS, verbose, lread=20000):
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
            logp_er = np.log(min(pread.pe["others"]["drop"][ib], pread.pe["others"]["gain"][ib]))
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
            logp_er = np.log(min(pread.pe["others"]["drop"][ie], pread.pe["others"]["gain"][ie]))
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
        if verbose:
            print(
                f"[ISOLATED] {profile[ib]} @ {ib} -- {profile[ie - 1]} @ {ie - 1}")
        if ie - ib >= 100:
            logp_l = poisson.logpmf(profile[ib], DEPTHS[state])
            logp_r = poisson.logpmf(profile[ie - 1], DEPTHS[state])
            if verbose:
                print(f"  logp(L POISSON)={logp_l}")
                print(f"  logp(R POISSON)={logp_r}")
        else:
            if verbose:
                print("  too short")
            return -np.inf
    elif p < 0:
        logp_l = logp_r
    elif n >= len(asgn):
        logp_r = logp_l
    #return logp_l + logp_r
    #return min(logp_l, logp_r)
    return max(logp_l, logp_r)


def logp_h_short(i, intvls, asgn, profile, pread, DEPTHS, verbose, lread=20000):
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
        print(
            f"[LEFT] H_est={pc}, {profile[ib]} ~ [RIGHT] H_est={nc}, {profile[ie - 1]}")
    if profile[ib] > pc and profile[ie - 1] > nc:
        return -np.inf
    return logp_hd_short2('H', i, intvls, asgn, profile, pread, DEPTHS, verbose, lread)


def logp_d_short(i, intvls, asgn, profile, pread, DEPTHS, verbose, lread=20000):
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
        print(
            f"[LEFT] H_est={pc}, {profile[ib]} ~ [RIGHT] H_est={nc}, {profile[ie - 1]}")
    if profile[ib] < pc and profile[ie - 1] < nc:
        return -np.inf
    return logp_hd_short2('D', i, intvls, asgn, profile, pread, DEPTHS, verbose, lread)


def update_state_short(i, intvls, asgn, profile, pread, DEPTHS, verbose):
    if not isinstance(asgn, list):
        asgn = list(asgn)
    max_logp = -np.inf
    max_s = None
    for s in STATES:
        logp = (logp_e(i, intvls, asgn, profile, pread, DEPTHS, verbose) if s == 'E'
                else logp_r_short(i, intvls, asgn, profile, DEPTHS, verbose) if s == 'R'
                else logp_h_short(i, intvls, asgn, profile, pread, DEPTHS, verbose) if s == 'H'
                else logp_d_short(i, intvls, asgn, profile, pread, DEPTHS, verbose))
        if verbose:
            print(f"intvl {intvls[i]}: logp state {s} = {logp}")
        if logp > max_logp:
            max_logp = logp
            max_s = s
    if max_s != asgn[i]:
        if verbose:
            print(f"State updated @{i}: {asgn[i]} -> {max_s}")
    return max_s


def check_sparsity(states, wlen=1000, thres_p=0.1):
    # print(states)
    for i in range(-(-len(states) // wlen)):
        s, t = i * wlen, (i + 1) * wlen
        p_normal = len(
            list(filter(lambda state: state in ('H', 'D'), states[s:t]))) / wlen
        if p_normal < thres_p:
            print(f"[{s},{t}): {p_normal}")
            for j in range(s, min(t, len(states))):
                # print(j)
                states[j] = 'E' if states[j] == 'E' else 'R'
    # print(states)
    return states


def get_states(long_merged_smooth_intvls, asgn, profile, merged_intvls, pread, DEPTHS, verbose=False):
    lmsi_to_asgn = dict(zip(long_merged_smooth_intvls, asgn))
    asgn_merged = [lmsi_to_asgn[intvl]
                   if intvl in lmsi_to_asgn and lmsi_to_asgn[intvl] != 'E' else '-' for intvl in merged_intvls]
    i_updates = set([i for i, state in enumerate(asgn_merged) if state == '-'])

    # TEMP: check entire sparsity
    states = ['E'] * len(profile)
    for (s, t), state in zip(long_merged_smooth_intvls, asgn):
        for i in range(s, t):
            states[i] = state
    p_normal = len(
        list(filter(lambda state: state in ('H', 'D'), states))) / len(states)

    logger.info(f"%Normal={p_normal * 100}")
    if p_normal >= 0.1:
        logger.info("Not too repetitive. Classify unreliable intervals")
    else:
        logger.info("Too repetitive. Do not classify unreliable intervals")

    for i, s, t, c, state in sorted([(j, merged_intvls[j][0], merged_intvls[j][1],
                               min(pread.counts[merged_intvls[j][0]], pread.counts[merged_intvls[j][1] - 1]),
                               state)
                              for j, state in enumerate(asgn_merged)], key=lambda x: x[3], reverse=True):
        # for i, state in enumerate(asgn_merged):
        if verbose:
            print(f"\n### {s, t, c}")
        #if state == '-':
        if i in i_updates:
            if p_normal >= 0.1:
                asgn_merged[i] = update_state_short(
                    i, merged_intvls, asgn_merged, profile, pread, DEPTHS, verbose)
            else:
                asgn_merged[i] = 'R'
    for i, s, t, c, state in sorted([(j, merged_intvls[j][0], merged_intvls[j][1],
                               min(pread.counts[merged_intvls[j][0]], pread.counts[merged_intvls[j][1] - 1]),
                               state)
                              for j, state in enumerate(asgn_merged)], key=lambda x: x[3], reverse=False):
        # for i, state in enumerate(asgn_merged):
        if verbose:
            print(f"\n### {s, t, c}")
        #if state == '-':
        if i in i_updates:
            if p_normal >= 0.1:
                asgn_merged[i] = update_state_short(
                    i, merged_intvls, asgn_merged, profile, pread, DEPTHS, verbose)
            else:
                asgn_merged[i] = 'R'
    states = ['E'] * len(profile)
    for (s, t), state in zip(merged_intvls, asgn_merged):
        for i in range(s, t):
            states[i] = state
    return states


def remove_slips(pread) -> None:
    def _updown(i):
        if pread.counts[i] == pread.counts[i - 1]:
            return 1
        else:
            return (pread.counts[i] - pread.counts[i - 1]) // abs(pread.counts[i] - pread.counts[i - 1])

    slip_intvls = []
    prev_s = None
    base = None
    updown = None
    hp = False
    start = -1
    for i, s in enumerate(pread.states):
        if s != prev_s:
            if hp:
                if _updown(i) == updown:
                    left_s, right_s = pread.states[start - 1], pread.states[i]
                    #print(start, i, left_s, right_s)
                    mod_s = ('E' if 'E' in (left_s, right_s)
                             else prev_s)
                    slip_intvls.append((start, i, mod_s))
            hp = True
            start = i
            prev_s = s
            updown = _updown(i)
            if updown == -1:
                base = pread.seq[i]
            else:
                base = pread.seq[i - pread.K + 1]
        if pread.counts[i] == pread.counts[i - 1]:
            hp = False
        if _updown(i) != updown:
            hp = False
        else:
            if updown == -1:
                if base != pread.seq[i]:
                    hp = False
            else:
                if base != pread.seq[i - pread.K + 1]:
                    hp = False

    logger.debug(f"Slip intvls = {slip_intvls}")
    for i, j, s in slip_intvls:
        for k in range(i, j):
            pread.states[k] = s

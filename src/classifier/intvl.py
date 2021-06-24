from typing import Tuple, Dict
from copy import deepcopy
import numpy as np
from scipy.stats import skellam
from logzero import logger
from ..type import ProfiledRead


def find_ns_points(pread: ProfiledRead,
                   thres_p_self: float = 0.001,
                   thres_p_others: float = 0.05,
                   min_change: int = 2,
                   max_change: int = 5,
                   thres_r: int = 1000) -> None:
    """Find every position i where count change from (i-1) -> i
    can be a state change. This is just for reducing the size of
    candidates of state change points and thus should be performed
    conservatively, i.e. false positives are OK but false negatives are NG.
    """
    pread.ns = [(max(pread.pe["self"]["drop"][i],
                     pread.pe["self"]["gain"][i]) > thres_p_self
                 or min(pread.pe["others"]["drop"][i],
                        pread.pe["others"]["gain"][i]) < thres_p_others)
                for i in range(pread.length)]
    for i in range(1, pread.length):
        if abs(pread.counts[i] - pread.counts[i - 1]) <= min_change:
            pread.ns[i] = False
        elif abs(pread.counts[i] - pread.counts[i - 1]) > max_change:
            pread.ns[i] = True
    for i in range(1, pread.length):
        if pread.ns[i] and min(pread.counts[i - 1:i + 1]) >= thres_r:
            pread.ns[i] = False
    logger.info(f"{pread.length} counts -> {sum(pread.ns)} non-smooth points")


def ns_to_intvls(pread: ProfiledRead) -> None:
    chpt = [0] + [i for i, x in enumerate(pread.ns) if x] + [pread.length]
    pread.intvls = [(chpt[i], chpt[i + 1]) for i in range(len(chpt) - 1)]


def p_diff_changes(i,
                   j,
                   pread,
                   rlen=20000) -> float:
    # TODO: when within E-intervals?
    n_drop = pread.counts[i - 1] - pread.counts[i]
    n_gain = pread.counts[j] - pread.counts[j - 1]
    l = max(pread.counts[i - 1], pread.counts[j]) / rlen
    return skellam.pmf(n_drop - n_gain, l, l)


def check_drop(i,
               pread,
               thres_p_diff=1e-20,
               verbose=False) -> Tuple[float, Tuple[int, int], float, Tuple[int, int]]:
    if verbose:
        print(f"Count: {pread.counts[i - 1]} -> {pread.counts[i]}")
    max_p_self, max_p_others = 0., 0.
    max_intvl_self, max_intvl_others = None, None
    p_self_drop, p_others_drop = [pread.pe[etype]
                                  ['drop'][i] for etype in ("self", "others")]
    ipk = i - 1 + pread.K
    if verbose:
        print(f"j >= {ipk} + n - m")
    # Unique sequence change
    m = 0
    for n in range(10):
        # TODO: check if the sequence is possible given m,n (only for self?)
        j = ipk + n - m
        if j >= pread.length:
            continue
        if verbose:
            print(
                f"j = {j} (m = {m}, n = {n}): {pread.counts[j - 1]} -> {pread.counts[j]}")
        p_self_gain, p_others_gain = [
            pread.pe[etype]['gain'][j] for etype in ("self", "others")]
        p_self = p_self_drop * p_self_gain * \
            (pread.ctx[0].emodel.pe(1) ** max(1, n))
        p_others = p_others_drop * p_others_gain * \
            (pread.ctx[0].emodel.pe(1) ** max(1, n))
        p_diff = p_diff_changes(i, j, pread)
        if verbose:
            print(
                f"*** Pr(self), Pr(others) = {p_self}, {p_others}, p_diff = {p_diff}")
        if p_self > max_p_self:
            max_p_self = p_self
            max_intvl_self = (i, j)
        # NOTE: filtering by p_diff because errors in others spanning two states can be relatively easily corrected
        if p_diff > thres_p_diff and p_others > max_p_others:
            max_p_others = p_others
            max_intvl_others = (i, j)
    # Non-unique sequence change
    for ulen, ctx in enumerate(pread.ctx, start=1):
        m = ctx.lens[0][i - 1] * ulen
        if m == 0:
            continue
        useq = pread.seq[i - ulen:i]
        n = 0   # TODO: partial unit for DS
        while pread.seq[i + n:i + n + ulen] == useq:
            n += ulen
        if verbose:
            print(f"{ctx.emodel.name} length = {m} (left), {n} (right)")
        j = ipk + n - m
        if j >= pread.length:
            continue
        if verbose:
            print(
                f"j = {j} (m = {m}, n = {n}): {pread.counts[j - 1]} -> {pread.counts[j]}")
        p_self_gain, p_others_gain = [
            pread.pe[etype]['gain'][j] for etype in ("self", "others")]
        p_self = p_self_drop * p_self_gain
        p_others = p_others_drop * p_others_gain
        p_diff = p_diff_changes(i, j, pread)
        if verbose:
            print(
                f"*** Pr(self), Pr(others) = {p_self}, {p_others}, p_diff = {p_diff}")
        if p_self > max_p_self:
            max_p_self = p_self
            max_intvl_self = (i, j)
        if p_diff > thres_p_diff and p_others > max_p_others:
            max_p_others = p_others
            max_intvl_others = (i, j)
        # TODO: self Error -> self Error
    return (max_p_self, max_intvl_self, max_p_others, max_intvl_others)


def check_gain(i,
               pread,
               thres_p_diff=1e-20,
               verbose=False) -> Tuple[float, Tuple[int, int], float, Tuple[int, int]]:
    if verbose:
        print(f"Count: {pread.counts[i - 1]} -> {pread.counts[i]}")
    max_p_self, max_p_others = 0., 0.
    max_intvl_self, max_intvl_others = None, None
    p_self_gain, p_others_gain = [pread.pe[etype]
                                  ['gain'][i] for etype in ("self", "others")]
    ipk = i + 1 - pread.K
    if verbose:
        print(f"j >= {ipk} + n - m")
    # Unique sequence change
    m = 0
    for n in range(10):
        # TODO: check if the sequence is possible given m,n (only for self?)
        j = ipk - n + m
        if j < 0:
            continue
        if verbose:
            print(
                f"j = {j} (m = {m}, n = {n}): {pread.counts[j - 1]} -> {pread.counts[j]}")
        p_self_drop, p_others_drop = [
            pread.pe[etype]['drop'][j] for etype in ("self", "others")]
        p_self = p_self_drop * p_self_gain * \
            (pread.ctx[0].emodel.pe(1) ** max(1, n))
        p_others = p_others_drop * p_others_gain * \
            (pread.ctx[0].emodel.pe(1) ** max(1, n))
        p_diff = p_diff_changes(j, i, pread)
        if verbose:
            print(
                f"*** Pr(self), Pr(others) = {p_self}, {p_others}, p_diff = {p_diff}")
        if p_self > max_p_self:
            max_p_self = p_self
            max_intvl_self = (j, i)
        if p_diff > thres_p_diff and p_others > max_p_others:
            max_p_others = p_others
            max_intvl_others = (j, i)
    # Non-unique sequence change
    for ulen, ctx in enumerate(pread.ctx, start=1):
        m = ctx.lens[1][i] * ulen
        if m == 0:
            continue
        useq = pread.seq[i - pread.K + 1:i - pread.K + 1 + ulen]
        n = 0   # TODO: partial unit for DS
        while pread.seq[i - pread.K + 1 - n - ulen:i - pread.K + 1 - n] == useq:
            n += ulen
        if verbose:
            print(f"{ctx.emodel.name} length = {m} (left), {n} (right)")
        j = ipk - n + m
        if j < 0:
            continue
        if verbose:
            print(
                f"j = {j} (m = {m}, n = {n}): {pread.counts[j - 1]} -> {pread.counts[j]}")
        p_self_drop, p_others_drop = [
            pread.pe[etype]['drop'][j] for etype in ("self", "others")]
        p_self = p_self_drop * p_self_gain
        p_others = p_others_drop * p_others_gain
        p_diff = p_diff_changes(j, i, pread)
        if verbose:
            print(
                f"*** Pr(self), Pr(others) = {p_self}, {p_others}, p_diff = {p_diff}")
        if p_self > max_p_self:
            max_p_self = p_self
            max_intvl_self = (j, i)
        if p_diff > thres_p_diff and p_others > max_p_others:
            max_p_others = p_others
            max_intvl_others = (j, i)
    return (max_p_self, max_intvl_self, max_p_others, max_intvl_others)


def p_e_intvl_len(l,
                  pread) -> float:
    return pread.ctx[0].emodel.pe(1) ** (l // (pread.K / 2))


def calc_pe_intvls(pread,
                   thres_p_self_each=1e-5,
                   verbose=False) -> Tuple[Dict[Tuple[int, int], float],
                                           Dict[Tuple[int, int], float]]:
    """
    optional arguments:
      @ thres_p_self_each : Probabilies are computed for every smooth interval
                            that has Pr(error in self | count drop) or Pr(...|gain)
                            higher than this threshold at its ends.
    """
    pe_intvls = {}
    po_intvls = {}
    for s, t, c in sorted([(s, t, min(pread.counts[s], pread.counts[t - 1]))
                           for s, t in pread.intvls], key=lambda x: x[2]):
        if verbose:
            print(f"\n### {s, t, c}")
        max_p_self_s, max_p_self_t = 0., 0.
        max_p_others_s, max_p_others_t = 0., 0.
        max_intvl_self_s, max_intvl_others_s, max_intvl_self_t, max_intvl_others_t = [None] * 4
        if s > 0 and pread.counts[s - 1] > pread.counts[s]:
            # Single error event
            if verbose:
                print(f"\n## s={s}")
            max_p_self_s, max_intvl_self_s, max_p_others_s, max_intvl_others_s = check_drop(
                s, pread, verbose=verbose)
            if verbose:
                print(max_p_self_s, max_intvl_self_s,
                      max_p_others_s, max_intvl_others_s)
        if t < pread.length and pread.counts[t - 1] < pread.counts[t]:
            if verbose:
                print(f"\n## t={t}")
            max_p_self_t, max_intvl_self_t, max_p_others_t, max_intvl_others_t = check_gain(
                t, pread, verbose=verbose)
            if verbose:
                print(max_p_self_t, max_intvl_self_t,
                      max_p_others_t, max_intvl_others_t)
        if max(pread.pe["self"]["drop"][s],
               pread.pe["self"]["gain"][t] if t < pread.length else 0) < thres_p_self_each:
            # This interval must not be an E-interval
            pe_intvls[(s, t)] = 0.
        else:
            max_p_self_mult = 0.
            if (t - s) >= pread.K and s > 0 and pread.counts[s - 1] > pread.counts[s] and t < pread.length and pread.counts[t - 1] < pread.counts[t]:
                # This single interval is due to multiple error events
                max_p_self_mult = pread.pe["self"]["drop"][s] * \
                    pread.pe["self"]["gain"][t] * p_e_intvl_len(t - s, pread)
                if max_p_self_mult >= max(max_p_self_s, max_p_self_t):
                    if verbose:
                        logger.debug(f"({s},{t}) mult! p={max_p_self_mult}")
                    if max_p_self_mult < max(max_p_others_s, max_p_others_t):
                        if verbose:
                            logger.warn(f"({s},{t}) Mult < Others!!!")
                    pe_intvls[(s, t)] = max_p_self_mult
                    continue
            if max(max_p_self_s, max_p_self_t) <= max(max_p_others_s, max_p_others_t):
                continue
            pe_intvls[max_intvl_self_s] = max(
                max_p_self_s, pe_intvls[max_intvl_self_s] if max_intvl_self_s in pe_intvls else 0.)
            pe_intvls[max_intvl_self_t] = max(
                max_p_self_t, pe_intvls[max_intvl_self_t] if max_intvl_self_t in pe_intvls else 0.)
        po_intvls[max_intvl_others_s] = max(
            max_p_others_s, po_intvls[max_intvl_others_s] if max_intvl_others_s in po_intvls else 0.)
        po_intvls[max_intvl_others_t] = max(
            max_p_others_t, po_intvls[max_intvl_others_t] if max_intvl_others_t in po_intvls else 0.)
    return pe_intvls, po_intvls


def correct_intvls(pread,
                   verbose=False) -> None:
    # Correct boundary counts within an interval for each merged interval
    pread.corrected_counts = deepcopy(pread.counts)
    for s, t in pread.long_none_intvls:
        n_gain = ([max(pread.counts[i + 1] - pread.counts[i], 0)
                   for i in range(s, min(s + pread.K - 1, t - 1))]
                  + ([-max(pread.counts[i] - pread.counts[i + 1], 0)
                      for i in range(s, s + max(pread.ctx[0].lens[1][s + pread.K - 1],
                                                pread.ctx[1].lens[1][s + pread.K - 1] * 2))]
                     if s + pread.K - 1 < t else []))
        n_drop = ([max(pread.counts[i] - pread.counts[i + 1], 0)
                   for i in range(max(t - pread.K + 1, s), t - 1)]
                  + ([-max(pread.counts[i + 1] - pread.counts[i], 0)
                      for i in range(t - max(pread.ctx[0].lens[0][t - pread.K + 1],
                                             pread.ctx[1].lens[0][t - pread.K + 1] * 2), t - 1)]
                     if s < t - pread.K + 1 else []))
        if verbose:
            print(
                f"intvl [{s}, {t}): gains={np.array(n_gain)}, drops={np.array(n_drop)}")
        # NOTE: Only (s, t-1) for each long non-E intervals are corrected
        cc_pattern_s = pread.counts[s] + max(sum(n_gain), 0)
        cc_pattern_tm1 = pread.counts[t - 1] + max(sum(n_drop), 0)

        cc_naive_s = max(pread.counts[s:min(s + 2 * pread.K, t)])
        cc_naive_tm1 = max(pread.counts[max(t - 2 * pread.K, s):t])
        #cc_naive_s = cc_naive_tm1 = 0

        pread.corrected_counts[s] = max(cc_pattern_s, cc_naive_s)
        pread.corrected_counts[t - 1] = max(cc_pattern_tm1, cc_naive_tm1)


def remove_ns_intvsl(pread,
                     rlen=20000,
                     thres_p=0.0001,
                     verbose=False) -> None:
    pread.long_none_smooth_intvls = list(filter(lambda intvl: is_smooth(intvl, pread.corrected_counts),
                                                pread.long_none_intvls))


def is_smooth(intvl,
              profile,
              rlen=20000,
              thres_p=0.0001,
              verbose=False) -> bool:
    # Check if an interval is smooth (use after smoothing)
    s, t = intvl
    l = (profile[s] + profile[t - 1]) / 2 * (t - s) / rlen
    p = skellam.pmf(profile[s] - profile[t - 1], l, l)
    if p < thres_p and verbose:
        print(f"[{s},{t}) {profile[s]}->{profile[t-1]} ({t-s} bp, {abs(profile[s]-profile[t-1])} diff) p={p:.5f}")
    return p >= thres_p

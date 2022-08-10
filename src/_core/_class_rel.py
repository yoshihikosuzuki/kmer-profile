from dataclasses import dataclass, field, InitVar
from typing import Optional, NamedTuple, Union, Tuple, List, Dict, Set
from copy import deepcopy
from math import inf, log, exp
from scipy.stats import poisson, binom
from logzero import logger
from .. import STATES, StateT, Intvl, ProfiledRead, OFFSET, E_PO_BASE, R_LOGP
from .._plot import color_asgn
from .._main import ClassParams
from ._util import calc_logp_trans


def is_eq_prefix(eqs):
    if eqs[0] != True:
        return False
    i = 1
    while i < len(eqs) and eqs[i]:
        i += 1
    while i < len(eqs):
        if eqs[i]:
            return False
        i += 1
    return True

def is_eq_suffix(eqs):
    if eqs[len(eqs) - 1] != True:
        return False
    i = len(eqs) - 2
    while i >= 0 and eqs[i]:
        i -= 1
    while i >= 0:
        if eqs[i]:
            return False
        i -= 1
    return True


def classify_rel(pread, cp, verbose):
    cr_f, df, hf, hdrrf = classify_rel_fw(pread, cp, verbose)
    cr_b, db, hb, hdrrb = classify_rel_bw(pread, cp, verbose)
    
    if verbose:
        print(f"  FWD: {color_asgn(cr_f.asgn)}")
        if cr_f.asgn == cr_b.asgn:
            print(f"BWD: (= FWD)")
        else:
            print(f"  BWD: {color_asgn(cr_b.asgn)}")
            print(f"hdrr FWD={hdrrf}, BWD={hdrrb}")

    # By the ratio of HD-ratios (very slightly better then above)
    # cr = cr_f if abs(hdrrf - 1.) <= abs(hdrrb - 1.) else cr_b

    if cr_f.asgn == cr_b.asgn:
        cr = cr_f
    else:
        # Check if prefix/suffix is identical
        eqs = [f == b for f, b in zip(cr_f.asgn, cr_b.asgn)]
        if is_eq_prefix(eqs):
            cr = cr_f
        elif is_eq_suffix(eqs):
            cr = cr_b
        else:
            cr = cr_f if abs(hdrrf - 1.) <= abs(hdrrb - 1.) else cr_b   # TODO: better selection method?

    asgn = cr.asgn
    if verbose:
        print(f"  REL: ", end='')
    for I, s in zip(pread.rel_intvls, asgn):
        I.asgn = s
        if verbose:
            print(f"{color_asgn(I.asgn)}", end='')
    print()
    return cr_f, cr_b


def classify_rel_fw(pread, cp, verbose):
    cr_f = ClassRel(pread, cp, direction="forward", verbose=verbose, verbose_prob=False)
    asgn = cr_f.asgn
    for I, s in zip(pread.rel_intvls, asgn):
        I.asgn = s
    if all([I.asgn != 'H' for I in pread.rel_intvls]):
        # Only D. Check if it is actually H
        d_intvls = list(filter(lambda I: I.asgn == 'D', pread.rel_intvls))
        dmean = calc_mean_cov(d_intvls)
        # if abs(dmean - cp.depths['H']) <= abs(dmean - cp.depths['D']):
        if dmean < cp.depths['D']:
            _cp = deepcopy(cp)
            _cp.depths['H'] = d_intvls[0].ccb
            _cp.depths['D'] = _cp.depths['H'] + cp.depths['H']
            logger.info(f"Read {pread.id}: recompute forward DP ({_cp.depths})")
            cr_f = ClassRel(pread, _cp, direction="forward", verbose=verbose, verbose_prob=False)
            asgn = cr_f.asgn
            for I, s in zip(pread.rel_intvls, asgn):
                I.asgn = s
            # Only D again. If it is closer to global H-cov, set to H
            if all([I.asgn != 'H' for I in pread.rel_intvls]):
                d_intvls = list(filter(lambda I: I.asgn == 'D', pread.rel_intvls))
                dmean = calc_mean_cov(d_intvls)
                if abs(dmean - cp.depths['H']) <= abs(dmean - cp.depths['D']):
                # if dmean < cp.depths['D']:
                    logger.info(f"Read {pread.id}: force H")
                    cr_f.asgn = cr_f.asgn.replace('D', 'H')
    # TODO: only H?
    d_intvls = list(filter(lambda I: I.asgn == 'D', pread.rel_intvls))
    h_intvls = list(filter(lambda I: I.asgn == 'H', pread.rel_intvls))
    d_diff = abs(d_intvls[0].ccb - d_intvls[-1].cce) if len(d_intvls) > 0 else 0
    h_diff = abs(h_intvls[0].ccb - h_intvls[-1].cce) if len(h_intvls) > 0 else 0
    hdrr = (d_intvls[0].ccb / h_intvls[0].ccb) / (d_intvls[-1].cce / h_intvls[0].cce) if len(d_intvls) > 0 and len(h_intvls) > 0 else 1
    return (cr_f, d_diff, h_diff, hdrr)


def classify_rel_bw(pread, cp, verbose):
    cr_b = ClassRel(pread, cp, direction="backward", verbose=verbose, verbose_prob=False)
    asgn = cr_b.asgn
    for I, s in zip(pread.rel_intvls, asgn):
        I.asgn = s
    if all([I.asgn != 'H' for I in pread.rel_intvls]):
        # Only D. Check if it is actually H
        d_intvls = list(filter(lambda I: I.asgn == 'D', pread.rel_intvls))
        dmean = calc_mean_cov(d_intvls)
        # if abs(dmean - cp.depths['H']) <= abs(dmean - cp.depths['D']):
        if dmean < cp.depths['D']:
            _cp = deepcopy(cp)
            _cp.depths['H'] = d_intvls[-1].cce
            _cp.depths['D'] = _cp.depths['H'] + cp.depths['H']
            logger.info(f"Read {pread.id}: recompute backward DP ({_cp.depths})")
            cr_b = ClassRel(pread, _cp, direction="backward", verbose=verbose, verbose_prob=False)
            asgn = cr_b.asgn
            for I, s in zip(pread.rel_intvls, asgn):
                I.asgn = s
            # Only D again. If it is closer to global H-cov, set to H
            if all([I.asgn != 'H' for I in pread.rel_intvls]):
                d_intvls = list(filter(lambda I: I.asgn == 'D', pread.rel_intvls))
                dmean = calc_mean_cov(d_intvls)
                if abs(dmean - cp.depths['H']) <= abs(dmean - cp.depths['D']):
                # if dmean < cp.depths['D']:
                    logger.info(f"Read {pread.id}: force H")
                    cr_b.asgn = cr_b.asgn.replace('D', 'H')
    d_intvls = list(filter(lambda I: I.asgn == 'D', pread.rel_intvls))
    h_intvls = list(filter(lambda I: I.asgn == 'H', pread.rel_intvls))
    d_diff = abs(d_intvls[0].ccb - d_intvls[-1].cce) if len(d_intvls) > 0 else 0
    h_diff = abs(h_intvls[0].ccb - h_intvls[-1].cce) if len(h_intvls) > 0 else 0
    hdrr = (d_intvls[0].ccb / h_intvls[0].ccb) / (d_intvls[-1].cce / h_intvls[0].cce) if len(d_intvls) > 0 and len(h_intvls) > 0 else 1
    return (cr_b, d_diff, h_diff, hdrr)


def _log(x): return -inf if x == 0. else log(x)


def calc_mean_cov(intvls: List[Intvl]) -> float:
    assert len(intvls) > 0
    return (sum([(I.ccb + I.cce) / 2 * (I.e - I.b) for I in intvls])
            / sum([I.e - I.b for I in intvls]))


class PosCnt(NamedTuple):
    """Positional k-mer count (or estimated coverage)."""
    pos: int
    cnt: int


CellT = Tuple[int, StateT]
AsgnT = Union[List[str], str]
CovsT = Dict[StateT, PosCnt]


@dataclass(repr=False)
class ClassRel:
    """Class for a (pseudo-)DP of classification of reliable intervals.

    positional arguments:
      @ pread
      @ cp

    optional arguments:
      @ direction   : Must be one of {"forward" (default), "backward"}.
      @ verbose
      @ verbose_prob

    instance variables:
      @ F     : = `direction == "forward"`
      @ intvls: List of reliable intervals.
      @ N     : = `len(intvls)`
      @ L     : = `pread.length`
      @ dp    : `dp[i, s]` = Max logp when `i`-th interval's class is `s`.
      @ st    : `st[i, s][t]` = Estimated `t` (in {H/D/R})-cov and its position at the DP cell.
      @ bt    : `bt[i, s]` = The best assignments up to `i` at the DP cell.
      @ rpos  : Absolutely REPEAT intervals.
      @ asgn  : Classifications for reliable intervals.
    """
    pread:        ProfiledRead
    cp:           ClassParams
    direction:    InitVar[str] = "forward"
    F:            bool = field(init=False)
    verbose:      bool = False
    verbose_prob: bool = False

    intvls: List[Intvl] = field(init=False)
    N:      int = field(init=False)
    L:      int = field(init=False)
    dp:     Dict[CellT, float] = field(init=False, default_factory=dict)
    st:     Dict[CellT, CovsT] = field(init=False, default_factory=dict)
    bt:     Dict[CellT, str] = field(init=False, default_factory=dict)
    dh_ratio: Dict[CellT, float] = field(init=False, default_factory=dict)
    rpos:   Set[int] = field(init=False, default_factory=set)
    asgn:   str = field(init=False)

    def __post_init__(self, direction: str) -> None:
        assert direction in ("forward", "backward")
        self.F = (direction == "forward")
        self.intvls = deepcopy(self.pread.rel_intvls)
        self.N = len(self.intvls)
        self.L = self.pread.length

        self.run()

    def run(self) -> None:   # TODO: No rel intvl case
        self._init()
        for i in (range(1, self.N) if self.F else
                  reversed(range(self.N - 1))):
            self._update(i)
        self._backtrace()

    def _pred(self, i: int) -> int:
        return i - 1 if self.F else i + 1

    def _succ(self, i: int) -> int:
        return i + 1 if self.F else i - 1

    def _offset(self, i: int) -> int:
        return i - OFFSET if self.F else i + OFFSET

    def _expand_intvl(self, I: Intvl) -> Tuple[int, int, int, int]:
        """   L := (I.b, I.ccb) o--------o (I.e - 1, I.cce) =: R
        For [forward|backward], [L|R] is the beginning, [R|L] is the end.
        Use the beginning for transition from the predecessor, and
        use the end for update of H/D/R-cov.
        """
        L, R = PosCnt(I.b, I.ccb), PosCnt(I.e - 1, I.cce)
        return ((*L, *R) if self.F else (*R, *L))

    def _find_max_dp(self, i: int) -> Tuple[StateT, float]:
        """Find max state and logp in DP for `i`-th interval."""
        max_s, max_logp = None, -inf
        for s in STATES:
            if self.dp[i, s] == -inf:
                continue
            logp = self.dp[i, s]
            if max_logp < logp:
                max_s, max_logp = s, logp
        return max_s, max_logp

    def _find_max_dp_trans(self,
                           i: int,
                           logp_trans: Dict[Tuple[StateT, StateT], float],
                           s: Optional[StateT] = None,
                           t: Optional[StateT] = None) -> Tuple[StateT, float]:
        """For transition from `_pred(i)` (state `s`) to `i` (state `t`),
        find max state while fixing one of `s` and `t`."""
        assert (s is None) != (t is None), \
            "Only one of `s` and `t` must be specified"
        i_pred = self._pred(i)
        max_x, max_logp = None, -inf
        for x in STATES:
            _s = s if s is not None else x
            _t = t if t is not None else x
            logp = self.dp[i_pred, _s] + logp_trans[_s, _t]
            if max_logp < logp:
                max_x, max_logp = x, logp
        return max_x, max_logp

    def logp_e(self, i: int) -> float:
        """Basically use the error probability computed in the wall detection.
        In case of false-negative detection of E-intvls, compute another
        probability `logp_po` based only on the wall counts and E-cov.
        `E_PO_BASE` is for low-coverage non-E-intvls.
        """
        I = self.intvls[i]
        logp_er = _log(I.pe)   # TODO: store log pe
        logp_po = sum([poisson.logpmf(c, self.cp.depths['E'])
                       for c in (I.ccb, I.cce)]) + E_PO_BASE
        logp = max(logp_er, logp_po)
        if self.verbose_prob:
            print(f"ER={logp_er:5.0f}{'*' if logp_er >= logp_po else ' '} "
                  f"PO={logp_po:5.0f}{'*' if logp_po >= logp_er else ' '}")
        return logp

    def logp_r(self, i: int, st_pred: CovsT) -> float:
        """Given an imaginary R-cov (and its position) larger than D-cov,
        compute the probability of transition from it to `i`-th interval.
        If R-cov is larger, larger counts are required to have higher prob.
        If R-cov is smaller, smaller counts are classified as R.
        """
        I = self.intvls[i]
        beg_pos, beg_cnt, _, _ = self._expand_intvl(I)
        st = st_pred['R']

        logp_sf = -inf
        # logp_sf = calc_logp_trans(self._pred(st.pos), beg_pos,
        #                           st.cnt, beg_cnt,
        #                           st.cnt, self.cp.read_len)
        logp_er = (binom.logpmf(beg_cnt, st.cnt, 1 - 0.01)
                   if beg_cnt < st.cnt else -inf)
        logp = max(logp_sf, logp_er)
        if self.verbose_prob:
            print(f"SF={logp_sf:5.0f}{'*' if logp_sf >= logp_er else ' '} "
                  f"ER={logp_er:5.0f}{'*' if logp_er >= logp_sf else ' '}")

        # FIXME: revise the following codes
        if logp > R_LOGP:
            return logp
        if max(I.ccb, I.cce) >= self.cp.depths['R']:
            if self.verbose_prob:
                print(' ' * 6 + f"Counts >= Global R-cov")
            return R_LOGP
        if max(I.ccb, I.cce) >= st.cnt:
            if self.verbose_prob:
                print(' ' * 6 + f" Counts >= Est R-cov")
            return R_LOGP
        return logp

    def logp_h(self, i: int, st_pred: CovsT, s: StateT) -> float:
        I = self.intvls[i]
        beg_pos, beg_cnt, _, _ = self._expand_intvl(I)
        # Haplo transition
        st = st_pred['H']
        logp_sf_h = calc_logp_trans(self._pred(st.pos), beg_pos,
                                    st.cnt, beg_cnt,
                                    st.cnt, self.cp.read_len)
        # Diplo transition
        logp_sf_d = 0.
        key = (self._pred(i), s)
        if key in self.dh_ratio:
            dh_ratio = self.dh_ratio[key]
            st = st_pred['D']
            logp_sf_h = calc_logp_trans(self._pred(st.pos), beg_pos,
                                        st.cnt, int(dh_ratio * beg_cnt),
                                        st.cnt, self.cp.read_len)
        logp = logp_sf_h + logp_sf_d
        if self.verbose_prob:
            print(f"SF={logp_sf_h:5.0f} + {logp_sf_d:5.0f} -> {logp:5.0f}")
        return logp

    def logp_d(self, i: int, st_pred: CovsT, s: StateT) -> float:
        I = self.intvls[i]
        beg_pos, beg_cnt, _, _ = self._expand_intvl(I)
        # Haplo transition
        logp_sf_d = 0.
        key = (self._pred(i), s)
        if key in self.dh_ratio:
            dh_ratio = self.dh_ratio[key]
            st = st_pred['H']
            logp_sf_h = calc_logp_trans(self._pred(st.pos), beg_pos,
                                        st.cnt, int(beg_cnt / dh_ratio),
                                        st.cnt, self.cp.read_len)
        # Diplo transition
        st = st_pred['D']
        logp_sf_h = calc_logp_trans(self._pred(st.pos), beg_pos,
                                    st.cnt, beg_cnt,
                                    st.cnt, self.cp.read_len)
        logp = logp_sf_h + logp_sf_d
        if self.verbose_prob:
            print(f"SF={logp_sf_h:5.0f} + {logp_sf_d:5.0f} -> {logp:5.0f}")
        return logp


    def calc_logp(self,
                  s: StateT,
                  t: StateT,
                  i: int,
                  st_pred: CovsT) -> float:
        return (self.logp_e(i) if t == 'E'
                else self.logp_h(i, st_pred, s) if t == 'H'
                else self.logp_d(i, st_pred, s) if t == 'D'
                else self.logp_r(i, st_pred))

    def _init(self) -> None:
        # Prepare shared variables
        depths = self.cp.depths
        dp, st, bt = self.dp, self.st, self.bt

        # Set direction-specific variables
        i, pos_init = ((0, self._offset(0)) if self.F
                       else (self.N - 1, self._offset(self.L)))
        I = self.intvls[i]
        _, beg_cnt, end_pos, end_cnt = self._expand_intvl(I)

        # Main code
        if self.verbose:
            self._print_intvl(i)
        for s in STATES:
            st[i, s] = {t: PosCnt(pos_init, depths[t]) for t in 'HDR'}
            bt[i, s] = s
        dp[i, 'E'] = self.logp_e(i)
        dp[i, 'H'] = poisson.logpmf(beg_cnt, depths['H'])
        dp[i, 'D'] = poisson.logpmf(beg_cnt, depths['D'])
        dp[i, 'R'] = self.logp_r(i, st[i, 'R'])

        st[i, 'H'].update({'H': PosCnt(end_pos, end_cnt),
                           'D': PosCnt(self._offset(end_pos), end_cnt + depths['H'])})
        st[i, 'D'].update({'H': PosCnt(self._offset(end_pos), max(end_cnt // 2, end_cnt - depths['H'])),
                           'D': PosCnt(end_pos, end_cnt)})
        st[i, 'R'].update({'R': PosCnt(end_pos, min(depths['R'], end_cnt))})

        # TODO: set dh_ratio = 2?

        # Normalize
        for s in STATES:
            dp[i, s] = exp(dp[i, s])
        p_sum = sum([dp[i, s] for s in STATES])
        assert p_sum > 0., f"No possible state @ {i}"
        for s in STATES:
            dp[i, s] = _log(dp[i, s] / p_sum)

        if self.verbose:
            max_s, _ = self._find_max_dp(i)
            for s in STATES:
                print(f"{color_asgn(s)}:", end='')
                if dp[i, s] == -inf:
                    print()
                    continue
                print(f"({dp[i, s]:5.0f}; {self._st_to_str(st[i, s])})"
                      f"{'*' if s == max_s else ' '} {color_asgn(bt[i, s])}")
            print()

    def _update(self, i: int) -> None:
        if self.verbose:
            self._print_intvl(i)

        # Prepare shared variables
        depths = self.cp.depths
        intvls = self.intvls
        dp, st, bt = self.dp, self.st, self.bt

        # Set direction-specific variables
        I = intvls[i]
        beg_pos, beg_cnt, end_pos, end_cnt = self._expand_intvl(I)
        i_pred = self._pred(i)

        # Compute transition probabilities that are normalized
        # so that \sum_{s', t'} p(s' -> t') [p(t' <- s')] = 1
        logp_trans = {}
        for s in STATES:
            if dp[i_pred, s] == -inf:   # invalid pred state
                for t in STATES:
                    logp_trans[s, t] = 0.
                continue
            st_pred = st[i_pred, s]
            for t in STATES:
                if self.verbose_prob:
                    print(f"({s}; {self._st_to_str(st_pred)} ->) {t}: " if self.F else
                          f"{t} (<- {s}; {self._st_to_str(st_pred)}): ", end='')
#                 # TODO: Check H < D < R
#                 is_violated = False
#                 if t == 'R':
#                     if max(st_pred['H'].cnt, st_pred['D'].cnt) >= beg_cnt:
#                         is_violated = True
#                 elif t == 'D':
#                     if st_pred['H'].cnt >= beg_cnt or st_pred['R'].cnt <= beg_cnt:
#                         is_violated = True
#                 elif t == 'H':
#                     if st_pred['D'].cnt <= beg_cnt:
#                         is_violated = True
#                 logp = -inf if is_violated else self.calc_logp(t, i, st_pred)
                logp = self.calc_logp(s, t, i, st_pred)
                logp_trans[s, t] = exp(logp)
        p_sum = sum([logp_trans[s, t] for s in STATES for t in STATES])
        assert p_sum > 0., f"No possible state @ {i}"
        for s in STATES:
            for t in STATES:
                logp_trans[s, t] = _log(logp_trans[s, t] / p_sum)

        # Check if every path is converged to R
        only_r = True
        for s in STATES:
            max_t, _ = self._find_max_dp_trans(i, logp_trans, s=s)
            if max_t is not None and max_t != 'R':
                only_r = False
                break
        if only_r:
            if self.verbose:
                print(f"Absolutely REPEAT @ {i}\n")
            self.rpos.add(i)
            intvls[i] = intvls[i_pred]
            for s in STATES:
                dp[i, s] = dp[i_pred, s]
                if dp[i, s] == -inf:
                    continue
                bt[i, s] = (bt[i_pred, s] + s if self.F else   # TODO: append 'R'?
                            s + bt[i_pred, s])
                st[i, s] = st[i_pred, s]
            return

        # Let Pr{best pred = H -> H} == Pr{best pred = D -> D} because larger counts allow higher fluctuation,
        # resulting in shift to D in the middle of H curve
        max_s_h, _ = self._find_max_dp_trans(i, logp_trans, t='H')
        max_s_d, _ = self._find_max_dp_trans(i, logp_trans, t='D')
        if max_s_h == 'H' and max_s_d == 'D':
            logp_trans['H', 'H'] = logp_trans['D', 'D'] = min(logp_trans['H', 'H'], logp_trans['D', 'D'])

        # Find best path for each state and update estimated coverages
        for t in STATES:
            max_s, max_logp = self._find_max_dp_trans(i, logp_trans, t=t)
            dp[i, t] = max_logp
            if max_s is not None:
                bt[i, t] = (bt[i_pred, max_s] + t if self.F else
                            t + bt[i_pred, max_s])
                st_pred = st[i_pred, max_s]

                if t == 'E':
                    st[i, t] = st_pred
                elif t == 'R':
                    r_cnt = min(depths['R'], end_cnt)
                    st[i, t] = {'H': PosCnt(self._offset(end_pos), st_pred['H'].cnt),
                                'D': PosCnt(self._offset(end_pos), st_pred['D'].cnt),
                                'R': (st_pred['R'] if st_pred['R'].cnt < r_cnt
                                      else PosCnt(self._offset(end_pos), r_cnt))}
                elif t == 'H':
                    curr_h = end_cnt
                    dh_ratio = self.calc_dh_ratio('H',
                                                  bt[i, t],
                                                  intvls[:i+1] if self.F else intvls[i:])   # TODO: intvls から s は分かるのでH,D共通にする？
                    if dh_ratio is None:
                        if 'D' in bt[i, t]:
                            curr_d = st_pred['D'].cnt
                        else:
                            curr_d = curr_h + depths['H']
                    else:
                        curr_d = int(curr_h * dh_ratio)
                        self.dh_ratio[i, t] = dh_ratio
                    curr_r = int(curr_d * self.cp.DR_RATIO)
                    st[i, t] = {'H': PosCnt(self._offset(end_pos), curr_h),
                                'D': PosCnt(self._offset(end_pos), curr_d),
                                'R': PosCnt(self._offset(end_pos), curr_r)}
                else:
                    curr_d = end_cnt
                    dh_ratio = self.calc_dh_ratio('D',
                                                  bt[i, t],
                                                  intvls[:i+1] if self.F else intvls[i:])
                    if dh_ratio is None:
                        if 'H' in bt[i, t]:
                            curr_h = st_pred['H'].cnt
                        else:
                            curr_h = max(int(curr_d / 2), curr_d - depths['H'])
                    else:
                        curr_h = int(curr_d / dh_ratio)
                        self.dh_ratio[i, t] = dh_ratio
                    curr_r = int(curr_d * self.cp.DR_RATIO)
                    st[i, t] = {'H': PosCnt(self._offset(end_pos), curr_h),
                                'D': PosCnt(self._offset(end_pos), curr_d),
                                'R': PosCnt(self._offset(end_pos), curr_r)}
                # H < D < R requirement
                if not (st[i, t]['H'].cnt < st[i, t]['D'].cnt
                        and st[i, t]['D'].cnt < st[i, t]['R'].cnt):
                    dp[i, t] = -inf

        if self.verbose:
            print("t\s", end='')
            for s in STATES:
                if dp[i_pred, s] == -inf:
                    print(' ' * 6 + color_asgn(s), end='')
                else:
                    print(f" {dp[i_pred, s]:5.0f}{color_asgn(s)}", end='')
            print()
            max_t, _ = self._find_max_dp(i)
            for t in STATES:
                print(f"{color_asgn(t)}:", end='')
                max_s, _ = self._find_max_dp_trans(i, logp_trans, t=t)
                for s in STATES:
                    print(
                        f" {logp_trans[s, t]:5.0f}{'*' if s == max_s else ' '}", end='')
                if (i, t) in st:
                    if self.F:
                        _asgn = self._set_r_asgn(bt[i, t], self.rpos)
                    else:
                        _asgn = ''.join(list(reversed(self._set_r_asgn(list(reversed(bt[i, t])),
                                                                       {self.N - 1 - rp for rp in self.rpos}))))
                    _arrow = "->"# if self.F else "<-"
                    print(f" {color_asgn(max_s)}{_arrow}{color_asgn(t)}"
                          f"({dp[i, t]:5.0f}; {self._st_to_str(st[i_pred, max_s])}{_arrow}{self._st_to_str(st[i, t])})"
                          f"{'*' if t == max_t else ' '} {color_asgn(_asgn)}", end='')
                print()
            print()

    def _backtrace(self) -> None:
        last_pos = self.N - 1 if self.F else 0
        max_s, _ = self._find_max_dp(last_pos)
        self.asgn = list(self.bt[last_pos, max_s])
        # Set absolute R
        for rp in self.rpos:
            self.asgn[rp] = 'R'
        self.asgn = ''.join(self.asgn)

    def _find_nn(self,
                 forward: bool,
                 i: int,
                 s: StateT,
                 asgn: AsgnT,
                 intvls: List[Intvl]) -> int:
        """Find index of interval with state `s` nearest from `intvls[i]`
        """
        idx = i
        if forward:
            while idx < len(intvls) and asgn[idx] != s:
                idx += 1
        else:
            while idx >= 0 and asgn[idx] != s:
                idx -= 1
        return idx

    def calc_dh_ratio(self, init_s: str, asgn: AsgnT, intvls: List[Intvl]):
        def is_out(idx, _f):
            return ((_f and idx < 0)
                    or (not _f and idx >= len(asgn)))

        assert init_s in ('H', 'D')
        assert len(asgn) == len(intvls)
        ss = 'DHD' if init_s == "D" else 'HDH'
        idxs = [len(asgn) if self.F else -1]
        for s in ss:
            idxs.append(self._find_nn(not self.F,
                                      self._pred(idxs[-1]),
                                      s, asgn, intvls))
            if is_out(idxs[-1], self.F):
                return None
        idxs = idxs[1:]
        assert asgn[len(asgn) - 1 if self.F else 0] == init_s, \
            "The focal interval must be included"
        assert not is_out(idxs[0], not self.F)

        # TODO: use PosCnt?
        s1_pos, s1_cnt, _, _ = self._expand_intvl(intvls[idxs[0]])
        _, _, t_pos, t_cnt = self._expand_intvl(intvls[idxs[1]])
        _, _, s2_pos, s2_cnt = self._expand_intvl(intvls[idxs[2]])
        if not self.F:
            s1_pos, s1_cnt, s2_pos, s2_cnt = s2_pos, s2_cnt, s1_pos, s1_cnt

        pos_ratio = (t_pos - s2_pos) / (s1_pos - s2_pos)
        est_s_cnt = s2_cnt + (s1_cnt - s2_cnt) * pos_ratio
        dh_ratio = est_s_cnt / t_cnt if init_s == 'D' else t_cnt / est_s_cnt
        if self.verbose_prob:
            print(f"D/H ratio={dh_ratio:5.2f} (init_s={init_s}, asgn={asgn}, idxs={idxs})")

        return dh_ratio

    def _set_r_asgn(self, asgn: AsgnT, rpos: Set[int]) -> AsgnT:
        if isinstance(asgn, str):
            asgn = list(asgn)
        for rp in rpos:
            asgn[rp] = 'R'
        return ''.join(asgn) if isinstance(asgn, str) else asgn

    def _print_intvl(self, i: int) -> None:
        I = self.intvls[i]
        print(f"# RI[{i}] = ({I.b}, {I.e}): {I.ccb} ~~ {I.cce}")

    def _st_to_str(self, st: CovsT) -> str:
        return f"{st['H'].cnt:2}@{st['H'].pos},{st['D'].cnt:2}@{st['D'].pos},{st['R'].cnt:3}@{st['R'].pos}"

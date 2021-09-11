from dataclasses import dataclass
from typing import Optional, NamedTuple, Tuple
from math import inf, log
from scipy.stats import poisson, binom
from .. import Etype, STATES, StateT, Intvl, ProfiledRead, E_PO_BASE, R_LOGP
from .._plot import color_asgn
from .._main import ClassParams
from ._util import minus_sigma, linear_interplation, calc_logp_trans, calc_p_error


def classify_unrel(pread, cp, verbose, verbose_prob=False):
    _ = ClassUnrel(pread, cp, verbose=verbose, verbose_prob=verbose_prob)


def _log(x): return -inf if x == 0. else log(x)
def is_updated(I): return not (I.is_rel and I.asgn in ('H', 'D'))


class PosCnt(NamedTuple):
    """Positional k-mer count (or estimated coverage)."""
    pos: int
    cnt: int


@dataclass(repr=False)
class ClassUnrel:
    pread:        ProfiledRead
    cp:           ClassParams
    n_sigma:      int = 1           # TODO: stop using n_sigma
    verbose:      bool = False
    verbose_prob: bool = False

    def __post_init__(self):
        self.run()

    def run(self):
        pread = self.pread

        # Find best class for each interval (except reliable H/D-intervals)
        # from larger count to smaller, and then smaller to larger.
        _ascend = [x[0] for x in
                   sorted([(idx, min(I.cb, I.ce))
                           for idx, I in enumerate(pread.intvls)
                           if is_updated(I)],
                          key=lambda x: x[1])]
        _descend = reversed(_ascend)
        for idx in _descend:              # TODO: more robust order?
            self._update(idx)
        for idx in _ascend:
            self._update(idx)

        # Convert interval classifications into k-mer classifications
        assert all([I.asgn in STATES for I in pread.intvls]), \
            "Unclassified intervals"
        pread.states = ['-'] * pread.length
        for I in pread.intvls:
            for i in range(I.b, I.e):
                assert pread.states[i] == '-', "Conflict"
                pread.states[i] = I.asgn
        assert all([s in STATES for s in pread.states]), \
            "Unclassified k-mers"

        # Extend short E-intvls and resolve slippage of low-complexity bases
#         self.extend_e()
#         self.resolve_slip()

        pread.states = ''.join(pread.states)

    def _update(self, i: int) -> None:
        if self.verbose:
            self._print_intvl(i)

        I = self.pread.intvls[i]
        max_s, max_logp = None, -inf
        for s in 'REHD':
            logp = self.calc_logp(s, i)
            if max_logp < logp:
                max_s, max_logp = s, logp
            if max_logp == 0.:   # >= R-cov
                break
        if self.verbose and I.asgn != max_s:
            print(f"Updated: {color_asgn(I.asgn)} -> {color_asgn(max_s)}")
        I.asgn = max_s

    def _find_nn(self, i: int, s: StateT, only_rel: bool) -> Tuple[Optional[int], Optional[int]]:
        """For each direction, find index of interval with state `s`
        nearest from `intvls[i]`. If not found, return None.
        """
        def _is_target(I: Intvl):
            return I.asgn == s and (I.is_rel if only_rel else True)

        intvls = self.pread.intvls
        l = i - 1
        while l >= 0 and not _is_target(intvls[l]):
            l -= 1
        if l < 0:
            l = None
        r = i + 1
        while r < len(intvls) and not _is_target(intvls[r]):
            r += 1
        if r >= len(intvls):
            r = None
        if self.verbose_prob:
            print(f"(NN {s}=({l},{r})) ", end='')
        return (l, r)

    def _est_cov(self,
                 i: int,
                 x: int,
                 s: StateT,
                 from_est: bool = False) -> int:
        """Estimate the true count of the class `s` at position `x` of the `i`-th interval
        using reliable intervals.
        NOTE: `x` == `intvls[i].b` or `intvls[i].e - 1`

        [Case 1] Use a simple linear interpolation given reliable intervals
                 of the same class at both sides:
                          L.e - 1
                   L o-----o
                                  I o-----o     R.b
                 (I.b or I.e - 1 =) x            o-----o R

        [Case 2] Use a "zig-zag" interploation given a H-D coverage ratio estimated from
                 reliable intervals of both Haplo and Diplo:
                                                                       x
                   D2 o-----o        D1  o-----o                       o-----o I(Diplo)
                                                      ...
                            H o-----o                        o-----o L(Haplo)
                 (estimate H-D ratio from D2, H, D1)
        """
        assert s in ('H', 'D')
        intvls = self.pread.intvls
        l, r = self._find_nn(i, s, only_rel=True)
        if l is not None and r is not None:
            L, R = intvls[l], intvls[r]
            return linear_interplation(x, (L.e - 1, L.cce), (R.b, R.ccb))
        elif l is not None:
            L = intvls[l]
            return L.cce   # TODO: better estimation using H intervals and H-D ratio
        elif r is not None:
            R = intvls[r]
            return R.ccb
        else:
            # No D-reliable intervals. Use H-reliable intervals if exist
            if from_est:   # Came from H-estimation, meaning no H and D reliable intervals
                return None
            cov = self._est_cov(i, x, 'D' if s == 'H' else 'H', from_est=True)
            if cov is not None:
                return cov // 2 if s == 'H' else cov * 2
            else:   # No H and D reliable intervals
                return self.cp.depths[s]

    def calc_logp(self, s: StateT, i: int) -> float:
        assert s in STATES, f"Invalid state: {s}"
        return (self.logp_e if s == 'E'
                else self.logp_h if s == 'H'
                else self.logp_d if s == 'D'
                else self.logp_r)(i)

    def logp_e(self, i: int) -> float:
        if self.verbose_prob:
            print(f"[{color_asgn('E')}] ", end='')
        I = self.pread.intvls[i]
        logp_er = _log(I.pe)
        logp_po = sum([poisson.logpmf(c, self.cp.depths['E'])
                       for c in (I.cb, I.ce)]) + E_PO_BASE
        logp = max(logp_er, logp_po)
        if self.verbose_prob:
            print(f"ER={logp_er:5.0f}{'*' if logp_er >= logp_po else ' '} "
                  f"PO={logp_po:5.0f}{'*' if logp_po >= logp_er else ' '}")
        return logp

    def logp_r(self, i: int) -> float:
        if self.verbose_prob:
            print(f"[{color_asgn('R')}] ", end='')
        intvls = self.pread.intvls
        I = intvls[i]
        if max(I.cb, I.ce) >= self.cp.depths['R']:
            if self.verbose_prob:
                print(f"> Global R-cov")
            return 0.
        # est_cnt = self._est_cov(i, I.b, s)
        l, r = self._find_nn(i, 'D', only_rel=True)   # FIXME: reconsider coverage estimation
        if l is None and r is None:
            dcov_l = dcov_r = self.cp.depths['D']
        elif l is None:
            dcov_l = dcov_r = intvls[r].cb
        elif r is None:
            dcov_l = dcov_r = intvls[l].ce
        else:
            dcov_l, dcov_r = intvls[l].ce, intvls[r].cb
        rcov_l, rcov_r = int(dcov_l * self.cp.DR_RATIO), int(dcov_r * self.cp.DR_RATIO)
        if I.cb >= rcov_l or I.ce >= rcov_r:
            if self.verbose_prob:
                print(f"> Est R-cov (B: {I.cb} >= {rcov_l} or E: {I.ce} >= {rcov_r})")   # FIXME: "slipping interval" in repeats
            return R_LOGP
        logp_l = binom.logpmf(I.cb, rcov_l, 1 - 0.01)   # TODO: use smaller n-sigma and use calc_logp
        logp_r = binom.logpmf(I.ce, rcov_r, 1 - 0.01)
        logp = logp_l + logp_r
        if self.verbose_prob:
            print(f"ER={logp_l:5.0f} + {logp_r:5.0f} -> logp={logp:5.0f}")
        return logp

    def logp_hd(self, s: StateT, i: int) -> float:
        cp = self.cp
        intvls = self.pread.intvls
        I = intvls[i]
        l_rel, r_rel = self._find_nn(i, s, only_rel=True)

        # TODO: Fast return when it is almost impossible transition
        # TODO: memoization of probabilities (given dependencies)

        ### --------------------------- LEFT TRANSITION ---------------------------------- ###
        logp_l_er = logp_l_sf = logp_l_sf_er = -inf
        # Case 1. Errors in others from adjacent interval with the same class
        l = i - 1
        if l >= 0 and intvls[l].asgn == s:
            logp_l_er = _log(I.pe_o.b)
        # Case 2. Sampling fluctuation from nearest neighbor reliable interval
        if l_rel is not None:
            L = intvls[l_rel]
            logp_l_sf = calc_logp_trans(L.e - 1, I.b, L.cce, I.cb,
                                        L.cce, cp.read_len)
        # Case 3. Estimated true count at I.b and errors in others
        est_cnt = self._est_cov(i, I.b, s)
        # TODO: calculate max error probability using contexts within [I.b - K + 1..I.b]
        max_erate = 0.1   # TODO: use contexts
        if est_cnt >= I.cb:
            logp_l_sf_er = _log(calc_p_error(est_cnt, I.cb, max_erate, Etype.OTHERS))
        logp_l = max(logp_l_er, logp_l_sf, logp_l_sf_er)
        if self.verbose_prob:
            print(f"[L] SF={logp_l_sf:5.0f} "
                  f"ER={logp_l_er:5.0f} "
                  f"SF-ER={logp_l_sf_er:5.0f} (logp_l={logp_l:5.0f})", end='')

        ### --------------------------- RIGHT TRANSITION ---------------------------------- ###
        logp_r_er = logp_r_sf = logp_r_sf_er = -inf
        # Case 1. Errors in others from adjacent interval with the same class
        r = i + 1
        if r < len(intvls) and intvls[r].asgn == s:
            logp_r_er = _log(I.pe_o.e)
        # Case 2. Sampling fluctuation from nearest neighbor reliable interval
        if r_rel is not None:
            R = intvls[r_rel]
            logp_r_sf = calc_logp_trans(I.e - 1, R.b, I.ce, R.ccb,
                                        R.ccb, cp.read_len)
        # Case 3. Estimated true count at I.b and errors in others
        est_cnt = self._est_cov(i, I.e - 1, s)
        # TODO: calculate max error probability using contexts within [I.b - K + 1..I.b]
        max_erate = 0.1
        if est_cnt >= I.ce:
            logp_r_sf_er = _log(calc_p_error(est_cnt, I.ce, max_erate, Etype.OTHERS))
        logp_r = max(logp_r_er, logp_r_sf, logp_r_sf_er)
        if self.verbose_prob:
            print(f"[R] SF={logp_r_sf:5.0f} "
                  f"ER={logp_r_er:5.0f} "
                  f"SF-ER={logp_r_sf_er:5.0f} (logp_r={logp_r:5.0f})", end='')

        if logp_l == -inf and logp_r == -inf:
            logp_l = poisson.logpmf(I.cb, self.cp.depths[s])
            logp_r = poisson.logpmf(I.ce, self.cp.depths[s])
            if self.verbose_prob:
                print(f"No other {s}-intvl. PO={logp_l:5.0f} + {logp_r:5.0f} ", end='')
        elif logp_l == -inf:
            logp_l = logp_r
        elif logp_r == -inf:
            logp_r = logp_l
        logp = logp_l + logp_r
        if self.verbose_prob:
            print(f"-> logp={logp:5.0f}")
        return logp

    def logp_h(self, i: int) -> float:
        if self.verbose_prob:
            print(f"[{color_asgn('H')}] ", end='')
        # intvls = self.pread.intvls
        # I = intvls[i]
        # l, r = self._find_nn(i, 'D', only_rel=True)
        # if l is not None:
        #     dcov_l = intvls[l].ce
        #     if minus_sigma(dcov_l, self.n_sigma) <= I.cb:
        #         if self.verbose_prob:
        #             print(f"[B] > D - {self.n_sigma} sigma")
        #         return -inf
        # if r is not None:
        #     dcov_r = intvls[r].cb
        #     if minus_sigma(dcov_r, self.n_sigma) <= I.ce:
        #         if self.verbose_prob:
        #             print(f"[E] > D - {self.n_sigma} sigma")
        #         return -inf
        return self.logp_hd('H', i)

    def logp_d(self, i: int) -> float:
        if self.verbose_prob:
            print(f"[{color_asgn('D')}] ", end='')
        return self.logp_hd('D', i)

    def _print_intvl(self, i: int) -> None:
        I = self.pread.intvls[i]
        print(f"# I[{i}] = ({I.b}, {I.e}): {I.cb} ~~ {I.ce} (asgn={color_asgn(I.asgn)})")

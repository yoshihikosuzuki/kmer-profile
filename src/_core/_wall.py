from typing import Dict
from math import inf
from logzero import logger
from .. import (Etype, Ctype, Wtype, ThresT, PerrorInO, ErrorIntvl, Intvl, ProfiledRead,
                PE_THRES, PTHRES_DIFF_EO, PTHRES_DIFF_REL, MAX_N_HC)
from .._main import ClassParams
from ._util import calc_p_error, calc_p_trans


def calc_cthres(cp: ClassParams,
                verbose: bool = False) -> Dict:
    cthres = {ThresT.INIT: {}, ThresT.FINAL: {}}
    for ctype in Ctype:
        emodel = cp.emodels[ctype.value]
        for l in range(1, emodel.max_ulen + 1):
            pe = emodel.pe(l)
            for cout in range(1, cp.depths['R']):
                for thres_t in ThresT:
                    for etype, cdef in zip(Etype, (cout, 0)):
                        cthres[thres_t][(ctype, l, cout, etype)] = cdef
                is_found = [[False, False], [False, False]]
                for cin in range(cout + 1):
                    if all([xx for x in is_found for xx in x]):
                        break
                    pvalue = calc_p_error(cout, cin, pe, Etype.SELF)
                    ct = {Etype.SELF: cin - 1, Etype.OTHERS: cout - cin + 1}
                    for thres_t in ThresT:
                        for etype in Etype:
                            if (not is_found[thres_t.value][etype.value]
                                    and pvalue < PE_THRES[thres_t][etype]):
                                cthres[thres_t][(ctype, l, cout, etype)] = ct[etype]
                                is_found[thres_t.value][etype.value] = True
            if verbose:
                print(f"{ctype}, l={l}")
                for etype in Etype:
                    for thres_t in ThresT:
                        print(f"{etype}, {thres_t}")
                        for cout in range(1, cp.depths['R']):
                            print(f"{cthres[thres_t][(ctype, l, cout, etype)]} ", end='')
                        print()
                print()
    return cthres


def calc_p_diff_pair(i, j, pread, cp):
    n_drop = pread.counts[i - 1] - pread.counts[i]
    n_gain = pread.counts[j] - pread.counts[j - 1]
    cov = max(pread.counts[i - 1], pread.counts[j])
    return calc_p_trans(i, j, n_drop, n_gain, cov, cp.read_len)


def cthres_ng(cin, cthres, key, etype):
    if etype == Etype.SELF:
        return cin >= cthres[key]
    else:
        return cin < cthres[key]


def find_gain(pread, cp, perrors, i, etype, cout, cin, ctype, clen, cerate, verbose=False):
    cthres = cp.cthres[ThresT.FINAL]
    HC_ERATE = cp.emodels[Ctype.HP.value].pe(1)

    max_j, max_pe = None, -inf
    ipk = i + pread.K - 1

    # Low-complexity error
    ulen = ctype.value + 1
    m = clen * ulen
    useq = pread.seq[i - ulen:i]
    n = 0
    while pread.seq[i + n:i + n + ulen] == useq:
        n += ulen
    j = ipk + n - m
    if j <= i:   # Too long (>= K bp) low-complexity sequences
        return None
    if j >= pread.length:
        j = pread.length
        pe = perrors[i, etype, Wtype.DROP] ** 2   # TODO: なぜここで boundary E-intvl が弾けていないのか
        cout_j, cin_j = "*", "*"
    else:
        cin_j, cout_j = pread.counts[j - 1:j + 1]
        pe = -inf
        if cin_j <= cout_j:
            ct_key_j = (ctype, clen, cout_j, etype)
            if not (ct_key_j in cthres and cthres_ng(cin_j, cthres, ct_key_j, etype)):
                if (etype == Etype.SELF
                        or calc_p_diff_pair(i, j, pread, cp) >= PTHRES_DIFF_EO):
                    update_perror(perrors, j, etype, Wtype.GAIN,
                                cout_j, cin_j, cerate)
                    pe = (perrors[i, etype, Wtype.DROP] *
                        perrors[j, etype, Wtype.GAIN])
    if max_pe < pe:
        max_j, max_pe = j, pe
    if verbose:
        print(f"  *LC({i},{j}:{cin_j}->{cout_j};{pe}) ", end='')

    # High-complexity error
    m = 0
    for n in range(MAX_N_HC + 1):
        j = ipk + n - m
        if j >= pread.length:
            continue
        data = pread.counts[j - 1:j + 1]
        if len(data) != 2:
            continue
        cin_j, cout_j = data

        if not (cin_j <= cout_j):
            continue
        # Impossible cases
        ct_key_i = (Ctype.HP, 1, cout, etype)
        ct_key_j = (Ctype.HP, 1, cout_j, etype)
        if ((ct_key_i in cthres and cthres_ng(cin, cthres, ct_key_i, etype))
                or (ct_key_j in cthres and cthres_ng(cin_j, cthres, ct_key_j, etype))):
            continue

        if (etype == Etype.SELF
                or calc_p_diff_pair(i, j, pread, cp) < PTHRES_DIFF_EO):   # FIXME: SELF always skipped!!
            continue

        pe_i = calc_p_error(cout, cin, HC_ERATE, etype)
        pe_j = calc_p_error(cout_j, cin_j, HC_ERATE, etype)
        pe = pe_i * pe_j
        if max_pe < pe:
            max_j, max_pe = j, pe
        if verbose:
            print(f"  *HC({i},{j}:{cin_j}->{cout_j};{pe}) ", end='')

    return (i, max_j, max_pe)


def find_drop(pread, cp, perrors, i, etype, cout, cin, ctype, clen, cerate, verbose=False):
    cthres = cp.cthres[ThresT.FINAL]
    HC_ERATE = cp.emodels[Ctype.HP.value].pe(1)

    max_j, max_pe = None, -inf
    imk = i - pread.K + 1

    # Low-complexity error
    ulen = ctype.value + 1
    m = clen * ulen
    useq = pread.seq[imk:imk + ulen]
    n = 0
    while pread.seq[imk - n - ulen:imk - n] == useq:
        n += ulen
    j = imk - n + m
    if j >= i:   # Too long (>= K bp) low-complexity sequences
        return None
    if j <= 0:
        j = 0
        pe = perrors[i, etype, Wtype.GAIN] ** 2
        cout_j, cin_j = "*", "*"
    else:
        cout_j, cin_j = pread.counts[j - 1:j + 1]
        pe = -inf
        if cin_j <= cout_j:
            ct_key_j = (ctype, clen, cout_j, etype)
            if not (ct_key_j in cthres and cthres_ng(cin_j, cthres, ct_key_j, etype)):
                if (etype == Etype.SELF
                        or calc_p_diff_pair(j, i, pread, cp) >= PTHRES_DIFF_EO):
                    update_perror(perrors, j, etype, Wtype.DROP,
                                cout_j, cin_j, cerate)
                    pe = (perrors[j, etype, Wtype.DROP] *
                        perrors[i, etype, Wtype.GAIN])
    if max_pe < pe:
        max_j, max_pe = j, pe
    if verbose:
        print(f"  *LC({cout_j}->{cin_j}:{j},{i};{pe}) ", end='')

    # High-complexity error
    m = 0
    for n in range(MAX_N_HC + 1):
        j = imk - n + m
        if j <= 0:
            continue
        data = pread.counts[j - 1:j + 1]
        if len(data) != 2:
            continue
        cout_j, cin_j = data

        # Impossible cases
        if not (cin_j <= cout_j):
            continue
        ct_key_i = (Ctype.HP, 1, cout, etype)
        ct_key_j = (Ctype.HP, 1, cout_j, etype)
        if ((ct_key_i in cthres and cthres_ng(cin, cthres, ct_key_i, etype))
                or (ct_key_j in cthres and cthres_ng(cin_j, cthres, ct_key_j, etype))):
            continue

        if (etype == Etype.SELF
                or calc_p_diff_pair(j, i, pread, cp) < PTHRES_DIFF_EO):
            continue

        pe_i = calc_p_error(cout, cin, HC_ERATE, etype)
        pe_j = calc_p_error(cout_j, cin_j, HC_ERATE, etype)
        pe = pe_i * pe_j
        if max_pe < pe:
            max_j, max_pe = j, pe
        if verbose:
            print(f"  *HC({cout_j}->{cin_j}:{j},{i};{pe}) ", end='')

    return (max_j, i, max_pe)


def update_perror(perrors, i, etype, wtype, cout, cin, pe):
    key = (i, etype, wtype)
    if key not in perrors:
        perrors[key] = calc_p_error(cout, cin, pe, etype)
    return


def find_pair(pread, cp, perrors, i, etype, wtype, cout, cin, ctype, l, pe, verbose=False):
    data = (find_gain if wtype == Wtype.DROP
                    else find_drop)(pread, cp, perrors, i, etype, cout, cin, ctype, l, pe, verbose)
    if data is None:
        return None
    else:
        b, e, max_pe = data
        return ErrorIntvl(b=b, e=e, pe=max_pe)


def find_walls(pread: ProfiledRead,
               cp: ClassParams,
               min_cnt_cng: int = 3,
               max_cnt_cng: int = 5,
               verbose: bool = False):
    """List all (candidates of) walls and calculate Pr{E in self} for each interval.

    NOTE: It is fine to somewhat over-detect candidates of E-intervals.
          The most important thing here is to calculate Pr{E} for all candidates.
    """
    perrors = {}   # perrors[i, etype, wtype]
    walls = {etype: set() for etype in Etype}
    e_intvls, o_intvls = set(), set()
    paired_walls = {etype: set() for etype in Etype}   # Already-paired positions

    # Find E-intvls by single errors
    for i in range(1, pread.length):
        cim1, ci = pread.counts[i - 1:i + 1]
        if min(cim1, ci) >= cp.depths['R']:   # TODO: check the possibility of high count E-mers
            continue
        cng = abs(cim1 - ci)
        if cng < min_cnt_cng:
            continue

        # Find context type and length that gives max error prob
        wtype, cout, cin = ((Wtype.DROP, cim1, ci) if cim1 > ci
                            else (Wtype.GAIN, ci, cim1))
        max_ctx, max_l, max_pe = None, None, -inf
        for ctype in Ctype:
            l = pread.ctx[i][ctype.value][wtype.value]
            pe = cp.emodels[ctype.value].pe(l)
            if max_pe < pe:
                max_ctx, max_l, max_pe = ctype, l, pe
        if verbose:
            print(f"@ {i:5} (cout={cout:2}, cin={cin:2}, {wtype}, "
                  f"ctx=({max_ctx}, {max_l:2} units, pe={max_pe})")

        for etype in Etype:
            if verbose:
                print(f"  {etype:12}: ", end='')
            if i in paired_walls[etype]:
                if verbose:
                    print(f"[Already paired]")
                continue

            # Check precomputed thresholds if < R-cov
            ct_key = (max_ctx, max_l, cout, etype)
            if ct_key not in cp.cthres[ThresT.INIT]:
                ct_final = None
                if verbose:
                    print(f"[No cthres] ", end='')
            else:
                ct_init, ct_final = [cp.cthres[t][ct_key] for t in ThresT]
                if verbose:
                    print(
                        f"ct(init,final)=({ct_init:2},{ct_final:2}), ", end='')
                if not (cng > max_cnt_cng or cin < max(ct_init, 3)):
                    if verbose:
                        print(f"< ct_init [Not S/O wall]")
                    continue

            # Check if it is wall
            if etype == Etype.SELF:
                if ct_final is not None and cin >= ct_final:
                    # Cannot be E-intvl
                    if verbose:
                        print(f">= ct_final [Not S]")
                        continue
                else:
                    # Find the pair wall and if Perror(S) > pthres(O), add to E-intvls
                    update_perror(perrors, i, etype, wtype, cout, cin, max_pe)
                    if verbose:
                        print(f"pe={perrors[i, etype, wtype]} ", end='')
                    if perrors[i, etype, wtype] >= PE_THRES[ThresT.FINAL][etype]:
                        intvl = find_pair(pread, cp, perrors, i, etype, wtype,
                                          cout, cin, max_ctx, max_l, max_pe, verbose)
                        if intvl is None:
                            walls[etype].add(i)   # TODO: is it OK?
                        else:
                            if verbose:
                                print(
                                    f"intvl ({intvl.b},{intvl.e}) pe={intvl.pe} ", end='')
                            if intvl.pe >= PE_THRES[ThresT.FINAL][etype]:
                                walls[etype] |= set([intvl.b, intvl.e])
                                e_intvls.add(intvl)
                                paired_walls[etype] |= set([intvl.b, intvl.e])
                                if verbose:
                                    print(f"[S WALL & INTVL]")
                            else:
                                if verbose:
                                    print()
                    else:
                        if verbose:
                            print()
            else:   # Etype.OTHERS
                if cng >= cp.depths['H'] or (ct_final is not None and cin < ct_final):
                    # Cannot be E in O
                    walls[etype].add(i)
                    if verbose:
                        print(f"cng >= H-cov OR < ct_final [Wall]")
                    continue
                else:
                    # Find the pair wall and if Perror(O) > pthres(O), add to O-intvls
                    update_perror(perrors, i, etype, wtype, cout, cin, max_pe)
                    if verbose:
                        print(f"pe={perrors[i, etype, wtype]} ", end='')
                    if perrors[i, etype, wtype] < PE_THRES[ThresT.FINAL][etype]:
                        # Never paired
                        walls[etype].add(i)
                        if verbose:
                            print(f"[O WALL]")
                    else:
                        intvl = find_pair(pread, cp, perrors, i, etype, wtype,
                                          cout, cin, max_ctx, max_l, max_pe, verbose)
                        if intvl is None:
                            walls[etype].add(i)   # TODO: is it OK?
                        else:
                            if verbose:
                                print(
                                    f"intvl ({intvl.b},{intvl.e}) pe={intvl.pe} ", end='')
                            if intvl.pe < PE_THRES[ThresT.FINAL][etype]:
                                walls[etype].add(i)
                                if verbose:
                                    print(f"[O WALL]")
                            else:
                                o_intvls.add(intvl)
                                paired_walls[etype] |= set([intvl.b, intvl.e])
                                if verbose:
                                    print(f"[O INTVL]")
        if verbose:
            print()

    # if verbose:
    #     print(f"S-walls: {sorted(walls[Etype.SELF])}")
    #     print(f"O-walls: {sorted(walls[Etype.OTHERS])}")

    # From walls by non-O, exclude positions explained by O or within E-intvls
    for o_intvl in o_intvls:
        for i in (o_intvl.b, o_intvl.e):
            walls[Etype.OTHERS].discard(i)
    for e_intvl in e_intvls:
        for i in range(e_intvl.b + 1, e_intvl.e):
            walls[Etype.OTHERS].discard(i)

    # if verbose:
    #     print(f"S-intvls: {sorted([(I.b, I.e) for I in e_intvls])}")
    #     print(f"O-intvls: {sorted([(I.b, I.e) for I in o_intvls])}")

    # Find E-intvls by multiple errors and boundary E-intvls
    e_intvls_single_pos = set([(I.b, I.e) for I in e_intvls])
    e_intvls_multi = set()
    already_paired = set()
    pt = PE_THRES[ThresT.FINAL][Etype.SELF]
    for i in sorted(walls[Etype.OTHERS] - walls[Etype.SELF]):
        if i in already_paired:
            continue
        for wtype in Wtype:
            key_i = (i, Etype.SELF, wtype)
            if key_i in perrors:
                pe_i = perrors[key_i]
                if pe_i < pt:
                    continue
                search_range = (range(i + 1, min(i + 200, pread.length + 1))
                                if wtype == Wtype.DROP
                                else reversed(range(max(i - 200, 0), i)))
                for j in search_range:
                    if j == 0 or j == pread.length:   # Boundary intervals
                        pe = pe_i * pe_i
                        if pe >= pt:
                            b, e = (0, i) if j == 0 else (i, pread.length)
                            e_intvls_multi.add(ErrorIntvl(b=b, e=e, pe=pe))
                            already_paired.add(i)
                            if verbose:
                                print(f"Boundary ({b},{e}): pe={pe}")
                    if j not in (walls[Etype.OTHERS] | walls[Etype.SELF]):
                        continue
                    if wtype == Wtype.DROP:
                        b, e, key_j = (i, j, (j, Etype.SELF, Wtype.GAIN))
                    else:
                        b, e, key_j = (j, i, (j, Etype.SELF, Wtype.DROP))
                    if (b, e) not in e_intvls_single_pos:
                        if key_j in perrors:
                            pe_j = perrors[key_j]
                            pe = pe_i * pe_j
                            if pe < pt:
                                continue
                            e_intvls_multi.add(ErrorIntvl(b=b, e=e, pe=pe))
                            already_paired |= set([i, j])
                            if verbose:
                                print(f"Multi(from {wtype}) ({b},{e}): pe={pe}")
                    if j in walls[Etype.OTHERS]:
                        break
    for e_intvl in e_intvls_multi:
        for i in range(e_intvl.b + 1, e_intvl.e):
            walls[Etype.OTHERS].discard(i)
    e_intvls |= e_intvls_multi

    # Remove E-intvls fully contained within E-intvls
    e_intvls_to_be_removed = set()
    for I in e_intvls:
        for J in e_intvls:
            if I == J:
                continue
            if J.b <= I.b and I.e <= J.e:
                e_intvls_to_be_removed.add(I)
    e_intvls -= e_intvls_to_be_removed

    # Merge overlapping E-intvls
    e_intvls_to_be_removed = set()
    e_intvls_to_be_added = set()
    sorted_e_intvls = sorted(e_intvls, key=lambda x: x.b)
    i = 0
    while i < len(sorted_e_intvls) - 1:
        j = i
        while j < len(sorted_e_intvls) - 1:
            I, J = sorted_e_intvls[j: j + 2]
            if I.b < J.b and J.b < I.e and I.e < J.e:
                if verbose:
                    print(f"Merged ({I.b},{I.e}) & ({J.b},{J.e})")
                j += 1
            else:
                break
        if i < j:
            e_intvls_to_be_removed |= set(
                [sorted_e_intvls[x] for x in range(i, j + 1)])
            e_intvls_to_be_added.add(ErrorIntvl(b=sorted_e_intvls[i].b,
                                                e=sorted_e_intvls[j].e,
                                                pe=max(sorted_e_intvls[i].pe,
                                                       sorted_e_intvls[j].pe)))
        i = j + 1
    e_intvls -= e_intvls_to_be_removed
    e_intvls |= e_intvls_to_be_added

    # Remove O-intvls contained in E-intvls
    o_intvls_to_be_removed = set()
    for I in o_intvls:
        for J in e_intvls:
            if J.b <= I.b and I.e <= J.e:
                o_intvls_to_be_removed.add(I)
    o_intvls -= o_intvls_to_be_removed

    # Redefine walls by errors in self
    walls[Etype.SELF] = set([x for I in e_intvls for x in (I.b, I.e)])

    # Merge walls
    merged_walls = [False] * (pread.length + 1)
    for i in sorted(walls[Etype.SELF] | walls[Etype.OTHERS]):
        merged_walls[i] = True

    if verbose:
        ws, wo = [walls[etype] for etype in Etype]
        print(f"# of walls = {len(ws)} (S), {len(wo)} (O), {len(ws & wo)} (S&O)\n"
              f"only S (wall by E in S but also explained by E in O) = {sorted(ws - wo)}\n"
              f"only O (wall by not E in S) = {sorted(wo - ws)}\n"
              f"E-intvls = {[(I.b, I.e) for I in sorted(e_intvls, key=lambda I: I.b)]}\n"
              f"O-intvls = {[(I.b, I.e) for I in sorted(o_intvls, key=lambda I: I.b)]}")

    # Determine walls and intervals from `walls` and `*_intvls`
    pread.walls = sorted(walls[Etype.SELF] | walls[Etype.OTHERS] | set([0, pread.length]))
    e_intvls_by_pos = {(I.b, I.e): I for I in e_intvls}
    pread.intvls = []
    for i in range(len(pread.walls) - 1):
        b, e = pread.walls[i:i + 2]
        pread.intvls.append(
            Intvl(b=b, e=e,
                  pe=(e_intvls_by_pos[(b, e)].pe if (b, e) in e_intvls_by_pos   # TODO: log pe?
                      else 0.),
                  pe_o=PerrorInO(b=max([perrors[b, Etype.OTHERS, wtype]
                                        if (b, Etype.OTHERS, wtype) in perrors else 0.   # TODO: perrors=defaultdict(lambda: 0.)
                                        for wtype in Wtype]),
                                 e=max([perrors[e, Etype.OTHERS, wtype]
                                        if (e, Etype.OTHERS, wtype) in perrors else 0.
                                        for wtype in Wtype])),
                  cb=pread.counts[b],
                  ce=pread.counts[e - 1]))

    pread.e_intvls = e_intvls
    pread.o_intvls = o_intvls

    return


def correct_wall_cnt(I, pread):
    s, t = I.b, I.e
    n_gain = ([max(pread.counts[i + 1] - pread.counts[i], 0)
               for i in range(s, min(s + pread.K - 1, t - 1))]
              + ([-max(pread.counts[i] - pread.counts[i + 1], 0)
                  for i in range(s,
                                 s + max([pread.ctx[s + pread.K - 1][ctype.value][Wtype.GAIN.value] * ulen
                                          for ulen, ctype in enumerate(Ctype, start=1)]))]
                 if s + pread.K - 1 < t else []))
    n_drop = ([max(pread.counts[i] - pread.counts[i + 1], 0)
               for i in range(max(t - pread.K + 1, s), t - 1)]
              + ([-max(pread.counts[i + 1] - pread.counts[i], 0)
                  for i in range(t - max([pread.ctx[t - pread.K + 1][ctype.value][Wtype.DROP.value] * ulen
                                          for ulen, ctype in enumerate(Ctype, start=1)]),
                                 t - 1)]
                 if s < t - pread.K + 1 else []))
    ccb_ctx = pread.counts[s] + max(sum(n_gain), 0)
    cce_ctx = pread.counts[t - 1] + max(sum(n_drop), 0)

    ccb_naive = max(pread.counts[s:min(s + 2 * pread.K, t)])
    cce_naive = max(pread.counts[max(t - 2 * pread.K, s):t])

    I.ccb = max(ccb_ctx, ccb_naive)
    I.cce = max(cce_ctx, cce_naive)

    return


def is_reliable(I, pread, cp, verbose=False):
    # Requirment 1: |I| >= K
    if I.e - I.b < pread.K:
        return None
    # Requirment 2: Not > R-cov
    if max(I.cb, I.ce) >= cp.depths['R']:   # TODO: check again after count correction?
        return None
    # Requirment 3: Unlikely to be E-intvl
    if I.pe >= PE_THRES[ThresT.FINAL][Etype.SELF]:
        return None
    # Requirment 4: Smooth after count correction
    correct_wall_cnt(I, pread)
    p_trans = calc_p_trans(I.b, I.e, I.ccb, I.cce,
                           (I.ccb + I.cce) // 2, cp.read_len)
    if p_trans < PTHRES_DIFF_REL:
        if verbose:
            print(f"({I.b},{I.e}): p_trans={p_trans} < {PTHRES_DIFF_REL}")
        return None
    I.is_rel = True
    return I


def find_rel_intvls(pread, cp, verbose=False):
    pread.rel_intvls = list(filter(lambda x: x is not None,
                                   [is_reliable(I, pread, cp, verbose)
                                    for I in pread.intvls]))
    return

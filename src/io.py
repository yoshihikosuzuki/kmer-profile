import os
from typing import Optional, Tuple
from collections import Counter
from BITS.util.proc import run_command
from .type import StateThresholds, RelCounter, ProfiledRead


def load_count_dist(db_fname: str,
                    max_count: Optional[int] = None) -> Optional[Tuple[RelCounter, StateThresholds]]:
    """Load global k-mer count frequencies in a database using `REPkmer` command.

    positional arguments:
      @ db_fname : DAZZ_DB file name.

    optional arguments:
      @ max_count : `-x` option for `REPkmer` command.

    return value:
      @ k-mer count frequencies
      @ thresholds between k-mer states
    """
    if not (isinstance(db_fname, str) and os.path.exists(db_fname)):
        return None
    option = f"-x{max_count}" if max_count is not None else ""
    command = f"REPkmer {option} {db_fname}"
    lines = run_command(command).strip().split('\n')
    # Load thresholds for classification and then skip to the histogram
    for i, line in enumerate(lines):
        if line.strip().startswith("Error"):
            _, _, eh, _, _, _, hd, _, _, _, dr, _, _ = line.strip().split()
        if line.strip() == "K-mer Histogram":
            lines = lines[i:]
            break
    # Load the histogram
    count_freqs = Counter()
    for line in lines:
        data = line.strip().split()
        if len(data) != 3:
            continue
        kmer_count, freq, _ = data
        kmer_count = kmer_count[:-1]
        if kmer_count[-1] == '+':
            kmer_count = kmer_count[:-1]
        count_freqs[int(kmer_count)] = int(freq)
    return (RelCounter(count_freqs),
            StateThresholds(error_haplo=eh,
                            haplo_diplo=hd,
                            diplo_repeat=dr))


def load_kmer_profile(db_fname: str,
                      read_id: int) -> Optional[ProfiledRead]:
    """Load k-mer count profile of a single read using `KMlook` command.

    positional arguments:
      @ db_fname : DAZZ_DB file name.

    optional arguments:
      @ read_id : Single DAZZ_DB read ID to be loaded.

    return value:
      @ Sequence record with count profile
    """
    if not (isinstance(db_fname, str) and os.path.exists(db_fname)):
        return None
    command = f"KMlook {db_fname} {read_id}"
    lines = run_command(command).strip().split('\n')
    read_length = int(lines[2].strip().split()[2])
    bases, counts = [''] * read_length, [0] * read_length
    for line in lines[3:]:
        data = line.strip().split()
        if data[0][-1] == ':':
            if data[2] == 'Z':
                pos, base = data[:2]
                count = 0
            else:
                pos, base, count = data[:3]
            pos = int(pos[:-1])
            bases[pos] = base
            counts[pos] = int(count)
    return ProfiledRead(seq=''.join(bases),
                        counts=counts)

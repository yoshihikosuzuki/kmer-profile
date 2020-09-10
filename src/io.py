from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from typing_extensions import TypedDict
from collections import Counter
from BITS.util.proc import run_command


class CountThresholds(TypedDict):
    error_haplo: int
    haplo_diplo: int
    diplo_repeat: int


@dataclass
class REPkmerResult:
    count_freqs: Counter
    thresholds: CountThresholds

    @property
    def count_rel_freqs(self) -> Counter:
        return to_rel_freqs(self.count_freqs)


@dataclass
class CountProfile:
    counts: List[int]
    bases: List[str]

    def count_freqs(self, max_count: Optional[int] = None) -> Counter:
        counts = (self.counts if max_count is None
                  else [min(count, max_count) for count in self.counts])
        return Counter(list(filter(lambda x: x > 0, counts)))

    def count_rel_freqs(self, max_count: Optional[int] = None) -> Counter:
        return to_rel_freqs(self.count_freqs(max_count))


def to_rel_freqs(c: Counter) -> Counter:
    tot_freq = sum(list(c.values()))
    return {k: v / tot_freq * 100 for k, v in c.items()}


def load_count_dist(db_fname: str,
                    max_count: Optional[int] = None) -> Optional[REPkmerResult]:
    """Load global k-mer count frequencies in the database."""
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
    return REPkmerResult(count_freqs=count_freqs,
                         thresholds=CountThresholds(error_haplo=eh,
                                                    haplo_diplo=hd,
                                                    diplo_repeat=dr))


def load_kmer_profile(db_fname: str,
                      read_id: int) -> Optional[CountProfile]:
    """Load k-mer count profile of a single read."""
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
    return CountProfile(bases=bases,
                        counts=counts)

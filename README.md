# kmer-profile
Scripts for kmer profile analysis

## Requirements

Followings are required to be installed before installing this package (i.e. these are not automatically installed):

- Python >= 3.7
- [FASTK](https://github.com/thegenemyers/FASTK)
- [FASTK_python](https://github.com/yoshihikosuzuki/FASTK_python)

## Optional requirements

- Followings must be installed in your `PATH` if necessary
  - [Seqkit](https://bioinf.shenwei.me/seqkit/) (if you use fasta/fastq files)
  - [DAZZ_DB](https://github.com/thegenemyers/DAZZ_DB) (if you use db/dam files)

## How to install 

```bash
$ git clone https://github.com/yoshihikosuzuki/kmer-profile
$ cd kmer-profile
$ python3 setup.py install
```

A binary executable for visualization, `kmer_profile`, is installed in addition to a python module.

## How to use

### Visualize k-mer count histogram and count profile

Run the following command and then open `http://localhost:8050` in a browser (port number can be changed with the option `-p`).

```bash
$ kmer_profile -s <sequence_file> <fastk_prefix>
```

To show the help message, run `$ kmer_profile -h`:

```text
usage: kmer_profiler [-h] [-s SEQ_FNAME] [-c CLASS_FNAME] [-p PORT_NUMBER]
                     [-d]
                     fastk_prefix

K-mer count profile visualizer

positional arguments:
  fastk_prefix          Prefix of FastK's output files.

optional arguments:
  -h, --help            show this help message and exit
  -s SEQ_FNAME, --seq_fname SEQ_FNAME
                        Name of the input file for FastK. Must be
                        .db/dam/fast[a|q]. [None]
  -c CLASS_FNAME, --class_fname CLASS_FNAME
                        K-mer classification result file name. [None]
  -p PORT_NUMBER, --port_number PORT_NUMBER
                        Port number to run the server. [8050]
  -d, --debug_mode      Run a Dash server in a debug mode.
```

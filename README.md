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

You are supposed to have FastK's outputs with the `-p` option (i.e. you need to have a `.prof` file) for your dataset.

### Visualize k-mer count histogram and count profile

Run the following command and then open `http://localhost:8050` in a browser (port number can be changed with the option `-p`).

```bash
$ kmer_profiler [-s <sequence_file>] <fastk_prefix>
```

To show the help message, run `$ kmer_profile -h`:

```text
usage: kmer_profiler [-h] [-s SEQ_FNAME] [-c CLASS_FNAME] [-p PORT_NUMBER]
                     [-f IMG_FORMAT] [-d]
                     fastk_prefix

K-mer count profile visualizer

positional arguments:
  fastk_prefix          Prefix of FastK's output files. Both
                        `<fastk_prefix>.hist` and `<fastk_prefix>.prof` must
                        exist.

optional arguments:
  -h, --help            show this help message and exit
  -s SEQ_FNAME, --seq_fname SEQ_FNAME
                        Name of the input file for FastK containing reads.
                        Must be .db/dam/fast[a|q]. Used for displaying baes in
                        profile plot [None]
  -c CLASS_FNAME, --class_fname CLASS_FNAME
                        File name of K-mer classification result. [None]
  -p PORT_NUMBER, --port_number PORT_NUMBER
                        Port number of localhost to run the server. [8050]
  -f IMG_FORMAT, --img_format IMG_FORMAT
                        Format of plot images you can download with camera
                        icon. ['svg']
  -d, --debug_mode      Run a Dash server in a debug mode.
```

**!!! IMPORTANT !!!**: If you use `DOWNLOAD HTML` buttons in the viewer, disable your browser's cache (only in the viewer's tab) For example, in Chrome, go to `Developer Tool` → `Network` tab → `Disable cache` checkbox and keep the Developer Tool open while using the viewer.

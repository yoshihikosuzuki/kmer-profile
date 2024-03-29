# kmer-profile

A [Dash](https://plotly.com/dash/) app for interactive visualization of read profiles generated with [FASTK](https://github.com/thegenemyers/FASTK).

![Example view](assets/example-view.png)

**NOTE**: This package also contains python version of [ClassPro](https://github.com/yoshihikosuzuki/ClassPro), but it is only for development and not intended for usage by users.

## Requirements

Followings are required to be installed before installing this package (i.e. these are not automatically installed):

- Python >= 3.7
- [FASTK](https://github.com/thegenemyers/FASTK)
- [FASTK_python](https://github.com/yoshihikosuzuki/FASTK_python)

### Optional requirements

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

## How to use the visualizer

You are supposed to have FastK's outputs with the `-p` option (i.e. you need to have a `.prof` file) for your dataset.

You can start the visualizer with the command named `kmer_profiler`. The usage is as follows (this message can also be shown via `$ kmer_profile -h`):

```text
usage: kmer_profiler [-h] [-s SEQ_FNAME] [-c CLASS_FNAME]
                     [-t TRUTH_CLASS_FNAME] [-p PORT_NUMBER] [-f IMG_FORMAT]
                     [-d]
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
  -t TRUTH_CLASS_FNAME, --truth_class_fname TRUTH_CLASS_FNAME
                        File name of ground truth of K-mer classification
                        result. [None]
  -p PORT_NUMBER, --port_number PORT_NUMBER
                        Port number of localhost to run the server. [8050]
  -f IMG_FORMAT, --img_format IMG_FORMAT
                        Format of plot images you can download with camera
                        icon. ['svg']
  -d, --debug_mode      Run a Dash server in a debug mode.
```

After running this, open `http://localhost:8050` (port number can be changed with the option `-p`) in a browser.

**[NOTE]**
To use `DOWNLOAD HTML` buttons in the app, you might need to disable your browser's cache (only of the appr's tab). To do this, e.g. in Chrome, go to `Developer Tool` → `Network` tab → `Disable cache` checkbox and keep the Developer Tool open while using the viewer.

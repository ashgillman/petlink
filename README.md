Python tools for handling Interfile and DICOM variants.

# Usage

## Converting DICOM to interfile
From command line:

    pettools dcm2ifl INFILE.ptd OUTFILE

or in Python:

```python
from petlink import InterfileCSA
iflcsa = InterfileCSA.from_file(in_file)
iflcsa.to_interfile(out_file)
```

## Exploring Interfile data
```python
from petlink import Interfile
ifl = Interfile('listmode.hl')

# access keys
ifl['patient orientation']  # e.g., HFS or head_in

# print all keys
print(ifl)

# access data as numpy array
ifl.get_data()

# parse date/time key pairs as Python datetime objects
ifl.get_datetime('study') - ifl.get_datetime('tracer injection')                                             
# -> datetime.timedelta(seconds=548)
```

## Exploring ListMode data
```python
from petlink import ListMode
lm = ListMode('listmode.ptd')  # or .IMA or .dcm+.bf

# get the duration
lm.duration   # time in ms

# raw data
lm.data  # 1D numpy array

# DICOM metadata
lm.dcm  # pydicom object

# Interfile header
lm.ifl  # Interfile object, see above
```

### Indexing ListMode data by time

```python
start_time_ms, end_time_ms = 1000, 2000

between_one_and_two_seconds = lm.tloc[start_time_ms:end_time_ms]

```

# Installation
The package is configured with setuptools, and should integrate well with pip.
The simplest option is:

    pip install --user git+https://github.com/ashgillman/petlink.git

Or to pull the repo locally an use:

    pip install --user .

Nix users can run `nix-shell` or `nix-env -if .`. 

# Building
To use without installation, cython modules will need to be build in-place.

    python3 setup.py build_ext --inplace

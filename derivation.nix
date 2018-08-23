{ buildPythonPackage
#, python
, click
, cython
, numpy
, progressbar2
, pydicom
, pyparsing
, scipy
, scikitlearn
, pytest
, pytestrunner
}:

let
  python_build_dependencies = [
    pytest
    pytestrunner
  ];

  python_dependencies = [
    #python        # The Interpreter itself
    click         # CLI
    cython        # Wrapping cpp in Python
    numpy         # MATLAB-like vectorised computation
    progressbar2  # Progress feedback
    pydicom       # DICOM reading for .ptd files
    pyparsing     # Parsing of text
    scipy         # Numeric tools
    scikitlearn   # Learning tools
  ];

in
  buildPythonPackage rec {
    pname = "petlink";
    version = "0.0.0";
    src = builtins.fetchGit { url = ./.; };  # respect .gitignore

    checkInputs = python_build_dependencies;
    propagatedBuildInputs = python_dependencies;
    doCheck = false;  # don't have test data
  }

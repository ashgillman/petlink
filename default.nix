{ pkgs ? import <mypkgs> {}
, stdenv ? pkgs.stdenv
, pythonPackages ? pkgs.python3Packages
}:

let
  python_build_dependencies =
    with pythonPackages;
    [
      pytest
      pytestrunner
    ];

  python_dependencies =
    with pythonPackages;
    [
      python        # The Interpreter itself
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
  pythonPackages.buildPythonPackage rec {
    pname = "petlink";
    version = "0.0.0";
    src = ./.;

    buildInputs = python_build_dependencies;
    propagatedBuildInputs = python_dependencies;
    doCheck = false;  # don't have test data
  }

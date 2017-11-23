{ pkgs ? import <mypkgs> {}
, stdenv ? pkgs.stdenv
, pythonPackages ? pkgs.python35Packages
}:

let
  python_build_dependencies =
    with pythonPackages;
    [
      pytest
    ];

  python_dependencies =
    with pythonPackages;
    [
      python     # The Interpreter itself
      numpy      # MATLAB-like vectorised computation
      pydicom    # DICOM reading for .ptd files
      pyparsing  # Parsing of text
    ];

in
  pythonPackages.buildPythonPackage rec {
    name = "petlink";
    src = ./.;

    buildInputs = python_build_dependencies;
    propagatedBuildInputs = python_dependencies;
  }

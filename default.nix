{ pkgs ? import <nixpkgs> {}
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
      python     # The Interpreter itself
      cython     # Wrapping cpp in Python
      numpy      # MATLAB-like vectorised computation
      pydicom    # DICOM reading for .ptd files
      pyparsing  # Parsing of text
    ];

in
  pythonPackages.buildPythonPackage rec {
    pname = "petlink";
    version = "0.0.0";
    src = builtins.filterSource
      (path: type:
           ! (type == "directory" && baseNameOf path == "data")
        && ! (type == "directory" && baseNameOf path == "build")
        && ! (type == "directory" && baseNameOf path == ".eggs")
        && ! (type == "directory" && baseNameOf path == ".gcroots")
        && ! (type == "symlink"   && baseNameOf path == "result")
      )
      ./.;

    buildInputs = python_build_dependencies;
    propagatedBuildInputs = python_dependencies;
    doCheck = false;  # don't have test data
  }

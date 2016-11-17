{ pkgs ? import <nixpkgs> {}
, stdenv ? pkgs.stdenv
, pythonPackages ? pkgs.python34Packages
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
      pyparsing  # Parsing of text
    ];

in
  pythonPackages.buildPythonPackage rec {
    name = "petlink";
    src = ./.;

    buildInputs = python_build_dependencies;
    propagatedBuildInputs = python_dependencies;
  }

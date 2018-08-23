{ pkgs ? import <nixpkgs> {}
, python3Packages ? pkgs.python3Packages
}:

let callPackage = pkgs.newScope python3Packages;
in  callPackage ./derivation.nix {}

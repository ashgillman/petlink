(import ./default.nix {}).overrideAttrs (
  oldAttrs: {
    shellHook = ''
      export PYTHONPATH=$PWD/install/lib/python3.6/site-packages/:$PYTHONPATH
    '';
  }
)

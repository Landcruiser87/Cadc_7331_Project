let
  jupyter = import (builtins.fetchGit {
    url = https://github.com/tweag/jupyterWith;
    rev = "";
  }){};
  
  iPython = jupyter.kernels.iPythonWith {
    name = "python";
    packages = p: with p; [
      numpy
      pandas
      rpy2
      matplotlib
      scikitlearn
      seaborn
      scipy
      future
      ipywidgets
      scikitimage
      tzlocal
      simplegeneric
      pprintpp
    ];
  };

  iR = jupyter.kernels.juniperWith {
    name = "R";
    packages = p: with p; [
       mlbench
       arules
       tidyverse
       ggplot2
       dplyr
    ];
  };

#  iHaskell = jupyter.kernels.iHaskellWith {
#    name = "haskell";
#    packages = p: with p; [ hvega formatting ];
#  };

  jupyterEnvironment = jupyter.jupyterlabWith {
    kernels = [iPython iR];
  };
in
  jupyter.mkDockerImage {
    name = "jupyter-image";
    jupyterlab = jupyterEnvironment;
  }

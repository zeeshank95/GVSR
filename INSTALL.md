# Installation

We provide instructions to install the required dependencies.

Requirements:
+ python>=3.6
+ pytorch==1.5 (should work with pytorch >=1.5 as well but not tested)

1. Clone the repo with all the submodules:
    ```
    git clone --recurse-submodule https://github.com/zeeshank95/GVSR.git
    cd GVSR
    export ROOT=$(pwd)
    ```

2. To use the same environment you can use conda and the environment file vsitu_pyt_env.yml file provided.
Please refer to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for details on installing conda.

    ```
    MINICONDA_ROOT=[to your Miniconda/Anaconda root directory]
    conda env create -f vsitu_pyt_env.yml --prefix $MINICONDA_ROOT/envs/vsitu_pyt
    conda activate vsitu_pyt
    ```

3. Install submodules:

    + cocoapi:
    ```
    cd $ROOT/cocoapi/PythonAPI
    make
    ```
    + coco-caption: (NOTE: You may need to install java). No additional steps are needed.

    + coval:
    ```
    cd $ROOT/coval
    pip install -e .
    ```
    + fairseq
    cd $ROOT/fairseq
    pip install -e .

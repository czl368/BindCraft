#!/bin/bash
################## BindCraft installation script
################## edited for virtual env

### BindCraft install begin, create base environment
echo -e "Installing BindCraft environment\n"

# loading modules
# TODO figure out how to install libgfortran5 (--upgrade scipy was to try to install libgfortran5)
# NOTE --> running into "Floating point exception (core dumped)" error
# assuming cuda-nvcc is already installed
# scipy-stack/2024a is a module that loads numpy, scipy, matplotlib, pandas, and other packages
# openmm/8.1.1 is a module that loads pdbfixer
module load python/3.10 StdEnv/2023 scipy-stack/2024a openmm/8.1.1 cudacore/.12.2.2 cudnn/8.9.5.29
module load python/3.10

# creating a virtual environment
ENVDIR=bindcraft_env
virtualenv $ENVDIR

# Load newly created BindCraft environment
echo -e "Loading BindCraft environment\n"
source $ENVDIR/bin/activate
echo -e "BindCraft environment activated at ${ENVDIR}/envs/BindCraft"
pip install --no-index --upgrade pip

# install packages
# --no-index option tells pip to not install from PyPI, but instead to install only from locally available packages
# (i.e. our wheels)
echo -e "Installing virtualenv requirements\n"
pip install --no-index biopython seaborn tqdm jupyter fsspec py3dmol chex dm-haiku flax"<0.10.0" dm-tree joblib ml-collections immutabledict optax jaxlib jax 
pip install ffmpeg

# install pyrosetta
# To test installation --> the following commands should not have an error
# python
# import pyrosetta; pyrosetta.init()
wget https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python310.linux/PyRosetta4.Release.python310.linux.release-387.tar.bz2
tar -vjxf PyRosetta4.Release.python310.linux.release-387.tar.bz2
cd PyRosetta4.Release.python310.linux.release-387
cd setup && python setup.py install

# storing packages in requirements.txt
pip freeze --local > requirements.txt
#deactivate


# From original install script
install_dir=$(pwd)

# install ColabDesign
echo -e "Installing ColabDesign\n"
pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps || { echo -e "Error: Failed to install ColabDesign"; exit 1; }
python -c "import colabdesign" >/dev/null 2>&1 || { echo -e "Error: colabdesign module not found after installation"; exit 1; }

# AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights \n"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

# download AF2 weights
mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }
[ -s "${params_file}" ] || { echo -e "Error: Could not locate downloaded AlphaFold2 weights"; exit 1; }

# extract AF2 weights
tar tf "${params_file}" >/dev/null 2>&1 || { echo -e "Error: Corrupt AlphaFold2 weights download"; exit 1; }
tar -xvf "${params_file}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2weights"; exit 1; }
[ -f "${params_dir}/params_model_5_ptm.npz" ] || { echo -e "Error: Could not locate extracted AlphaFold2 weights"; exit 1; }
rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }

# chmod executables --> files don't exist
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/BindCraft/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }
chmod +x "${install_dir}/BindCraft/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# finish
deactivate
echo -e "BindCraft environment set up\n"

################## finish script
t=$SECONDS 
echo -e "Successfully finished BindCraft installation!\n"
echo -e "Activate environment using command: \"source $ENVDIR/bin/activate \""
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."

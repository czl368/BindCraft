#!/bin/bash
################## BindCraft installation script
################## edited for virtual env

### BindCraft install begin, create base environment
echo -e "Installing BindCraft environment\n"

# creating a virtual environment
module load python/3.10
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
echo -e "Installing conda/virtualenv requirements\n"
#pip install --no-index tensorflow

# loading modules
# assuming cuda-nvcc is already installed
# scipy-stack/2024a is a module that loads numpy, scipy, matplotlib, pandas, and other packages
# openmm/8.1.1 is a module that loads pdbfixer
module load python/3.10
module load StdEnv/2023 scipy-stack/2024a 
module load openmm/8.1.1
module load StdEnv/2020 ffmpeg/4.3.2 
module load cudacore/.12.2.2 cudnn/8.9.5.29
# TODO figure out how to install libgfortran5 (--upgrade scipy was to try to install libgfortran5) and pyrosetta
#pip install --upgrade scipy
#pip install biopython seaborn libgfortran5 tqdm jupyter pyrosetta fsspec py3dmol chex dm-haiku flax"<0.10.0" dm-tree joblib ml-collections immutabledict optax jaxlib jax 
pip install --no-index biopython seaborn tqdm jupyter fsspec py3dmol chex dm-haiku flax"<0.10.0" dm-tree joblib ml-collections immutabledict optax jaxlib jax 

# storing packages in requirements.txt
pip freeze --local > requirements.txt
deactivate






# Below from original install script
# TODO - figure out which packages to install vs modules to load


# install required conda packages
echo -e "Instaling conda requirements\n"
if [ -n "$cuda" ]; then
    CONDA_OVERRIDE_CUDA="$cuda" $pkg_manager install pip pandas matplotlib numpy"<2.0.0" biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku flax"<0.10.0" dm-tree joblib ml-collections immutabledict optax jaxlib=*=*cuda* jax cuda-nvcc cudnn -c conda-forge -c nvidia  --channel https://conda.graylab.jhu.edu -y || { echo -e "Error: Failed to install conda packages."; exit 1; }
else
    $pkg_manager install pip pandas matplotlib numpy"<2.0.0" biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku flax"<0.10.0" dm-tree joblib ml-collections immutabledict optax jaxlib jax cuda-nvcc cudnn -c conda-forge -c nvidia  --channel https://conda.graylab.jhu.edu -y || { echo -e "Error: Failed to install conda packages."; exit 1; }
fi

# make sure all required packages were installed
required_packages=(pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer seaborn tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku dm-tree joblib ml-collections immutabledict optax jaxlib jax cuda-nvcc cudnn)
missing_packages=()

# Check each package
for pkg in "${required_packages[@]}"; do
    conda list "$pkg" | grep -w "$pkg" >/dev/null 2>&1 || missing_packages+=("$pkg")
done

# If any packages are missing, output error and exit
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following packages are missing from the environment:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
fi

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

# chmod executables
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# finish
conda deactivate
echo -e "BindCraft environment set up\n"

############################################################################################################
############################################################################################################
################## cleanup
echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
$pkg_manager clean -a -y
echo -e "$pkg_manager cleaned up\n"

################## finish script
t=$SECONDS 
echo -e "Successfully finished BindCraft installation!\n"
echo -e "Activate environment using command: \"$pkg_manager activate BindCraft\""
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
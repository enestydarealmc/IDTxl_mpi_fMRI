#SBATCH --job-name=macaque_mpi
#SBATCH --output=macaque_idtxl_mpi_%j.txt
#SBATCH --time=2:00:00
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb

cd /usr/users/$USER

export PYTHONPATH=/usr/users/$USER/IDTxl
export JAVA_HOME=/usr/users/$USER/jdk-16.0.1

srun --nodes=4 -n=16 --mpi=pmi2 \
  python -m mpi4py.futures macaque_idtxl_mpi.py \
  --bold-path SmallDegree/data_fslfilter_concat/concat_BOLDfslfilter_01.txt \
  --truth-path SmallDegree/graph/Macaque_SmallDegree_graph.txt \
  --cmi-estimator OpenCLKraskovCMI \
  --verbose \
  --max-workers 15

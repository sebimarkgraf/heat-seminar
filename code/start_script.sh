#!/usr/bin/bash -x
#SBATCH --exclusive
#SBATCH --job=BigDataTools
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=00:10:00
#SBATCH --output=output_file.out
#SBATCH --error=output_file.out

module restore HeAT
source $HOME/heat-seminar/code/venv/bin/activate
unset PYTHONPATH
export OMP_NUM_THREADS=X

function runbenchmark()
{
    echo $(date)
    echo "Benchmark Parameters:"
    echo "Total number of tasks: ${SLURM_NTASKS}"
    echo "Number of Nodes: ${SLURM_JOB_NUM_NODES}"
    echo "Tasks per Node: ${SLURM_NTASKS_PER_NODE}"
    mpirun python ./clustering.py
}

runbenchmark

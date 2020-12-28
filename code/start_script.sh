#!/usr/bin/bash -x
#SBATCH --exclusive
#SBATCH --job=BigDataTools
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=output_file.out
#SBATCH --error=output_file.out
#SBATCH --mem=90000mb

module restore HeAT
source $HOME/heat-seminar/code/venv/bin/activate
unset PYTHONPATH

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

#! /usr/bin/env bash
#SBATCH --exclusive
#SBATCH --J=BigDataTools
#SBATCH --nodes=N
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=X
#SBATCH --time=hh:mm:ss
#SBATCH --output=output_file.out
#SBATCH --error=output_file.out

module restore HeAT
source /home/kit/stud/ulxhv/haet_seminar/code/venv/bin/activate
unset PYTHONPATH
export OMP_NUM_THREADS=X

function runbenchmark()
{
    echo $(date)
    echo "Benchmark Parameters:"
    echo "Total number of tasks: ${SLURM_NTASKS}"
    echo "Number of Nodes: ${SLURM_JOB_NUM_NODES}"
    echo "Tasks per Node: ${SLURM_NTASKS_PER_NODE}"

    srun python -u clustering.py
}
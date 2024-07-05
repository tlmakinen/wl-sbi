#PBS -S /bin/bash
#PBS -N imnn_N192
#PBS -j oe
#PBS -o N192_run.log
#PBS -n
#PBS -l nodes=h12:has1gpu:ppn=40,walltime=24:00:00
#PBS -M l.makinen21@imperial.ac.uk

module load cuda/12.2
module load intelpython/3-2024.0.0
XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS
source /home/makinen/venvs/lemur/bin/activate


cd /home/makinen/repositories/wl-sbi/automate_run/

# configs, noise level, previous noise level (-1 to start from scratch)
python run.py config_N192.yaml 0.125 -1

# retrain on same noise level to the same patience setting as before
python run.py config_N192.yaml 0.125 0

# keep running for increasing noise levels
python run.py config_N192.yaml 0.15 0

python run.py config_N192.yaml 0.20 1

python run.py config_N192.yaml 0.25 2

python run.py config_N192.yaml 0.30 3

python run.py config_N192.yaml 0.35 4

python run.py config_N192.yaml 0.40 5

python run.py config_N192.yaml 0.45 6

python run.py config_N192.yaml 0.50 7

python run.py config_N192.yaml 0.55 8

python run.py config_N192.yaml 0.60 9

python run.py config_N192.yaml 0.65 10

python run.py config_N192.yaml 0.70 11

python run.py config_N192.yaml 0.75 12

python run.py config_N192.yaml 0.80 13

python run.py config_N192.yaml 0.85 14

python run.py config_N192.yaml 0.90 15

python run.py config_N192.yaml 0.95 16

python run.py config_N192.yaml 1.0 17
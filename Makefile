job:
	@make venv
	@make setup
	@make run

run:
	python3 './Source Code/main.py' --model PSPNet --root-dir ./Data --output-dir "$HOME/scratch/medical-segmentation"

setup: requirements.txt
	pip install --upgrade --no-index -r requirements.txt

venv:
	module load python/3.8
	virtualenv --no-download "${SLURM_TMPDIR}/env"
	source "${SLURM_TMPDIR}/env/bin/activate"

#!/bin/bash

# Abfrage der Parameter###################################################################################################
echo "Bitte geben Sie den Job-Namen ein (Beispiel: my_job): "
read Jobname

echo "Bitte geben Sie die Anzahl der Knoten ein (Beispiel: 2, drücken Sie Enter für 1): "
read Node
Node=${Node:-1}

echo "Bitte geben Sie die Anzahl der Tasks ein (Beispiel: 10, drücken Sie Enter für 1): "
read Task
Task=${Task:-1}

echo "Bitte geben Sie die Anzahl der CPUs pro Task ein (Beispiel: 4, drücken Sie Enter für 4): "
read CPU
CPU=${CPU:-4}

echo "Bitte geben Sie den Arbeitsspeicher für den Task an: (Beispiel: 4G für 4 Gigabyte, drücken Sie Enter für 16G): "
read Memory
Memory=${Memory:-16G}

echo "Bitte geben Sie die Anzahl der benötigten GPUs an (Beispiel: 2, drücken Sie Enter für keine GPUs): "
read GPU
GPU=${GPU:-0}

# Eingabe des Environments und der Datei##################################################################################
Dir=$(pwd)
echo "Verfügbare Environments:"
find $Dir/environments -maxdepth 1 -mindepth 1 -type d -printf "%f\n"

echo "Bitte geben Sie den Namen des Environments an (Beispiel: envtest): "
read Env

find $Dir -maxdepth 1 -type f -name "*.py"

echo "Bitte geben Sie den Pfad der Datei an (Beispiel: /data/resources/user/work/script.py): "
read File

# Erstellen der SBATCH-Datei############################################################################################
SBATCH_FILE="$Jobname.slurm"
cat <<EOL > $SBATCH_FILE
#!/bin/bash
#SBATCH --job-name=$Jobname
#SBATCH --nodes=$Node
#SBATCH --ntasks=$Task
#SBATCH --cpus-per-task=$CPU
#SBATCH --mem=$Memory
#SBATCH --output=out/slurm-%j.out
#SBATCH --error=err/slurm-%j.err
EOL

# Add GPU configuration only if GPU is not 0
if [ "$GPU" -ne 0 ]; then
    echo "#SBATCH --gres=gpu:$GPU" >> $SBATCH_FILE
fi

cat <<EOL >> $SBATCH_FILE
source $Dir/environments/$Env/bin/activate
python3 $File
EOL

# Ausgabe der SBATCH-Datei##############################################################################################
echo "Die SBATCH-Datei wurde erstellt:"
cat $SBATCH_FILE

# Einreichen des Jobs###################################################################################################
echo "Möchten Sie den Job jetzt einreichen? (j/n)"
read SUBMIT

if [ "$SUBMIT" == "j" ]
then
    sbatch $SBATCH_FILE
else
    echo "Job wurde nicht eingereicht."
fi

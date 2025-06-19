#!/bin/bash

# Directory für die Environments
TARGET_DIR="environments"

# Prüfen, ob Zielordner existiert
if [ ! -d "$TARGET_DIR" ]
then
    mkdir -p "$TARGET_DIR"
    echo "Ordner $TARGET_DIR wurde erstellt."
fi

#Erstellung des Environments###################################################################################################
echo "Bitte geben Sie den Namen des Environments ein: "
read env

if [ -z "$env" ]
then
    echo "Keine gültige Eingabe"
    exit 1
fi

# Prüfen, ob Environment bereits existiert
ENV_NAME="$TARGET_DIR/$env"
if [ -d "$ENV_NAME" ]
then
    echo "Environment $ENV_NAME existiert bereits."
    exit 1
fi

# Erstellen des Environments
python3 -m venv $ENV_NAME

# Aktivieren des Environments
source $ENV_NAME/bin/activate

# Installieren der Requirements
pip install -r requirements.txt

sleep 10

#Erstellung des Job Skripts###################################################################################################
# Pfad zur Slurm-Job-Skriptdatei
job_script="job.slurm"

# Inhalt der Slurm-Job-Skriptdatei
cat <<EOF > $job_script
#!/bin/bash
#SBATCH --job-name=py-job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=out/slurm-%j.out
#SBATCH --error=err/slurm-%j.err

source $ENV_NAME/bin/activate

python3 mistral_train.py
EOF

chmod +x $job_script

#Abschluss####################################################################################################################
# Pfad zur Aktivierung des virtuellen Environments
echo "$ENV_NAME/bin/activate"
echo "Environment $ENV_NAME wurde erstellt und Pakete sind installiert."

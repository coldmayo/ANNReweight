file=${1}
epochs=${2}
set -e
source "../.venv/bin/activate" # insert path to your virtual envirnment
python -u $file $epochs

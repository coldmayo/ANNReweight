file=${1--}
set -e
source "../.venv/bin/activate" # insert path to your virtual envirnment
python -u $file

ATTENTION_GIT="$HOME/Attention_git"
GEMINI="$ATTENTION_GIT/gemini3/Attention_evaluation"


SCRIPT="$GEMINI/main.py"
GOLD_EN="$HOME/Attention/k3/GOLD/test.en"
GOLD_DE="$HOME/Attention/k3/GOLD/test.de"
# LISTE_CHEMIN_MATRICE="$ATTENTION_GIT/marco_script/list_filename_matrice/concat/full_matrice/list_chemin_0_0.txt"
LISTE_CHEMIN_MATRICE="$ATTENTION_GIT/marco_script/list_filename_matrice/concat/cutted_matrice_corrected/list_chemin.txt"
PARCOR_PATH="$ATTENTION_GIT/marco_script/data/ParCorFull2/parcor-full/corpus"
CANMT_SYSTEM="concat"

OUTPUT_FILE="$GEMINI/Output/debug.results"


python $GEMINI/main.py \
  --corpus-source $GOLD_EN \
  --corpus-target $GOLD_DE \
  --system-data $LISTE_CHEMIN_MATRICE \
  --parcor-path $PARCOR_PATH \
  --verbose \
  --log-file "./Output/debug_attention.txt"
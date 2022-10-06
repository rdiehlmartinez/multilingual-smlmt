echo "Requesting ${1:-1} GPUs for interactive session"
sintr --qos=INTR -A BUTTERY-SL3-GPU --gres=gpu:${1:-1} -t 1:0:0  -p ampere -N 1

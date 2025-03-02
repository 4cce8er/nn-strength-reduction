#!/bin/zsh

echo "############# $1 -O3 #############"
make clean;
make OPT_FLAGS="-O3" -f $1
make OPT_FLAGS="-O3" OPT=1 -f $1
ls -lh --color=always *.elf
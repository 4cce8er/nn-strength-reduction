#!/bin/zsh

echo "############# $1 -Os #############"
make clean;
make OPT_FLAGS="-Os" -f $1
make OPT_FLAGS="-Os" OPT=1 -f $1
ls -lh --color=always *.elf
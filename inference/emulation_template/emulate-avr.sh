qemu-system-avr \
    -machine mega2560 \
    -bios $1 \
    -gdb tcp::3333 \
    -icount 0 \
    -accel tcg,one-insn-per-tb=on \
    -nographic \
    -plugin ./qemu-bin/contrib/plugins/libmemcnt.so \
    -plugin ./qemu-bin/tests/plugin/libinsn.so,inline=true \
    -d plugin \
    -D dump_$1.txt \
    -S

#    -S
    # -serial tcp::5678,server=on,wait=off \
    # -monitor stdio \
    # -S
#     -serial tcp::5678,server=on,wait=off \
    # -monitor stdio \
    # -D $1.txt \
#     -plugin /home/ale19/Programs/qemu-8.2.2/build/tests/plugin/libinsn.so \
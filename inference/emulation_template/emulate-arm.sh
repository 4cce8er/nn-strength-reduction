qemu-arm \
	-cpu cortex-m0 \
	-g 3333 \
	-one-insn-per-tb \
	-plugin ./qemu-bin/contrib/plugins/libmemcnt.so \
	-plugin ./qemu-bin/tests/plugin/libinsn.so,inline=true \
	-d plugin \
	-D dump-$1.txt \
	$1

# -plugin /home/ale19/Programs/qemu-8.2.2/build/contrib/plugins/libexeclog.so \
# -plugin /home/ale19/Programs/qemu-8.2.2/build/tests/plugin/libinsn.so \
# -plugin /home/ale19/Programs/qemu-8.2.2/build/tests/plugin/libmem.so,inline=true \
# -plugin /home/ale19/Programs/qemu-8.2.2/build/contrib/plugins/libhowvec.so
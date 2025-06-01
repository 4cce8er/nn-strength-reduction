qemu-arm \
	-cpu cortex-m0 \
	-g 3333 \
	-one-insn-per-tb \
	-plugin ./qemu-bin/contrib/plugins/libmemcnt.so \
	-plugin ./qemu-bin/contrib/plugins/libpcchangecny.so \
	-plugin ./qemu-bin/contrib/plugins/libmulcnt.so \
	-plugin ./qemu-bin/tests/plugin/libinsn.so,inline=true \
	-d plugin \
	-D dump-$1.txt \
	$1

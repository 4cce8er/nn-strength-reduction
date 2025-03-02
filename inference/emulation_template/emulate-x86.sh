qemu-x86_64 \
    -g 3333 \
	-one-insn-per-tb \
	-plugin ./qemu-bin/contrib/plugins/libmemcnt.so \
	-plugin ./qemu-bin/tests/plugin/libinsn.so,inline=true \
	-d plugin \
    -D dump-$1.txt \
	$1

# qemu-x86_64 \
#     -g 3333 \
# 	-one-insn-per-tb \
# 	-plugin ./qemu-bin/build/contrib/plugins/libexeclog.so \
# 	-d plugin \
#     -D dump-$1.txt \
# 	$1

	# -plugin ./qemu-bin/tests/plugin/libinsn.so \
	# -plugin ./qemu-bin/build/contrib/plugins/libexeclog.so \
	# -plugin ./qemu-bin/build/tests/plugin/libmem.so,inline=true,track=r,callback=true \
	# -plugin ./qemu-bin/build/contrib/plugins/libhowvec.so \
OUT="/dev/tty" # /dev/null

Os_FLAG="-Os"
O3_FLAG="-O3"
make clean > /dev/null
# x86 -Os
echo "############# x86 -Os #############"
make OPT_FLAGS=$Os_FLAG -f x86.mk > $OUT
make OPT_FLAGS=$Os_FLAG OPT=1 -f x86.mk > $OUT
ls -lh *.elf
# mv *.elf ./elf/
make clean > /dev/null
# x86 -O3
echo "############# x86 -O3 #############"
make OPT_FLAGS=$O3_FLAG -f x86.mk > $OUT
make OPT_FLAGS=$O3_FLAG OPT=1 -f x86.mk > $OUT
ls -lh *.elf
# mv *.elf ./elf/
make clean > /dev/null
arm -Os
echo "############# arm -Os #############"
make OPT_FLAGS=$Os_FLAG -f arm.mk > $OUT
make OPT_FLAGS=$Os_FLAG OPT=1 -f arm.mk > $OUT
ls -lh *.elf
mv *.elf ./elf/
make clean > /dev/null
arm -O3
echo "############# arm -O3 #############"
make OPT_FLAGS=$O3_FLAG -f arm.mk > $OUT
make OPT_FLAGS=$O3_FLAG OPT=1 -f arm.mk > $OUT
ls -lh *.elf
mv *.elf ./elf/
make clean > /dev/null
# riscv -Os
echo "############# riscv -Os #############"
make OPT_FLAGS=$Os_FLAG -f riscv.mk > $OUT
make OPT_FLAGS=$Os_FLAG OPT=1 -f riscv.mk > $OUT
ls -lh *.elf
# mv *.elf ./elf/
make clean > /dev/null
# riscv -O3
echo "############# riscv -O3 #############"
make OPT_FLAGS=$O3_FLAG -f riscv.mk > $OUT
make OPT_FLAGS=$O3_FLAG OPT=1 -f riscv.mk > $OUT
ls -lh *.elf
# mv *.elf ./elf/
make clean > /dev/null
# avr -Os
echo "############# avr -Os #############"
make OPT_FLAGS=$Os_FLAG -f avr.mk > $OUT
make OPT_FLAGS=$Os_FLAG OPT=1 -f avr.mk > $OUT
ls -lh *.elf
# mv *.elf ./elf/
make clean > /dev/null
# avr -O3
echo "############# avr -O3 #############"
make OPT_FLAGS=$O3_FLAG -f avr.mk > $OUT
make OPT_FLAGS=$O3_FLAG OPT=1 -f avr.mk > $OUT
ls -lh *.elf
# mv *.elf ./elf/
make clean > /dev/null
# msp430 -Os
echo "############# msp430 -Os #############"
make OPT_FLAGS=$Os_FLAG -f msp430.mk > $OUT
make OPT_FLAGS=$Os_FLAG OPT=1 -f msp430.mk > $OUT
ls -lh *.elf
# mv *.elf ./elf/
make clean > /dev/null
# msp430 -O3
echo "############# msp430 -O3 #############"
make OPT_FLAGS=$O3_FLAG -f msp430.mk > $OUT
make OPT_FLAGS=$O3_FLAG OPT=1 -f msp430.mk > $OUT
ls -lh *.elf
# mv *.elf ./elf/
make clean > /dev/null


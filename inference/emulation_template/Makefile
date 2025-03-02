# Shell
SHELL = /bin/sh

#
# Directories
#
SRC_DIR=./src
INC_DIR=./include
OBJ_DIR=./objs
BIN_DIR=./bins
ELF_DIR=./elfs
COMMON_DIR=./common

OPT?=0

.PHONY: help

all: help

# Use `make -B arm` to force recompilation
arm:
	make -f makefiles/arm.mk OPT=$(OPT) OPT_FLAGS="-Os"
	make -f makefiles/arm.mk OPT=$(OPT) OPT_FLAGS="-O3"
	make -f makefiles/arm.mk OPT=1 OPT_FLAGS="-Os"
	make -f makefiles/arm.mk OPT=1 OPT_FLAGS="-O3"
	ls -lh *.elf

msp430:
	make -f makefiles/msp430.mk OPT=$(OPT) OPT_FLAGS="-Os"
	make -f makefiles/msp430.mk OPT=$(OPT) OPT_FLAGS="-O3"
	make -f makefiles/msp430.mk OPT=1 OPT_FLAGS="-Os"
	make -f makefiles/msp430.mk OPT=1 OPT_FLAGS="-O3"
	ls -lh *.elf

avr:
	make -f makefiles/avr.mk OPT=$(OPT) OPT_FLAGS="-Os"
	make -f makefiles/avr.mk OPT=$(OPT) OPT_FLAGS="-O3"
	make -f makefiles/avr.mk OPT=1 OPT_FLAGS="-Os"
	make -f makefiles/avr.mk OPT=1 OPT_FLAGS="-O3"
	ls -lh *.elf

riscv:
	make -f makefiles/riscv.mk OPT=$(OPT) OPT_FLAGS="-Os"
	make -f makefiles/riscv.mk OPT=$(OPT) OPT_FLAGS="-O3"
	make -f makefiles/riscv.mk OPT=1 OPT_FLAGS="-Os"
	make -f makefiles/riscv.mk OPT=1 OPT_FLAGS="-O3"
	ls -lh *.elf


x86:
	make -f makefiles/x86.mk OPT=$(OPT) OPT_FLAGS="-Os"
	make -f makefiles/x86.mk OPT=$(OPT) OPT_FLAGS="-O3"
	make -f makefiles/x86.mk OPT=1 OPT_FLAGS="-Os"
	make -f makefiles/x86.mk OPT=1 OPT_FLAGS="-O3"
	ls -lh *.elf

help:
	@echo "Available architectures: $(VALID_ARCHS)"
	@echo "Usage: make ARCH=<architecture> [target]"
	@echo "Available targets: all, clean, dump, size, help, compiler_macros"

clean:
	rm -rf $(OBJ_DIR)/* $(BIN_DIR)/* $(ELF_DIR)/* *.out *.map *.s *.i *.o *.elf


size:

compiler_macros:

dump:

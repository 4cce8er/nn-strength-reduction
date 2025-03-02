ifdef OPT
BIN = msp430.opt.elf
else
BIN = msp430.elf
endif

include makefiles/common.mk

CC=msp430-elf-gcc-9.3.1
READELF=msp430-elf-readelf
OBJDUMP=msp430-elf-objdump
OBJCOPY=msp430-elf-objcopy
LD=msp430-elf-ld


MCU=msp430f5529
MCU_SERIES=f5series
MSPGCC_INC=/home/ale19/ti/msp430-gcc/msp430-elf/include
MSP430_INC=/home/ale19/ti/msp430-gcc/include

LDFLAGS += -Wl,-Map,$(BIN).map
LDFLAGS += -Wl,--gc-sections -L$(MSP430_INC)
LDFLAGS += -Wl,--start-group -lgcc -lc
LDFLAGS += -Wl,--end-group


CFLAGS += -DMSP430=1
CFLAGS += -mmcu=$(MCU) -mhwmult=$(MCU_SERIES) -mlarge -mcode-region=none -mdata-region=lower
CFLAGS += -I$(MSPGCC_INC) -I$(INC_DIR) -I$(MSP430_INC)

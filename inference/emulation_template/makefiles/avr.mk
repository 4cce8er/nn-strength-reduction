ifdef OPT
BIN = avr.opt.elf
else
BIN = avr.elf
endif

include makefiles/common.mk

CC=avr-gcc
OBJDUMP=avr-objdump
OBJCOPY=avr-objcopy
READELF=avr-readelf
ELFSIZE=avr-size


MCU=atmega2560
F_CPU=16000000UL

CFLAGS += -DAVR=1 -mmcu=$(MCU) -DF_CPU=$(F_CPU)

size:
	$(ELFSIZE) -C --mcu=$(MCU) $(BIN)
	$(READELF) -S $(BIN)
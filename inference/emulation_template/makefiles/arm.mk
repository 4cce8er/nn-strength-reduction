#
# MCU CONFIGS
#
MCU=cortex-m0

ifdef OPT
BIN = arm$(MCU).opt.elf
else
BIN = arm$(MCU).elf
endif

include makefiles/common.mk

CC=arm-none-eabi-gcc
AS=arm-none-eabi-as
LD=arm-none-eabi-ld
OBJDUMP=arm-none-eabi-objdump
OBJCOPY=arm-none-eabi-objcopy

ARM_DIR=./arm
LDFLAGS+=-T $(ARM_DIR)/linker.ld -specs=nosys.specs
CFLAGS += -nostartfiles -mcpu=$(MCU) -mthumb -ffreestanding

ASM_SRC := $(ARM_DIR)/startup.s
ASM_OBJ := $(OBJ_DIR)/startup.o

all: prebuild $(BIN) dump

$(ASM_OBJ): $(ASM_SRC)
	$(AS) $(ASM_SRC) -mcpu=$(MCU) --warn --fatal-warnings -o $(ASM_OBJ)

$(BIN): $(ASM_OBJ)
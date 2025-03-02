ENTRY(_start)
MEMORY
{
    ram : ORIGIN = 0x00010000, LENGTH = 0x2000 /* 8KB RAM */
    rom : ORIGIN = 0x00012000, LENGTH = 0xBB80 /* 0x20000/* 48KB ROM */
}
/*0x40000*/
SECTIONS
{
    .text : { *(.text*) } > rom
    .rodata : { *(.rodata*) } > rom
    .bss : { *(.bss*) } > ram
    .data : { *(.data*) } > ram
}
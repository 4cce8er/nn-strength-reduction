# Load the ELF file to read the symbols
file avr.elf
# file elfs/avr.elf-Os.nodebug.elf
# Connect to the remote server
target remote :3333
# add a breakpoint on main function
# break main

# to print the result
break main.c:80
# to check how many instructions
break __stop_program 

# break model.c:165
# Continue the execution until main
b main.c:42
b main.c:49
b main.c:52
b main.c:63
b main.c:78
# p/d result
c

# mon q
# quit
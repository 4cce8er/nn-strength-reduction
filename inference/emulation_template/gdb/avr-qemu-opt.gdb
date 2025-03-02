# Load the ELF file to read the symbols
file avr.opt.elf
# Connect to the remote server
target remote :3333
# add a breakpoint on main function
# break main
# to print the result
break main.c:70
# to check how many instructions
break __stop_program 

# break model.c:165
# Continue the execution until main
c
p/d result
c
mon q
quit
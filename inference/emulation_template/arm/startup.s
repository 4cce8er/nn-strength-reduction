.thumb
.thumb_func
.global _start
_start:
    @mov r0,=0x10000
    @mov sp,r0
    bl main
    b __stop_program

__stop_program:
    b .

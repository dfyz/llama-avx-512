section .text
global _start

num:
    ; 1e-18 produces a normal result
    ; 1e-19 produces a subnormal result
    ; 1e-23 produces zero
    dd 1e-18

_start:
    mov rdi, 1_000_000_000
    mov rsi, [rel num]
    vpbroadcastd zmm1, esi
    vpbroadcastd zmm0, esi

loop:
    vmulps zmm2, zmm1, zmm0
    dec rdi
    jnz loop

    xor rdi, rdi
    mov rax, 0x3c
    syscall

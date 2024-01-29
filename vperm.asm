section .text
global _start

vperm_byte:
    vmovupd zmm13, [rel byte_perm]
    mov rdi, 1_000_000_000

    vmovupd zmm1, [rel rnd_1]
    vmovupd zmm3, [rel rnd_3]
    vmovupd zmm5, [rel rnd_5]
    vmovupd zmm7, [rel rnd_7]
    vmovupd zmm9, [rel rnd_9]
    vmovupd zmm11, [rel rnd_11]

    mov     eax, 0xaaaaaaaa
    kmovd   k1, eax

loop_byte:
    vpermb zmm0, zmm13, zmm1
    vpsraw zmm0{k1}, zmm0, 0x4
    vpermb zmm2, zmm13, zmm3
    vpsraw zmm2{k1}, zmm2, 0x4
    vpermb zmm4, zmm13, zmm5
    vpsraw zmm4{k1}, zmm4, 0x4
    vpermb zmm6, zmm13, zmm7
    vpsraw zmm6{k1}, zmm6, 0x4
    vpermb zmm8, zmm13, zmm9
    vpsraw zmm8{k1}, zmm8, 0x4
    vpermb zmm10, zmm13, zmm11
    vpsraw zmm10{k1}, zmm10, 0x4
    dec rdi
    jnz loop_byte

    ret

vperm_word:
    vmovupd zmm13, [rel word_perm]
    mov rdi, 1_000_000_000

    vmovupd zmm1, [rel rnd_1]
    vmovupd zmm3, [rel rnd_3]
    vmovupd zmm5, [rel rnd_5]
    vmovupd zmm7, [rel rnd_7]
    vmovupd zmm9, [rel rnd_9]
    vmovupd zmm11, [rel rnd_11]

    mov     eax, 0xaaaaaaaa
    kmovd   k1, eax

loop_word:
    vpermw zmm0, zmm13, zmm1
    vpsraw zmm0{k1}, zmm0, 0x4
    vpermw zmm2, zmm13, zmm3
    vpsraw zmm2{k1}, zmm2, 0x4
    vpermw zmm4, zmm13, zmm5
    vpsraw zmm4{k1}, zmm4, 0x4
    vpermw zmm6, zmm13, zmm7
    vpsraw zmm6{k1}, zmm6, 0x4
    vpermw zmm8, zmm13, zmm9
    vpsraw zmm8{k1}, zmm8, 0x4
    vpermw zmm10, zmm13, zmm11
    vpsraw zmm10{k1}, zmm10, 0x4
    dec rdi
    jnz loop_word

    ret

_start:
    ; call vperm_byte
    call vperm_word

    xor rdi, rdi
    mov eax, 0x3c
    syscall

byte_perm:
db 0x04, 0x05, 0x04, 0x05, 0x06, 0x07, 0x06, 0x07, 0x08, 0x09, 0x08, 0x09, 0x0a, 0x0b, 0x0a, 0x0b, 0x0c, 0x0d, 0x0c, 0x0d, 0x0e, 0x0f, 0x0e, 0x0f, 0x10, 0x11, 0x10, 0x11, 0x12, 0x13, 0x12, 0x13, 0x18, 0x19, 0x18, 0x19, 0x1a, 0x1b, 0x1a, 0x1b, 0x1c, 0x1d, 0x1c, 0x1d, 0x1e, 0x1f, 0x1e, 0x1f, 0x20, 0x21, 0x20, 0x21, 0x22, 0x23, 0x22, 0x23, 0x24, 0x25, 0x24, 0x25, 0x26, 0x27, 0x26, 0x27

word_perm:
dw 0x02, 0x02, 0x03, 0x03, 0x04, 0x04, 0x05, 0x05, 0x06, 0x06, 0x07, 0x07, 0x08, 0x08, 0x09, 0x09, 0x0c, 0x0c, 0x0d, 0x0d, 0x0e, 0x0e, 0x0f, 0x0f, 0x10, 0x10, 0x11, 0x11, 0x12, 0x12, 0x13, 0x13

rnd_1:
db 58, 13, 51, 158, 62, 141, 76, 2, 22, 160, 91, 195, 59, 183, 175, 9, 23, 30, 34, 88, 138, 67, 202, 100, 93, 160, 93, 30, 167, 53, 144, 195

rnd_3:
db 189, 26, 225, 214, 32, 28, 213, 246, 232, 28, 135, 218, 86, 133, 98, 176, 77, 108, 71, 204, 41, 69, 128, 120, 55, 15, 62, 253, 196, 209, 2, 63

rnd_5:
db 204, 71, 5, 164, 57, 40, 48, 225, 171, 4, 154, 54, 75, 235, 198, 175, 73, 199, 14, 239, 72, 219, 45, 49, 77, 135, 108, 97, 231, 244, 158, 161

rnd_7:
db 103, 94, 64, 76, 137, 136, 79, 74, 68, 162, 143, 230, 113, 45, 76, 245, 49, 87, 28, 251, 108, 240, 150, 141, 186, 235, 70, 188, 59, 110, 117, 126

rnd_9:
db 47, 140, 232, 103, 160, 213, 161, 218, 126, 142, 229, 236, 21, 249, 131, 173, 199, 214, 118, 167, 161, 145, 122, 157, 116, 193, 116, 99, 17, 20, 94, 119

rnd_11:
db 26, 137, 5, 144, 222, 221, 144, 51, 59, 132, 19, 132, 231, 82, 60, 54, 228, 93, 179, 161, 244, 52, 10, 231, 109, 17, 208, 178, 176, 158, 112, 131

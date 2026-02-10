/*
   Copyright 2026 Huawei Technologies Co., Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
.macro TBL12
        tbl v14.16B, { v8.16B - v9.16B }, v14.16B
        tbl v15.16B, { v8.16B - v9.16B }, v15.16B
        tbl v16.16B, { v8.16B - v9.16B }, v16.16B
        tbl v17.16B, { v8.16B - v9.16B }, v17.16B
        tbl v18.16B, { v8.16B - v9.16B }, v18.16B
        tbl v19.16B, { v8.16B - v9.16B }, v19.16B
        tbl v20.16B, { v8.16B - v9.16B }, v20.16B
        tbl v21.16B, { v8.16B - v9.16B }, v21.16B
.endm
.macro TBL20
        tbl v22.16B, { v10.16B - v11.16B }, v22.16B
        tbl v23.16B, { v10.16B - v11.16B }, v23.16B
        tbl v24.16B, { v10.16B - v11.16B }, v24.16B
        tbl v25.16B, { v10.16B - v11.16B }, v25.16B
        tbl v26.16B, { v10.16B - v11.16B }, v26.16B
        tbl v27.16B, { v10.16B - v11.16B }, v27.16B
        tbl v28.16B, { v10.16B - v11.16B }, v28.16B
        tbl v29.16B, { v10.16B - v11.16B }, v29.16B
.endm
.macro ushr_all
    ushr    v18.16b, v14.16b, 4
    ushr    v19.16b, v15.16b, 4
    ushr    v20.16b, v16.16b, 4
    ushr    v21.16b, v17.16b, 4
    ushr    v26.16b, v22.16b, 4
    ushr    v27.16b, v23.16b, 4
    ushr    v28.16b, v24.16b, 4
    ushr    v29.16b, v25.16b, 4
.endm
.macro sli_and12
    sli    v14.16b, v31.16b, 4
    sli    v15.16b, v31.16b, 4
    sli    v16.16b, v31.16b, 4
    sli    v17.16b, v31.16b, 4
    add    v18.16b, v18.16b, v30.16B
    add    v19.16b, v19.16b, v30.16B
    add    v20.16b, v20.16b, v30.16B
    add    v21.16b, v21.16b, v30.16b
.endm
.macro sli_and20
    sli    v22.16b, v31.16b, 4
    sli    v23.16b, v31.16b, 4
    sli    v24.16b, v31.16b, 4
    sli    v25.16b, v31.16b, 4
    add    v26.16b, v26.16b, v30.16B
    add    v27.16b, v27.16b, v30.16B
    add    v28.16b, v28.16b, v30.16B 
    add    v29.16b, v29.16b, v30.16B
.endm
.macro uadalp12
    uadalp    v0.8h, v14.16B
    uadalp    v1.8h, v15.16B
    uadalp    v2.8h, v16.16B
    uadalp    v3.8h, v17.16B
.endm
.macro uadalp16
    uadalp    v4.8h, v18.16B 
    uadalp    v5.8h, v19.16B
    uadalp    v6.8h, v20.16B
    uadalp    v7.8h, v21.16B
.endm
.macro uadalp20
    uadalp    v0.8h, v22.16B
    uadalp    v1.8h, v23.16B
    uadalp    v2.8h, v24.16B
    uadalp    v3.8h, v25.16B
.endm
.macro uadalp24
    uadalp    v4.8h, v26.16B
    uadalp    v5.8h, v27.16B
    uadalp    v6.8h, v28.16B
    uadalp    v7.8h, v29.16B
.endm
    .align    2
    .p2align 4,,11
    .global    krl_table_lookup_fast_scan_bs64_asm
    .type    krl_table_lookup_fast_scan_bs64_asm, %function
krl_table_lookup_fast_scan_bs64_asm:
.LFB6954:
    .cfi_startproc
    str     x29, [sp, -64]!
    .cfi_def_cfa_offset 64
    .cfi_offset 29, -64
    .cfi_offset 30, -56
    mov     x29, sp    
    .cfi_def_cfa_register 29
    stp     d8, d9, [sp, 16]
    stp     d10, d11, [sp, 32]
    .cfi_offset 72, -32
    .cfi_offset 73, -24
    .cfi_offset 74, -16
    .cfi_offset 75, -8
    movi    v31.8h, 0x1, lsl 8
    movi    v30.8h, 0x10, lsl 8
    cmp w0, 4
    ble .L4    
    add    w6, w0, #1
    lsr    w7, w6, #1
    and    w8, w6, #2
    add    x0, x2, x7, lsl 5
    cbnz   w8, .L1
.L2:
    ldp    q14, q15, [x1]
    ldp    q16, q17, [x1, 32]
    ldp    q22, q23, [x1, 64]
    ldp    q24, q25, [x1, 96]
    ldp    q8, q9, [x2]
    ldp    q10, q11, [x2, 32]
    ushr_all
    sli_and12
    TBL12
    ldp     q8, q9, [x2, 64]
    uaddlp  v0.8h, v14.16B
    uaddlp  v1.8h, v15.16B
    uaddlp  v2.8h, v16.16B
    uaddlp  v3.8h, v17.16B
    ldp     q14, q15, [x1, 128]
    ldp     q16, q17, [x1, 160]
    uaddlp  v4.8h, v18.16B
    uaddlp  v5.8h, v19.16B
    uaddlp  v6.8h, v20.16B
    uaddlp  v7.8h, v21.16B
    sli_and20
    TBL20
    ldp    q10, q11, [x2, 96]
    add    x2, x2, 128
    uadalp20
    ldp    q22, q23, [x1, 192]
    ldp    q24, q25, [x1, 224]
    add    x1, x1, 256
    uadalp24
    cmp x0, x2
    beq .L40
    .p2align 3,,7
.L20:
    prfm pldl1keep, [x2, #192]
    prfm pldl1strm, [x1, #384]
    prfm pldl1strm, [x1, #448]
    ushr_all
    sli_and12
    TBL12
    ld1    { v8.16B - v9.16B }, [x2], 32
    uadalp12
    ldp    q14, q15, [x1]
    ldp    q16, q17, [x1, 32]
    uadalp16
    sli_and20 
    TBL20
    ld1    { v10.16B - v11.16B }, [x2], 32
    uadalp20
    ldp    q22, q23, [x1, 64]
    ldp    q24, q25, [x1, 96]
    add    x1, x1, 128
    uadalp24
    cmp    x2, x0
    bne .L20
.L40:
    ushr_all
    sli_and12
    TBL12
    uadalp12
    uadalp16
    sli_and20
    TBL20
    uadalp20
    uadalp24
.L50:
    stp    q0, q4, [x3]
    stp    q1, q5, [x3, 32]
    stp    q2, q6, [x3, 64]
    stp    q3, q7, [x3, 96]
    ldp    d10, d11, [sp, 32]
    ldp    d8, d9, [sp, 16]
    mov    sp, x29
    .cfi_def_cfa_register 31
    ldr    x29, [sp], 64
    .cfi_restore 30
    .cfi_restore 29
    .cfi_def_cfa_offset 0
    ret
    .p2align 2,,3
.L1:
    ldp     q22, q23, [x1]
    ldp     q24, q25, [x1, 32]
    ldp     q14, q15, [x1, 64]
    ldp     q16, q17, [x1, 96]
    ldp     q10, q11, [x2]
    ldp     q8, q9, [x2, 32]
    ushr    v26.16b, v22.16b, 4
    ushr    v27.16b, v23.16b, 4
    ushr    v28.16b, v24.16b, 4
    ushr    v29.16b, v25.16b, 4
    sli_and20
    TBL20
    ldp     q10, q11, [x2, 64]
    add     x2, x2, 96
    uaddlp  v0.8h, v22.16B
    uaddlp  v1.8h, v23.16B
    uaddlp  v2.8h, v24.16B
    uaddlp  v3.8h, v25.16B
    ldp     q22, q23, [x1, 128]
    ldp     q24, q25, [x1, 160]
    add     x1, x1, 192
    uaddlp  v4.8h, v26.16B
    uaddlp  v5.8h, v27.16B
    uaddlp  v6.8h, v28.16B
    uaddlp  v7.8h, v29.16B
    cmp x0, x2
    beq .L40
    b .L20
.L4:
    cmp w0, 2
    ble .L3
    ldp     q14, q15, [x1]
    ldp     q16, q17, [x1, 32]
    ldp     q22, q23, [x1, 64]
    ldp     q24, q25, [x1, 96]
    ldp     q8, q9, [x2]
    ldp     q10, q11, [x2, 32]
    ushr_all
    sli_and12
    TBL12
    uaddlp  v0.8h, v14.16B
    uaddlp  v1.8h, v15.16B
    uaddlp  v2.8h, v16.16B
    uaddlp  v3.8h, v17.16B
    uaddlp  v4.8h, v18.16B
    uaddlp  v5.8h, v19.16B
    uaddlp  v6.8h, v20.16B
    uaddlp  v7.8h, v21.16B
    sli_and20
    TBL20
    uadalp20
    uadalp24
    b .L50
.L3:
    ldp     q14, q15, [x1]
    ldp     q16, q17, [x1, 32]
    ldp     q8, q9, [x2]
    ushr    v18.16b, v14.16b, 4
    ushr    v19.16b, v15.16b, 4
    ushr    v20.16b, v16.16b, 4
    ushr    v21.16b, v17.16b, 4
    sli_and12
    TBL12
    uaddlp  v0.8h, v14.16B
    uaddlp  v1.8h, v15.16B
    uaddlp  v2.8h, v16.16B
    uaddlp  v3.8h, v17.16B
    uaddlp  v4.8h, v18.16B
    uaddlp  v5.8h, v19.16B
    uaddlp  v6.8h, v20.16B
    uaddlp  v7.8h, v21.16B
    b .L50
    .cfi_endproc
.LFE6954:
    .size    krl_table_lookup_fast_scan_bs64_asm, .-krl_table_lookup_fast_scan_bs64_asm
    .align    2
    .p2align 4,,11

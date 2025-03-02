#ifndef MACROS_H
#define MACROS_H

#include <stdint.h>
#include <stddef.h>

#define BUF_SIZE(buf) sizeof(buf) / sizeof(buf[0])

#if defined(__x86_64) || defined(__x86_64__)
// Macro to define a generic print function for arrays
#define PRINT_ARRAY(arr, size, formatSpecifier) \
    do {                                        \
        printf("{ ");                           \
        for (int i = 0; i < size; i++) {        \
            printf(formatSpecifier, arr[i]);    \
            if (i != size - 1) {                \
                printf(", ");                   \
            }                                   \
        }                                       \
        printf(" }\n");                         \
    } while (0)
// Macro to define a generic print function for matrixes
#define PRINT_MATRIX(arr, row, col, formatSpecifier) \
    do {                                             \
        printf("{ ");                                \
        for (int i = 0; i < row; i++) {              \
            for (int j = 0; j < col; j++) {          \
                printf(formatSpecifier, arr[i][j]);  \
                if (i != col - 1) {                  \
                    printf(", ");                    \
                }                                    \
            }                                        \
        }                                            \
        printf(" }\n");                              \
    } while (0)

#else
#define PRINT_ARRAY(arr, size, formatSpecifier) ((void)0)
#define PRINT_MATRIX(arr, row, col, formatSpecifier) ((void)0)
#endif

static inline int32_t multiply_by_quantize_mul(int64_t acc, int32_t q_mantissa, int32_t exp){
    const int32_t reduced_mantissa = q_mantissa < 0x7FFF0000 ? ((q_mantissa + (1 << 15)) >> 16) : 0x7FFF;
    const int64_t total_shifts = 15 - exp;
    const int64_t round = (int64_t)(1) << (total_shifts - 1);
    acc = acc * (int64_t)reduced_mantissa;
    acc = acc + round;
    int32_t result = acc >> total_shifts;
    return result;
}


#endif
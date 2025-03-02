import numpy as np

def to_np_dtype(type_str: str):
    match type_str:
        case "int8":
            return np.int8
        case "int32":
            return np.int32
        case _:
            return np.float64

def multiply_by_quantize_mul(x: np.int64, q_mantissa: np.int32, exponent: np.int32):
    # q_mantissa, exponent = quantize_multiplier(scale)
    # print(q_mantissa, exponent)
    reduced_mantissa = (
        ((q_mantissa + (1 << 15)) >> 16) if q_mantissa < 0x7FFF0000 else 0x7FFF
    )
    total_shifts = 15 - exponent
    x = np.int64(x) * np.int64(reduced_mantissa)
    x = x + (1 << (total_shifts - 1))
    result = np.right_shift(x, total_shifts)
    return result
#include "multiply.h"

// Multiply by -128 using the fewest operations
/*inline*/ int16_t multiply_n128(const int8_t n)
{
	int t1, t2;
	t1 = 0 - n;
	t2 = t1 << 7;
	return t2;
}
// Multiply by -127 using the fewest operations
/*inline*/ int16_t multiply_n127(const int8_t n)
{
	int t1, t2;
	t1 = n << 7;
	t2 = n - t1;
	return t2;
}
// Multiply by -126 using the fewest operations
/*inline*/ int16_t multiply_n126(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 6;
	t2 = n - t1;
	t3 = t2 << 1;
	return t3;
}
// Multiply by -125 using the fewest operations
/*inline*/ int16_t multiply_n125(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 6;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	return t4;
}
// Multiply by -124 using the fewest operations
/*inline*/ int16_t multiply_n124(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 5;
	t2 = n - t1;
	t3 = t2 << 2;
	return t3;
}
// Multiply by -123 using the fewest operations
/*inline*/ int16_t multiply_n123(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 5;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = n - t3;
	return t4;
}
// Multiply by -122 using the fewest operations
/*inline*/ int16_t multiply_n122(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 5;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -121 using the fewest operations
/*inline*/ int16_t multiply_n121(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = n - t1;
	t3 = t2 << 3;
	t4 = t3 - n;
	return t4;
}
// Multiply by -120 using the fewest operations
/*inline*/ int16_t multiply_n120(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 4;
	t2 = n - t1;
	t3 = t2 << 3;
	return t3;
}
// Multiply by -119 using the fewest operations
/*inline*/ int16_t multiply_n119(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = n - t1;
	t3 = t2 << 4;
	t4 = t3 + t2;
	return t4;
}
// Multiply by -118 using the fewest operations
/*inline*/ int16_t multiply_n118(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -117 using the fewest operations
/*inline*/ int16_t multiply_n117(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 3;
	t6 = t5 + t4;
	return t6;
}
// Multiply by -116 using the fewest operations
/*inline*/ int16_t multiply_n116(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -115 using the fewest operations
/*inline*/ int16_t multiply_n115(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = n - t3;
	t5 = t4 << 2;
	t6 = t5 + t4;
	return t6;
}
// Multiply by -114 using the fewest operations
/*inline*/ int16_t multiply_n114(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = n - t1;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -113 using the fewest operations
/*inline*/ int16_t multiply_n113(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = n - t1;
	t3 = t2 << 4;
	t4 = t3 - n;
	return t4;
}
// Multiply by -112 using the fewest operations
/*inline*/ int16_t multiply_n112(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = n - t1;
	t3 = t2 << 4;
	return t3;
}
// Multiply by -111 using the fewest operations
/*inline*/ int16_t multiply_n111(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 4;
	t4 = n - t3;
	return t4;
}
// Multiply by -110 using the fewest operations
/*inline*/ int16_t multiply_n110(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -109 using the fewest operations
/*inline*/ int16_t multiply_n109(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 1;
	t6 = n - t5;
	return t6;
}
// Multiply by -108 using the fewest operations
/*inline*/ int16_t multiply_n108(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 3;
	t4 = t3 + t2;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -107 using the fewest operations
/*inline*/ int16_t multiply_n107(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 2;
	t6 = n - t5;
	return t6;
}
// Multiply by -106 using the fewest operations
/*inline*/ int16_t multiply_n106(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6, t7;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	t6 = n - t5;
	t7 = t6 << 1;
	return t7;
}
// Multiply by -105 using the fewest operations
/*inline*/ int16_t multiply_n105(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = t2 - t3;
	return t4;
}
// Multiply by -104 using the fewest operations
/*inline*/ int16_t multiply_n104(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 3;
	return t5;
}
// Multiply by -103 using the fewest operations
/*inline*/ int16_t multiply_n103(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 3;
	t6 = n - t5;
	return t6;
}
// Multiply by -102 using the fewest operations
/*inline*/ int16_t multiply_n102(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 4;
	t4 = t3 + t2;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -101 using the fewest operations
/*inline*/ int16_t multiply_n101(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	t6 = n - t5;
	return t6;
}
// Multiply by -100 using the fewest operations
/*inline*/ int16_t multiply_n100(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -99 using the fewest operations
/*inline*/ int16_t multiply_n99(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 5;
	t4 = t3 + t2;
	return t4;
}
// Multiply by -98 using the fewest operations
/*inline*/ int16_t multiply_n98(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 4;
	t4 = t3 - n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -97 using the fewest operations
/*inline*/ int16_t multiply_n97(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 5;
	t4 = t3 - n;
	return t4;
}
// Multiply by -96 using the fewest operations
/*inline*/ int16_t multiply_n96(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 5;
	return t3;
}
// Multiply by -95 using the fewest operations
/*inline*/ int16_t multiply_n95(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 5;
	t4 = n - t3;
	return t4;
}
// Multiply by -94 using the fewest operations
/*inline*/ int16_t multiply_n94(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 4;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -93 using the fewest operations
/*inline*/ int16_t multiply_n93(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 5;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t2 - t3;
	return t4;
}
// Multiply by -92 using the fewest operations
/*inline*/ int16_t multiply_n92(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = n - t3;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -91 using the fewest operations
/*inline*/ int16_t multiply_n91(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 2;
	t6 = n - t5;
	return t6;
}
// Multiply by -90 using the fewest operations
/*inline*/ int16_t multiply_n90(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t2 - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -89 using the fewest operations
/*inline*/ int16_t multiply_n89(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	t6 = n - t5;
	return t6;
}
// Multiply by -88 using the fewest operations
/*inline*/ int16_t multiply_n88(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = n - t3;
	t5 = t4 << 3;
	return t5;
}
// Multiply by -87 using the fewest operations
/*inline*/ int16_t multiply_n87(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 3;
	t6 = n - t5;
	return t6;
}
// Multiply by -86 using the fewest operations
/*inline*/ int16_t multiply_n86(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6, t7;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 2;
	t6 = n - t5;
	t7 = t6 << 1;
	return t7;
}
// Multiply by -85 using the fewest operations
/*inline*/ int16_t multiply_n85(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 2;
	t6 = t5 + t4;
	return t6;
}
// Multiply by -84 using the fewest operations
/*inline*/ int16_t multiply_n84(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t2 - t3;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -83 using the fewest operations
/*inline*/ int16_t multiply_n83(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 2;
	t6 = n - t5;
	return t6;
}
// Multiply by -82 using the fewest operations
/*inline*/ int16_t multiply_n82(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6, t7;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	t6 = n - t5;
	t7 = t6 << 1;
	return t7;
}
// Multiply by -81 using the fewest operations
/*inline*/ int16_t multiply_n81(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 3;
	t6 = t5 + t4;
	return t6;
}
// Multiply by -80 using the fewest operations
/*inline*/ int16_t multiply_n80(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 4;
	return t5;
}
// Multiply by -79 using the fewest operations
/*inline*/ int16_t multiply_n79(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 4;
	t4 = n - t3;
	return t4;
}
// Multiply by -78 using the fewest operations
/*inline*/ int16_t multiply_n78(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -77 using the fewest operations
/*inline*/ int16_t multiply_n77(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 1;
	t6 = n - t5;
	return t6;
}
// Multiply by -76 using the fewest operations
/*inline*/ int16_t multiply_n76(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = n - t3;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -75 using the fewest operations
/*inline*/ int16_t multiply_n75(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = n - t1;
	t3 = t2 << 2;
	t4 = t3 + t2;
	return t4;
}
// Multiply by -74 using the fewest operations
/*inline*/ int16_t multiply_n74(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6, t7;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 1;
	t6 = n - t5;
	t7 = t6 << 1;
	return t7;
}
// Multiply by -73 using the fewest operations
/*inline*/ int16_t multiply_n73(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 1;
	t6 = n - t5;
	return t6;
}
// Multiply by -72 using the fewest operations
/*inline*/ int16_t multiply_n72(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 3;
	return t5;
}
// Multiply by -71 using the fewest operations
/*inline*/ int16_t multiply_n71(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = n - t3;
	return t4;
}
// Multiply by -70 using the fewest operations
/*inline*/ int16_t multiply_n70(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = n - t1;
	t3 = t2 << 2;
	t4 = t3 + t2;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -69 using the fewest operations
/*inline*/ int16_t multiply_n69(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + t2;
	t5 = t4 << 1;
	t6 = n - t5;
	return t6;
}
// Multiply by -68 using the fewest operations
/*inline*/ int16_t multiply_n68(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -67 using the fewest operations
/*inline*/ int16_t multiply_n67(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = n - t3;
	return t4;
}
// Multiply by -66 using the fewest operations
/*inline*/ int16_t multiply_n66(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -65 using the fewest operations
/*inline*/ int16_t multiply_n65(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 5;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	return t4;
}
// Multiply by -64 using the fewest operations
/*inline*/ int16_t multiply_n64(const int8_t n)
{
	int t1, t2;
	t1 = 0 - n;
	t2 = t1 << 6;
	return t2;
}
// Multiply by -63 using the fewest operations
/*inline*/ int16_t multiply_n63(const int8_t n)
{
	int t1, t2;
	t1 = n << 6;
	t2 = n - t1;
	return t2;
}
// Multiply by -62 using the fewest operations
/*inline*/ int16_t multiply_n62(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 5;
	t2 = n - t1;
	t3 = t2 << 1;
	return t3;
}
// Multiply by -61 using the fewest operations
/*inline*/ int16_t multiply_n61(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 5;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	return t4;
}
// Multiply by -60 using the fewest operations
/*inline*/ int16_t multiply_n60(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 4;
	t2 = n - t1;
	t3 = t2 << 2;
	return t3;
}
// Multiply by -59 using the fewest operations
/*inline*/ int16_t multiply_n59(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = n - t3;
	return t4;
}
// Multiply by -58 using the fewest operations
/*inline*/ int16_t multiply_n58(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -57 using the fewest operations
/*inline*/ int16_t multiply_n57(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = n - t1;
	t3 = t2 << 3;
	t4 = t3 - n;
	return t4;
}
// Multiply by -56 using the fewest operations
/*inline*/ int16_t multiply_n56(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = n - t1;
	t3 = t2 << 3;
	return t3;
}
// Multiply by -55 using the fewest operations
/*inline*/ int16_t multiply_n55(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = n - t3;
	return t4;
}
// Multiply by -54 using the fewest operations
/*inline*/ int16_t multiply_n54(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 3;
	t4 = t3 + t2;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -53 using the fewest operations
/*inline*/ int16_t multiply_n53(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	t6 = n - t5;
	return t6;
}
// Multiply by -52 using the fewest operations
/*inline*/ int16_t multiply_n52(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -51 using the fewest operations
/*inline*/ int16_t multiply_n51(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 4;
	t4 = t3 + t2;
	return t4;
}
// Multiply by -50 using the fewest operations
/*inline*/ int16_t multiply_n50(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -49 using the fewest operations
/*inline*/ int16_t multiply_n49(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 4;
	t4 = t3 - n;
	return t4;
}
// Multiply by -48 using the fewest operations
/*inline*/ int16_t multiply_n48(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 4;
	return t3;
}
// Multiply by -47 using the fewest operations
/*inline*/ int16_t multiply_n47(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 4;
	t4 = n - t3;
	return t4;
}
// Multiply by -46 using the fewest operations
/*inline*/ int16_t multiply_n46(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -45 using the fewest operations
/*inline*/ int16_t multiply_n45(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t2 - t3;
	return t4;
}
// Multiply by -44 using the fewest operations
/*inline*/ int16_t multiply_n44(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = n - t3;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -43 using the fewest operations
/*inline*/ int16_t multiply_n43(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 2;
	t6 = n - t5;
	return t6;
}
// Multiply by -42 using the fewest operations
/*inline*/ int16_t multiply_n42(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t2 - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -41 using the fewest operations
/*inline*/ int16_t multiply_n41(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	t6 = n - t5;
	return t6;
}
// Multiply by -40 using the fewest operations
/*inline*/ int16_t multiply_n40(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 3;
	return t5;
}
// Multiply by -39 using the fewest operations
/*inline*/ int16_t multiply_n39(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = n - t3;
	return t4;
}
// Multiply by -38 using the fewest operations
/*inline*/ int16_t multiply_n38(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -37 using the fewest operations
/*inline*/ int16_t multiply_n37(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 1;
	t6 = n - t5;
	return t6;
}
// Multiply by -36 using the fewest operations
/*inline*/ int16_t multiply_n36(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -35 using the fewest operations
/*inline*/ int16_t multiply_n35(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = n - t1;
	t3 = t2 << 2;
	t4 = t3 + t2;
	return t4;
}
// Multiply by -34 using the fewest operations
/*inline*/ int16_t multiply_n34(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -33 using the fewest operations
/*inline*/ int16_t multiply_n33(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	return t4;
}
// Multiply by -32 using the fewest operations
/*inline*/ int16_t multiply_n32(const int8_t n)
{
	int t1, t2;
	t1 = 0 - n;
	t2 = t1 << 5;
	return t2;
}
// Multiply by -31 using the fewest operations
/*inline*/ int16_t multiply_n31(const int8_t n)
{
	int t1, t2;
	t1 = n << 5;
	t2 = n - t1;
	return t2;
}
// Multiply by -30 using the fewest operations
/*inline*/ int16_t multiply_n30(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 4;
	t2 = n - t1;
	t3 = t2 << 1;
	return t3;
}
// Multiply by -29 using the fewest operations
/*inline*/ int16_t multiply_n29(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	return t4;
}
// Multiply by -28 using the fewest operations
/*inline*/ int16_t multiply_n28(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = n - t1;
	t3 = t2 << 2;
	return t3;
}
// Multiply by -27 using the fewest operations
/*inline*/ int16_t multiply_n27(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 3;
	t4 = t3 + t2;
	return t4;
}
// Multiply by -26 using the fewest operations
/*inline*/ int16_t multiply_n26(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -25 using the fewest operations
/*inline*/ int16_t multiply_n25(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 3;
	t4 = t3 - n;
	return t4;
}
// Multiply by -24 using the fewest operations
/*inline*/ int16_t multiply_n24(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 3;
	return t3;
}
// Multiply by -23 using the fewest operations
/*inline*/ int16_t multiply_n23(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = n - t3;
	return t4;
}
// Multiply by -22 using the fewest operations
/*inline*/ int16_t multiply_n22(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -21 using the fewest operations
/*inline*/ int16_t multiply_n21(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t2 - t3;
	return t4;
}
// Multiply by -20 using the fewest operations
/*inline*/ int16_t multiply_n20(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 2;
	return t5;
}
// Multiply by -19 using the fewest operations
/*inline*/ int16_t multiply_n19(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = n - t3;
	return t4;
}
// Multiply by -18 using the fewest operations
/*inline*/ int16_t multiply_n18(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -17 using the fewest operations
/*inline*/ int16_t multiply_n17(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	return t4;
}
// Multiply by -16 using the fewest operations
/*inline*/ int16_t multiply_n16(const int8_t n)
{
	int t1, t2;
	t1 = 0 - n;
	t2 = t1 << 4;
	return t2;
}
// Multiply by -15 using the fewest operations
/*inline*/ int16_t multiply_n15(const int8_t n)
{
	int t1, t2;
	t1 = n << 4;
	t2 = n - t1;
	return t2;
}
// Multiply by -14 using the fewest operations
/*inline*/ int16_t multiply_n14(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = n - t1;
	t3 = t2 << 1;
	return t3;
}
// Multiply by -13 using the fewest operations
/*inline*/ int16_t multiply_n13(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 1;
	t4 = n - t3;
	return t4;
}
// Multiply by -12 using the fewest operations
/*inline*/ int16_t multiply_n12(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 2;
	return t3;
}
// Multiply by -11 using the fewest operations
/*inline*/ int16_t multiply_n11(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = n - t3;
	return t4;
}
// Multiply by -10 using the fewest operations
/*inline*/ int16_t multiply_n10(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	t5 = t4 << 1;
	return t5;
}
// Multiply by -9 using the fewest operations
/*inline*/ int16_t multiply_n9(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	return t4;
}
// Multiply by -8 using the fewest operations
/*inline*/ int16_t multiply_n8(const int8_t n)
{
	int t1, t2;
	t1 = 0 - n;
	t2 = t1 << 3;
	return t2;
}
// Multiply by -7 using the fewest operations
/*inline*/ int16_t multiply_n7(const int8_t n)
{
	int t1, t2;
	t1 = n << 3;
	t2 = n - t1;
	return t2;
}
// Multiply by -6 using the fewest operations
/*inline*/ int16_t multiply_n6(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 2;
	t2 = n - t1;
	t3 = t2 << 1;
	return t3;
}
// Multiply by -5 using the fewest operations
/*inline*/ int16_t multiply_n5(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = n - t3;
	return t4;
}
// Multiply by -4 using the fewest operations
/*inline*/ int16_t multiply_n4(const int8_t n)
{
	int t1, t2;
	t1 = 0 - n;
	t2 = t1 << 2;
	return t2;
}
// Multiply by -3 using the fewest operations
/*inline*/ int16_t multiply_n3(const int8_t n)
{
	int t1, t2;
	t1 = n << 2;
	t2 = n - t1;
	return t2;
}
// Multiply by -2 using the fewest operations
/*inline*/ int16_t multiply_n2(const int8_t n)
{
	int t1, t2;
	t1 = 0 - n;
	t2 = t1 << 1;
	return t2;
}
// Multiply by -1 using the fewest operations
/*inline*/ int16_t multiply_n1(const int8_t n)
{
	int t1;
	t1 = 0 - n;
	return t1;
}
// Multiply by 2 using the fewest operations
/*inline*/ int16_t multiply_2(const int8_t n)
{
	int t1;
	t1 = n << 1;
	return t1;
}
// Multiply by 3 using the fewest operations
/*inline*/ int16_t multiply_3(const int8_t n)
{
	int t1, t2;
	t1 = n << 1;
	t2 = t1 + n;
	return t2;
}
// Multiply by 4 using the fewest operations
/*inline*/ int16_t multiply_4(const int8_t n)
{
	int t1;
	t1 = n << 2;
	return t1;
}
// Multiply by 5 using the fewest operations
/*inline*/ int16_t multiply_5(const int8_t n)
{
	int t1, t2;
	t1 = n << 2;
	t2 = t1 + n;
	return t2;
}
// Multiply by 6 using the fewest operations
/*inline*/ int16_t multiply_6(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 1;
	return t3;
}
// Multiply by 7 using the fewest operations
/*inline*/ int16_t multiply_7(const int8_t n)
{
	int t1, t2;
	t1 = n << 3;
	t2 = t1 - n;
	return t2;
}
// Multiply by 8 using the fewest operations
/*inline*/ int16_t multiply_8(const int8_t n)
{
	int t1;
	t1 = n << 3;
	return t1;
}
// Multiply by 9 using the fewest operations
/*inline*/ int16_t multiply_9(const int8_t n)
{
	int t1, t2;
	t1 = n << 3;
	t2 = t1 + n;
	return t2;
}
// Multiply by 10 using the fewest operations
/*inline*/ int16_t multiply_10(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	return t3;
}
// Multiply by 11 using the fewest operations
/*inline*/ int16_t multiply_11(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	return t4;
}
// Multiply by 12 using the fewest operations
/*inline*/ int16_t multiply_12(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	return t3;
}
// Multiply by 13 using the fewest operations
/*inline*/ int16_t multiply_13(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	return t4;
}
// Multiply by 14 using the fewest operations
/*inline*/ int16_t multiply_14(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 1;
	return t3;
}
// Multiply by 15 using the fewest operations
/*inline*/ int16_t multiply_15(const int8_t n)
{
	int t1, t2;
	t1 = n << 4;
	t2 = t1 - n;
	return t2;
}
// Multiply by 16 using the fewest operations
/*inline*/ int16_t multiply_16(const int8_t n)
{
	int t1;
	t1 = n << 4;
	return t1;
}
// Multiply by 17 using the fewest operations
/*inline*/ int16_t multiply_17(const int8_t n)
{
	int t1, t2;
	t1 = n << 4;
	t2 = t1 + n;
	return t2;
}
// Multiply by 18 using the fewest operations
/*inline*/ int16_t multiply_18(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 1;
	return t3;
}
// Multiply by 19 using the fewest operations
/*inline*/ int16_t multiply_19(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	return t4;
}
// Multiply by 20 using the fewest operations
/*inline*/ int16_t multiply_20(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 2;
	return t3;
}
// Multiply by 21 using the fewest operations
/*inline*/ int16_t multiply_21(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	return t4;
}
// Multiply by 22 using the fewest operations
/*inline*/ int16_t multiply_22(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 23 using the fewest operations
/*inline*/ int16_t multiply_23(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - n;
	return t4;
}
// Multiply by 24 using the fewest operations
/*inline*/ int16_t multiply_24(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 3;
	return t3;
}
// Multiply by 25 using the fewest operations
/*inline*/ int16_t multiply_25(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + t2;
	return t4;
}
// Multiply by 26 using the fewest operations
/*inline*/ int16_t multiply_26(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 27 using the fewest operations
/*inline*/ int16_t multiply_27(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	return t4;
}
// Multiply by 28 using the fewest operations
/*inline*/ int16_t multiply_28(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	return t3;
}
// Multiply by 29 using the fewest operations
/*inline*/ int16_t multiply_29(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + n;
	return t4;
}
// Multiply by 30 using the fewest operations
/*inline*/ int16_t multiply_30(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 1;
	return t3;
}
// Multiply by 31 using the fewest operations
/*inline*/ int16_t multiply_31(const int8_t n)
{
	int t1, t2;
	t1 = n << 5;
	t2 = t1 - n;
	return t2;
}
// Multiply by 32 using the fewest operations
/*inline*/ int16_t multiply_32(const int8_t n)
{
	int t1;
	t1 = n << 5;
	return t1;
}
// Multiply by 33 using the fewest operations
/*inline*/ int16_t multiply_33(const int8_t n)
{
	int t1, t2;
	t1 = n << 5;
	t2 = t1 + n;
	return t2;
}
// Multiply by 34 using the fewest operations
/*inline*/ int16_t multiply_34(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 1;
	return t3;
}
// Multiply by 35 using the fewest operations
/*inline*/ int16_t multiply_35(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + t2;
	return t4;
}
// Multiply by 36 using the fewest operations
/*inline*/ int16_t multiply_36(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	return t3;
}
// Multiply by 37 using the fewest operations
/*inline*/ int16_t multiply_37(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	return t4;
}
// Multiply by 38 using the fewest operations
/*inline*/ int16_t multiply_38(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 39 using the fewest operations
/*inline*/ int16_t multiply_39(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - n;
	return t4;
}
// Multiply by 40 using the fewest operations
/*inline*/ int16_t multiply_40(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 3;
	return t3;
}
// Multiply by 41 using the fewest operations
/*inline*/ int16_t multiply_41(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 + n;
	return t4;
}
// Multiply by 42 using the fewest operations
/*inline*/ int16_t multiply_42(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 43 using the fewest operations
/*inline*/ int16_t multiply_43(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	t6 = t5 + n;
	return t6;
}
// Multiply by 44 using the fewest operations
/*inline*/ int16_t multiply_44(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 2;
	return t5;
}
// Multiply by 45 using the fewest operations
/*inline*/ int16_t multiply_45(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	return t4;
}
// Multiply by 46 using the fewest operations
/*inline*/ int16_t multiply_46(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 47 using the fewest operations
/*inline*/ int16_t multiply_47(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 4;
	t4 = t3 - n;
	return t4;
}
// Multiply by 48 using the fewest operations
/*inline*/ int16_t multiply_48(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 4;
	return t3;
}
// Multiply by 49 using the fewest operations
/*inline*/ int16_t multiply_49(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = t3 - t2;
	return t4;
}
// Multiply by 50 using the fewest operations
/*inline*/ int16_t multiply_50(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + t2;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 51 using the fewest operations
/*inline*/ int16_t multiply_51(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	return t4;
}
// Multiply by 52 using the fewest operations
/*inline*/ int16_t multiply_52(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 2;
	return t5;
}
// Multiply by 53 using the fewest operations
/*inline*/ int16_t multiply_53(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 2;
	t6 = t5 + n;
	return t6;
}
// Multiply by 54 using the fewest operations
/*inline*/ int16_t multiply_54(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 55 using the fewest operations
/*inline*/ int16_t multiply_55(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = t3 - n;
	return t4;
}
// Multiply by 56 using the fewest operations
/*inline*/ int16_t multiply_56(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 3;
	return t3;
}
// Multiply by 57 using the fewest operations
/*inline*/ int16_t multiply_57(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = t3 + n;
	return t4;
}
// Multiply by 58 using the fewest operations
/*inline*/ int16_t multiply_58(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 59 using the fewest operations
/*inline*/ int16_t multiply_59(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - n;
	return t4;
}
// Multiply by 60 using the fewest operations
/*inline*/ int16_t multiply_60(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	return t3;
}
// Multiply by 61 using the fewest operations
/*inline*/ int16_t multiply_61(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + n;
	return t4;
}
// Multiply by 62 using the fewest operations
/*inline*/ int16_t multiply_62(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 5;
	t2 = t1 - n;
	t3 = t2 << 1;
	return t3;
}
// Multiply by 63 using the fewest operations
/*inline*/ int16_t multiply_63(const int8_t n)
{
	int t1, t2;
	t1 = n << 6;
	t2 = t1 - n;
	return t2;
}
// Multiply by 64 using the fewest operations
/*inline*/ int16_t multiply_64(const int8_t n)
{
	int t1;
	t1 = n << 6;
	return t1;
}
// Multiply by 65 using the fewest operations
/*inline*/ int16_t multiply_65(const int8_t n)
{
	int t1, t2;
	t1 = n << 6;
	t2 = t1 + n;
	return t2;
}
// Multiply by 66 using the fewest operations
/*inline*/ int16_t multiply_66(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 5;
	t2 = t1 + n;
	t3 = t2 << 1;
	return t3;
}
// Multiply by 67 using the fewest operations
/*inline*/ int16_t multiply_67(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 5;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	return t4;
}
// Multiply by 68 using the fewest operations
/*inline*/ int16_t multiply_68(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 2;
	return t3;
}
// Multiply by 69 using the fewest operations
/*inline*/ int16_t multiply_69(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	return t4;
}
// Multiply by 70 using the fewest operations
/*inline*/ int16_t multiply_70(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + t2;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 71 using the fewest operations
/*inline*/ int16_t multiply_71(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - n;
	return t4;
}
// Multiply by 72 using the fewest operations
/*inline*/ int16_t multiply_72(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 3;
	return t3;
}
// Multiply by 73 using the fewest operations
/*inline*/ int16_t multiply_73(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 + n;
	return t4;
}
// Multiply by 74 using the fewest operations
/*inline*/ int16_t multiply_74(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 75 using the fewest operations
/*inline*/ int16_t multiply_75(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + t2;
	return t4;
}
// Multiply by 76 using the fewest operations
/*inline*/ int16_t multiply_76(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 2;
	return t5;
}
// Multiply by 77 using the fewest operations
/*inline*/ int16_t multiply_77(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 3;
	t6 = t5 - t4;
	return t6;
}
// Multiply by 78 using the fewest operations
/*inline*/ int16_t multiply_78(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 79 using the fewest operations
/*inline*/ int16_t multiply_79(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 4;
	t4 = t3 - n;
	return t4;
}
// Multiply by 80 using the fewest operations
/*inline*/ int16_t multiply_80(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 4;
	return t3;
}
// Multiply by 81 using the fewest operations
/*inline*/ int16_t multiply_81(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 + t2;
	return t4;
}
// Multiply by 82 using the fewest operations
/*inline*/ int16_t multiply_82(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 + n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 83 using the fewest operations
/*inline*/ int16_t multiply_83(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 + n;
	t5 = t4 << 1;
	t6 = t5 + n;
	return t6;
}
// Multiply by 84 using the fewest operations
/*inline*/ int16_t multiply_84(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 2;
	return t5;
}
// Multiply by 85 using the fewest operations
/*inline*/ int16_t multiply_85(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + t2;
	return t4;
}
// Multiply by 86 using the fewest operations
/*inline*/ int16_t multiply_86(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6, t7;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	t6 = t5 + n;
	t7 = t6 << 1;
	return t7;
}
// Multiply by 87 using the fewest operations
/*inline*/ int16_t multiply_87(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 2;
	t6 = t5 - t4;
	return t6;
}
// Multiply by 88 using the fewest operations
/*inline*/ int16_t multiply_88(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 3;
	return t5;
}
// Multiply by 89 using the fewest operations
/*inline*/ int16_t multiply_89(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 1;
	t4 = t3 + n;
	t5 = t4 << 3;
	t6 = t5 + n;
	return t6;
}
// Multiply by 90 using the fewest operations
/*inline*/ int16_t multiply_90(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 91 using the fewest operations
/*inline*/ int16_t multiply_91(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 3;
	t6 = t5 - t4;
	return t6;
}
// Multiply by 92 using the fewest operations
/*inline*/ int16_t multiply_92(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 2;
	return t5;
}
// Multiply by 93 using the fewest operations
/*inline*/ int16_t multiply_93(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 5;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	return t4;
}
// Multiply by 94 using the fewest operations
/*inline*/ int16_t multiply_94(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 4;
	t4 = t3 - n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 95 using the fewest operations
/*inline*/ int16_t multiply_95(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 5;
	t4 = t3 - n;
	return t4;
}
// Multiply by 96 using the fewest operations
/*inline*/ int16_t multiply_96(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 5;
	return t3;
}
// Multiply by 97 using the fewest operations
/*inline*/ int16_t multiply_97(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 5;
	t4 = t3 + n;
	return t4;
}
// Multiply by 98 using the fewest operations
/*inline*/ int16_t multiply_98(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = t3 - t2;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 99 using the fewest operations
/*inline*/ int16_t multiply_99(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 5;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	return t4;
}
// Multiply by 100 using the fewest operations
/*inline*/ int16_t multiply_100(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + t2;
	t5 = t4 << 2;
	return t5;
}
// Multiply by 101 using the fewest operations
/*inline*/ int16_t multiply_101(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + t2;
	t5 = t4 << 2;
	t6 = t5 + n;
	return t6;
}
// Multiply by 102 using the fewest operations
/*inline*/ int16_t multiply_102(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 103 using the fewest operations
/*inline*/ int16_t multiply_103(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 1;
	t6 = t5 + n;
	return t6;
}
// Multiply by 104 using the fewest operations
/*inline*/ int16_t multiply_104(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 3;
	return t5;
}
// Multiply by 105 using the fewest operations
/*inline*/ int16_t multiply_105(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = t3 - t2;
	return t4;
}
// Multiply by 106 using the fewest operations
/*inline*/ int16_t multiply_106(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6, t7;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 2;
	t6 = t5 + n;
	t7 = t6 << 1;
	return t7;
}
// Multiply by 107 using the fewest operations
/*inline*/ int16_t multiply_107(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 2;
	t6 = t5 - n;
	return t6;
}
// Multiply by 108 using the fewest operations
/*inline*/ int16_t multiply_108(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 2;
	return t5;
}
// Multiply by 109 using the fewest operations
/*inline*/ int16_t multiply_109(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 3;
	t2 = t1 + n;
	t3 = t2 << 2;
	t4 = t3 - t2;
	t5 = t4 << 2;
	t6 = t5 + n;
	return t6;
}
// Multiply by 110 using the fewest operations
/*inline*/ int16_t multiply_110(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 111 using the fewest operations
/*inline*/ int16_t multiply_111(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 4;
	t4 = t3 - n;
	return t4;
}
// Multiply by 112 using the fewest operations
/*inline*/ int16_t multiply_112(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 4;
	return t3;
}
// Multiply by 113 using the fewest operations
/*inline*/ int16_t multiply_113(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 4;
	t4 = t3 + n;
	return t4;
}
// Multiply by 114 using the fewest operations
/*inline*/ int16_t multiply_114(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = t3 + n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 115 using the fewest operations
/*inline*/ int16_t multiply_115(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 1;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 2;
	t6 = t5 + t4;
	return t6;
}
// Multiply by 116 using the fewest operations
/*inline*/ int16_t multiply_116(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 3;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 2;
	return t5;
}
// Multiply by 117 using the fewest operations
/*inline*/ int16_t multiply_117(const int8_t n)
{
	int t1, t2, t3, t4, t5, t6;
	t1 = n << 2;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - n;
	t5 = t4 << 2;
	t6 = t5 - t4;
	return t6;
}
// Multiply by 118 using the fewest operations
/*inline*/ int16_t multiply_118(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 119 using the fewest operations
/*inline*/ int16_t multiply_119(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 + n;
	t3 = t2 << 3;
	t4 = t3 - t2;
	return t4;
}
// Multiply by 120 using the fewest operations
/*inline*/ int16_t multiply_120(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 3;
	return t3;
}
// Multiply by 121 using the fewest operations
/*inline*/ int16_t multiply_121(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 3;
	t4 = t3 + n;
	return t4;
}
// Multiply by 122 using the fewest operations
/*inline*/ int16_t multiply_122(const int8_t n)
{
	int t1, t2, t3, t4, t5;
	t1 = n << 4;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + n;
	t5 = t4 << 1;
	return t5;
}
// Multiply by 123 using the fewest operations
/*inline*/ int16_t multiply_123(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 5;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 - n;
	return t4;
}
// Multiply by 124 using the fewest operations
/*inline*/ int16_t multiply_124(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 5;
	t2 = t1 - n;
	t3 = t2 << 2;
	return t3;
}
// Multiply by 125 using the fewest operations
/*inline*/ int16_t multiply_125(const int8_t n)
{
	int t1, t2, t3, t4;
	t1 = n << 5;
	t2 = t1 - n;
	t3 = t2 << 2;
	t4 = t3 + n;
	return t4;
}
// Multiply by 126 using the fewest operations
/*inline*/ int16_t multiply_126(const int8_t n)
{
	int t1, t2, t3;
	t1 = n << 6;
	t2 = t1 - n;
	t3 = t2 << 1;
	return t3;
}
// Multiply by 127 using the fewest operations
/*inline*/ int16_t multiply_127(const int8_t n)
{
	int t1, t2;
	t1 = n << 7;
	t2 = t1 - n;
	return t2;
}

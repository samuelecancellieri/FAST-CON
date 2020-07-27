#if !defined(DEFINES_H)
#define DEFINES_H
#define SEED1 23
#define SEED2 24
#define DEFAULT_SCALE      16
#define DEFAULT_EDGEFACTOR 16.
#define MIN_GLOBAL_SCALE    8
#define MF_RECV_SIZE_SMALL_SCALE   16
#define MF_RECV_SIZE                4
#define TENTRYPE(n)               (4*(n)+2)
#define ROUNDING 8
#define VALUE_TO_REMOVE_BY_MASK -1
#define NO_PREDECESSOR  -1
//#define NO_PRED 0xFFFFFFFFFFFFFFFF
#define NO_CONNECTIONS  -1

// old value -3
#define OTHER_RANK 0x20000000

#define TMP_PREDECESSOR 0x1000000000000000

#define ROOT_VERTEX     -4

#define ST_RANK_SIZE 6

#define COLOR_MASK 0x40000000
#define COLOR_MASK_64 0x4000000000000000
#define RED 0
#define BLUE 1

#ifndef size_minus_one
#define size_minus_one (size-1)
#endif

#endif

#ifndef _HEADER_H_
#define _HEADER_H_

// Devo introdurre DBG_LEVEL nelle funzioni make_struct....
// Nel frattempo:
//#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
//#define GPU_DEBUG_0
//#define GPU_DEBUG_1
//#define GPU_DEBUG_2
//#endif

// Data types
#define INT64_T_MPI_TYPE MPI_LONG_LONG
#define INT_T int64_t
#define INT32_T int32_t
#define PRI64 PRId64
#define MPI_INT_T MPI_LONG_LONG
#if !defined(MPI_INT32_T)
#define MPI_INT32_T MPI_INT
#endif

#define SHORT_INT int

// Atomic add in global memory on 64 bit is allowed only on device
// with major >= 2, anyway it supports only uint64 and not int64
#if defined(ATOMIC_UINT32) 
#define ATOMIC_T  unsigned int
#else
#define ATOMIC_T unsigned long long
#endif


// Vars

#define SIZE_MUST_BE_A_POWER_OF_TWO
extern int rank, size;
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
extern int lgsize;
#endif

#define GIGABYTE (double)(1024*1024*1024)

typedef struct
{
  INT_T *edges;       // List of edges
  INT_T *offset;      // Offset of vertices 
  INT_T *degree;      // Degree of vertices
  INT_T nedges;       // Total number of edges
  INT_T nverts;       // Total number of vertices
} adjlist;

typedef struct {
	INT_T   *unique_edges;  // Array of unique vertex in the edge list size m_nelems
	INT_T   *proc_offset;
	INT32_T	*pedges;        // Array of pointers to unique vertex size p_nelems=nedges
	INT32_T *pverts;        // Array of pointers to local vertex with size = nverts
	INT32_T *mask;          // Array of predecessors to unique vertex size m_nelems
	INT32_T p_nelems;       // Number of elements in pedges = nedges
	INT32_T m_nelems;       // Number of unique vertex in the edge list
} mask;

// Const

/* BDIM sets the number of thread per block. */
#define BDIM            256
#define MAXBLOCKS       65535

#define PRINT_MAX_SCALE (10)
#define PRINT_MAX_NVERTS (1<<PRINT_MAX_SCALE)
#define PRINT_MAX_NEDGES (32*PRINT_MAX_NVERTS)

// Added in Makefile
// Default debug level == 0, no debug
// #define DBG_LEVEL 1 to add debug 
// #define DBG_TIME to collect timing


// graph500 macros

#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
#define MOD_SIZE(v) ((v) & size_minus_one)
#define DIV_SIZE(v) ((v) >> lgsize)
#else
#define MOD_SIZE(v) ((v) % size)
#define DIV_SIZE(v) ((v) / size)
#endif
#define VERTEX_OWNER(v) ((int)(MOD_SIZE(v)))
#define VERTEX_LOCAL(v) ((INT_T)(DIV_SIZE(v)))
#define VERTEX_2_LOCAL(v) ((INT32_T)(DIV_SIZE(v)))
#define VERTEX_TO_GLOBAL(i) ((INT_T)((i) * size + rank))
#define VERTEX_2_GLOBAL(i,r) ((INT_T)((i) * size + r))


// Macros

#define MIN(a,b)        (((a)<(b))?(a):(b))
#define LOGLVL 1
#define LOG(lvl, fout, format, ...)	{\
						if (lvl >= LOGLVL) {\
							fprintf(fout, format, ## __VA_ARGS__);\
							fflush(fout);\
						}\
					}


#define LOG_STATS(format, ...)  {\
                                if (rank == 0) {\
                                        fprintf(fp_stats, format, ## __VA_ARGS__);\
                                        fflush(fp_stats);\
                                }\
                        }







#define START_TIMER(dbg, start) \
        {\
                if (dbg > 0) { \
                	cudaThreadSynchronize(); \
                        start = MPI_Wtime(); \
                } \
        }       
#define STOP_TIMER(dbg, start, stop, t, fout, fcaller) \
        {\
                if (dbg > 0) { \
                	cudaThreadSynchronize(); \
                        stop = MPI_Wtime(); \
                        t = stop - start; \
                        PRINT_TIME(fcaller, fout, t); \
                } \
        } 


#endif
/*
 *
 *ST con define
 *
 *
 * 
#define COLOR_MASK 0x40000000
#define COLOR_MASK_64 0x4000000000000000
#define RED 0
#define BLUE 1
*/

/* This is to make PRId64 working with c++ compiler */
#ifdef __cplusplus
#define __STDC_FORMAT_MACROS
#endif
/* header of int64_t and PRId64 */
#include <inttypes.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "reference_common.h"
#include "header.h"
#include "make_bfs.h"
#include "make_bfs_func.h"
#include "cputils.h"
#include "gputils.h"
#include "adj_func.h"
#include "cputestfunc.h"
#include "defines.h"
#include "mythrustlib.h"

extern FILE *fp_bfs;
extern int nthreads, maxblocks;
extern int rank, size;
extern int64_t MaxLabel;
extern int64_t MaxGlobalLabel;
extern int global_scale;
extern double global_edgefactor;
extern int dbg_lvl;
extern int num_bfs_roots;       // The number of bfs to perform
extern size_t current_allocated_size;
extern size_t freed_memory_size;
extern int validation;
extern double d_send_size_factor;
extern double d_recv_size_factor;
extern double d_mask_size_factor;
extern unsigned int green_exe_time;

INT_T max_recv_vertex = 0;

#define RECV_BFS_TAG 50000  // TAG Used for MPI Send and Receive

//-----------------------------------------------------------------------------

#ifdef DEBUG_AIDS
#define PDEBUG(FMT, ARGS...) fprintf(stdout, FMT, ##ARGS)
#else
#define PDEBUG(FMT, ARGS...) do { } while(0)
#endif
#define PERROR(FMT, ARGS...) fprintf(stdout, FMT, ##ARGS)

#ifdef NET_MPI

#define NET_HAS_TX_P2P  0
#define NET_HAS_RX_P2P  0
#undef  NET_HAS_RDMA

#define NET_INIT() do { } while(0)

#define NET_FREE() do { } while(0)

#define INIT_MPI_VAR                            \
        MPI_Request recv_req[size];                 \
        int senderc=0;

#define POST_IRECV()                            \
    senderc= 0;                                 \
    memset(recv_req,0,size*sizeof(MPI_Request)); \
    for (i = 0; i < size; ++i){                 \
        if (recv_count_per_proc[i] > 0){            \
            MPI_Irecv((h_recv_buff + recv_offset_per_proc[i]), \
                  recv_count_per_proc[i], MPI_INT32_T,  \
                  i, RECV_BFS_TAG+rank, MPI_COMM_WORLD, &recv_req[senderc]);    \
                  senderc++;                \
        }                           \
    }


#define WAIT_IRECV()                            \
    {           \
       MPI_Waitall(senderc, recv_req, MPI_STATUSES_IGNORE);\
    }

#define POST_SEND(COUNT_PER_PROC)                           \
    for (i = 0; i < size; ++i){                 \
        if (COUNT_PER_PROC[i] > 0){         \
            MPI_Send((h_send_buff + send_offset_per_proc[i]), \
                    COUNT_PER_PROC[i], MPI_INT32_T, i,  \
                 RECV_BFS_TAG+i, MPI_COMM_WORLD );          \
        }                           \
    }


//-----------------------------------------------------------------------------

#elif defined(NET_MPI_GPU_AWARE)

#warning "using GPU-aware MPI"

#define NET_HAS_TX_P2P  1
#define NET_HAS_RX_P2P  1
#undef  NET_HAS_RDMA

#define NET_INIT() do { } while(0)

#define NET_FREE() do { } while(0)

#define INIT_MPI_VAR                            \
        MPI_Request recv_req[size];             \
        int senderc=0;

#define POST_IRECV()                                                    \
        senderc= 0;                                                     \
        memset(recv_req,0,size*sizeof(MPI_Request));                    \
        for (i = 0; i < size; ++i){                                     \
                if (recv_count_per_proc[i] > 0){                        \
            MPI_Irecv((d_recv_buffer32 + recv_offset_per_proc[i]), \
                                  recv_count_per_proc[i], MPI_INT32_T,  \
                                  i, RECV_BFS_TAG+rank, MPI_COMM_WORLD, &recv_req[senderc]); \
                        senderc++;                                      \
                }                                                       \
        }


#define WAIT_IRECV()                                                    \
        {                                                               \
                MPI_Waitall(senderc, recv_req, MPI_STATUSES_IGNORE);    \
        }

#define POST_SEND(COUNT_PER_PROC)                                       \
        for (i = 0; i < size; ++i){                                     \
                if (COUNT_PER_PROC[i] > 0){                             \
                        MPI_Send((d_buffer32 + send_offset_per_proc[i]), \
                                 COUNT_PER_PROC[i], MPI_INT32_T, i,     \
                                 RECV_BFS_TAG+i, MPI_COMM_WORLD );      \
                }                                                       \
        }


#else
#error "make your choice of a network"
#endif

//#define DEBUG_256 1
#ifdef DEBUG_256
#define COUNT_VISITED(tmp_array, l_nvisited, fcaller) \
    call_thrust_remove_copy(d_pred, nverts, tmp_array,  &actual_nvisited, NO_PREDECESSOR); \
    fprintf(stdout,"[rank %d, level %d]\tVisited=%"PRI64",\tActual visited=%"PRI64"\tCaller %s\n", rank, bfs_level, l_nvisited, actual_nvisited,fcaller);
#else
#define COUNT_VISITED(tmp_array, l_nvisited, fcaller) do { } while(0)
#endif


int run_make_bfs_mask(adjlist *dg, adjlist *hg, mask *d_bitmask)
{
    // Vars
    int bfs_root_idx;       // The counter of actual bfs
    int64_t* bfs_roots = NULL;  // The roots of bfs
    double* edge_counts = NULL; // Number of edges visited in each bfs
    int validation_passed;      // Flag for validate step
    uint64_t seed1, seed2;      // Seeds to init the random num. gen.

    int validation_passed_one;  // Result of validate function
    int64_t edge_visit_count;   // The number of visited edges 
    int64_t nglobalverts;       // Total number of vertices
    csr_graph cg;           // Reference data struct
    int64_t *h_pred = NULL;     // The array of predecessors
    INT_T root = -1;        // root of bfs
    INT_T nvisited;         // number of visited edges

    // Timing
    double* bfs_times = NULL;   // Execution time of each bfs
    double bfs_start, bfs_stop;

    double bfs_green_time = 0;  // Execution time for green graph500
    int green_res = 0, green_rc=0;

    int max_level = 0;

    double* validate_times = NULL;  // Execution time of validate step 
    double validate_start; 
    double validate_stop; 
    double *stcon_nodes_visited = NULL;



    // Device arrays
    int64_t *d_pred = NULL;     // Predecessor array
    INT_T *d_queue = NULL;      // The current queue
    INT_T *d_queue_off = NULL;  // Offset of vertices in the queue
    INT_T *d_queue_deg = NULL;  // Degree of vertices in the queue
    INT_T *d_next_off = NULL;   // Offset to make next level frontier

    INT_T nverts = hg->nverts;  // Number of my vertices
    INT_T nedges = hg->nedges;  // Number of my edges
    
    INT_T *d_send_buff = NULL;  // The buffer for vertices to send 
    INT_T d_send_size = 0;      // Size of the buffer to send
    INT_T d_recv_size = 0;      // Size of the buffer to recv
    INT_T h_send_size = 0;
    INT_T h_recv_size = 0;
    INT_T d_mask_size = 0;          
    if (global_scale <= MIN_GLOBAL_SCALE) {
        d_send_size = 4*nedges;
        d_recv_size = 4*nedges;
        d_mask_size = 4*nedges;
    } else {
        d_send_size = d_send_size_factor * (INT_T)d_bitmask->m_nelems;
        d_recv_size = d_recv_size_factor * (INT_T)d_bitmask->m_nelems;
        d_mask_size = d_mask_size_factor * (INT_T)d_bitmask->m_nelems;
    }
    if (d_mask_size < nverts) d_mask_size = nverts; // we need these when running on 1 node

    h_send_size = d_send_size; // We have only vertex (no more predecessors and vertex)
    h_recv_size = d_recv_size;

    //h_send_buff and h_recv_buff: host buffers to send and receive vertices
    INT32_T *h_send_buff = NULL;
    INT32_T *h_recv_buff = NULL;

    INT32_T *d_buffer32 = NULL;  // Buffer used to copy 32bits Local Vertex from/to HOST Memory
    INT32_T *d_recv_buffer32 = NULL;

    INT_T *d_mask_1 = NULL;     // Support arrays
    INT_T *d_mask_2 = NULL;

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
        printDeviceFreeMemory(stderr);
        fprintf(stderr,"rank %d, Current allocated size %.3f GB\n", rank, current_allocated_size/GIGABYTE);
#endif
    printDeviceFreeMemory(stdout);

    //dg->offset, dg->degree, dg->edges
    cudaMalloc((void**)&d_pred, nverts*sizeof(int64_t));
    checkCUDAError("run_make_bfs_multi: malloc d_pred");
    cudaMalloc((void**)&d_queue, 2*nverts*sizeof(INT_T));
    checkCUDAError("run_make_bfs_multi: malloc d_queue");
    cudaMalloc((void**)&d_queue_off, nverts*sizeof(INT_T));
    checkCUDAError("run_make_bfs_multi: malloc d_queue_off");
    cudaMalloc((void**)&d_queue_deg, nverts*sizeof(INT_T));
    checkCUDAError("run_make_bfs_multi: malloc d_queue_deg");
    cudaMalloc((void**)&d_next_off, (nverts+1)*sizeof(INT_T));
    checkCUDAError("run_make_bfs_multi: malloc d_next_off");
    cudaMalloc((void**)&d_send_buff, d_send_size * sizeof(INT_T));
    checkCUDAError("run_make_bfs_multi: cudaMalloc d_send_buff");
    cudaMalloc((void**)&d_mask_1, d_mask_size * sizeof(INT_T));
    checkCUDAError("run_make_bfs_multi: cudaMalloc d_mask_1");
    cudaMalloc((void**)&d_mask_2, d_mask_size * sizeof(INT_T));
    checkCUDAError("run_make_bfs_multi: cudaMalloc d_mask_2");
    cudaMalloc((void**)&d_buffer32, d_recv_size * sizeof(INT32_T));
    checkCUDAError("run_make_bfs_multi: cudaMalloc d_buffer32");

#if NET_HAS_RX_P2P
    cudaMalloc((void**)&d_recv_buffer32, d_recv_size * sizeof(INT32_T));
    checkCUDAError("run_make_bfs_multi: cudaMalloc d_recv_buffer32");
    cudaMemset(d_recv_buffer32, 0x5A, d_recv_size * sizeof(INT32_T));
#else
    d_recv_buffer32 = d_buffer32;
#endif

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
        print_device_array32(d_bitmask->pedges,d_bitmask->p_nelems, fp_bfs, "run_make_bfs_multi: pedges");
        print_device_array32(d_bitmask->pverts, nverts, fp_bfs, "run_make_bfs_multi: pverts");
        print_device_array(d_bitmask->unique_edges,d_bitmask->m_nelems, fp_bfs, "run_make_bfs_multi: unique_edges");
#endif

    // d_pred is always int64_t!
    current_allocated_size = current_allocated_size + (5*nverts + d_send_size + 2*d_mask_size)*sizeof(INT_T) + nverts*sizeof(int64_t);
    current_allocated_size += d_recv_size * sizeof(INT32_T);

    double G_current_allocated_size = current_allocated_size/GIGABYTE;
    // Assuming thrust uses 4*nedges (2*next_level_vertices). It may use up to 4*next_level_vertices

    double G_current_allocated_size_plus_thrust = (current_allocated_size + 4*d_send_size*sizeof(INT_T))/GIGABYTE;

    // Print arrays allocated in GPU
    if (rank == 0) {
        fprintf(stdout,"\n" 
            "################################\n"
            "run_make_bfs_multi:\n"
            "################################\n"
            "d_pred: nverts\n"
            "d_queue: nverts\n"
            "d_queue_offset: nverts\n"
            "d_next_off: nverts\n"
            "d_send_buff: d_send_size\n"
            "d_mask_1: d_mask_size\n"
            "d_mask_2: d_mask_size\n"
            "nverts=%"PRI64"\n"
            "nedges=%"PRI64"\n"
            "unique edges=%d\n"
            "d_send_size=%"PRI64"\n"
            "d_mask_size=%"PRI64"\n"
            "d_recv_size=%"PRI64"\n"
            "current_allocated_size=%.3f GB\n"
            "################################\n"
            "\n",
            nverts, nedges, d_bitmask->m_nelems, d_send_size, d_mask_size,d_recv_size,
            G_current_allocated_size);
        fprintf(stdout,"\n" 
            "########################################\n"
            "run_make_bfs_multi:\n"
            "########################################\n"
            "sort_owners_bfs:\n"
            "thrust: < 2*next_level_vertices\n" 
            "(after unique) <<? 4*nedges\n"
            "free it!\n"
            "########################################\n"
            "max allocated size considering 4*d_send_size\n"
            "by thrust = %.3f GB\n"
            "########################################\n"
            "\n", G_current_allocated_size_plus_thrust);
            
    }

    // Allocate host memory
    bfs_roots = (int64_t*)callmalloc(num_bfs_roots*sizeof(int64_t), 
                     "run_make_bfs_multi: bfs_roots");
    edge_counts = (double*)callmalloc(num_bfs_roots*sizeof(double), 
                      "run_make_bfs_multi: edge_counts");
    bfs_times = (double*)callmalloc(num_bfs_roots*sizeof(double), 
                    "run_make_bfs_multi: bfs_times");
    validate_times = (double*)callmalloc(num_bfs_roots*sizeof(double), 
                         "run_make_bfs_multi: validate_times");
    h_pred = (int64_t*)callmalloc(nverts*sizeof(int64_t), 
                      "run_make_bfs_multi: h_pred");

    // Use pinned memory
    cudaMallocHost((void**)&h_send_buff, h_send_size*sizeof(INT32_T));
    checkCUDAError("run_make_bfs_multi: cudaMallocHost h_send_buff");
    cudaMallocHost((void**)&h_recv_buff, h_recv_size*sizeof(INT32_T));
    checkCUDAError("run_make_bfs_multi: cudaMallocHost h_recv_buff");

    // Collect the total number of vertices
    nglobalverts = 0; 
    nglobalverts = MaxGlobalLabel + 1;

    // Convert my data struct to reference csr_graph data struct
    // Build csr structures on Device
    convert_to_csr(hg, &cg, nglobalverts, fp_bfs);

    // Sample randomly the roots of bfs
    seed1 = SEED1; seed2 = SEED2;
    find_bfs_roots(&num_bfs_roots, &cg, seed1, seed2, bfs_roots);

    // Perform bfs
    // Init the validate flag to 1 (not passed)
    validation_passed = 1;

    PRINT_SPACE(__func__, fp_bfs, "SCALE", global_scale);
    PRINT_SPACE(__func__, fp_bfs, "nverts", nverts);
    PRINT_SPACE(__func__, fp_bfs, "nedges", nedges);

    NET_INIT(); 
    INT32_T *d_mask = d_bitmask->mask;
    INT32_T tmp_empty;
    cudaMemcpy (&tmp_empty, d_mask, 1*sizeof(INT32_T), cudaMemcpyDeviceToHost);
    checkCUDAError("make_bfs_multi: fake cuda Memcpy");

    // Execute the loop to calculate energy consumption for Green Graph500
    bfs_root_idx = 0;

    if (green_exe_time > 0) {
       if (rank == 0) {
        fprintf(stdout,"\n"
                       "########################################################################\n"
                       "STARTING EXECUTION OF BFS SEARCH CYCLES - GREEN GRAPH 500 FOR %d seconds\n"
                       "########################################################################\n",
                        green_exe_time);
        }

        while (1) {
            root = bfs_roots[bfs_root_idx];
            // Zero the number of visited vertices
            nvisited = 0;
            // Set -1 to all predecessors
            cudaMemset(d_pred, NO_PREDECESSOR, nverts*sizeof(int64_t));
            checkCUDAError("run_run_make_bfs_maskbfs_multi: memset d_pred");
            cudaMemset(d_mask, NO_PREDECESSOR, sizeof(INT32_T) * (INT_T)d_bitmask->m_nelems);
            checkCUDAError("run_make_bfs_mask: memset d_bitmask->mask");

            // Output bfs idx and root
            LOG(dbg_lvl, fp_bfs, "\nRunning BFS %d, root vertex %"PRI64"\n", bfs_root_idx, root)

            MPI_Barrier(MPI_COMM_WORLD);

            // make_bfs
            bfs_start = MPI_Wtime();
            make_bfs_mask(root, hg, &nvisited, h_send_buff,
                        h_recv_buff, h_send_size, h_recv_size,
                        dg, d_pred, d_queue, d_queue_off,
                        d_queue_deg, d_next_off,
                        d_send_buff, d_send_size, d_recv_size,
                        d_mask_1, d_mask_2, d_mask_size, d_buffer32, d_recv_buffer32,
                        d_bitmask);

            bfs_stop = MPI_Wtime();
            bfs_green_time += bfs_stop - bfs_start;

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            if (rank==0) {
                fprintf(stdout,"Green Graph500\n"
                           "Execution Time = %g\n"
                           "Loop =%d\n", bfs_green_time, bfs_root_idx);
            }
#endif

            green_rc = (bfs_green_time>green_exe_time);
            MPI_Allreduce(&green_rc, &green_res, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (green_res > 0) break;

            bfs_root_idx = (bfs_root_idx+1) % num_bfs_roots;
        } // while (bfs_green_time < green_exe_time)

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            fprintf(stdout,"\n"
                       "########################################################################################\n"
                       "EXECUTION OF BFS SEARCH CYCLES - GREEN GRAPH 500 COMPLETED!!!! Elapsed time %.2f seconds\n"
                       "########################################################################################\n",
                        bfs_green_time);
        }
    }
    // Execute the loop to calculate TEPS for Graph500
    for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {
        root = bfs_roots[bfs_root_idx];
        // Zero the number of visited vertices 
        nvisited = 0;
        // Set -1 to all predecessors
        cudaMemset(d_pred, NO_PREDECESSOR, nverts*sizeof(int64_t));
        checkCUDAError("run_run_make_bfs_maskbfs_multi: memset d_pred");
        cudaMemset(d_mask, NO_PREDECESSOR, sizeof(INT32_T) * (INT_T)d_bitmask->m_nelems);
        checkCUDAError("run_make_bfs_mask: memset d_bitmask->mask");

        // Output bfs idx and root
        LOG(dbg_lvl, fp_bfs, "\nRunning BFS %d, root vertex %"PRI64"\n", bfs_root_idx, root)

        MPI_Barrier(MPI_COMM_WORLD);

        // make_bfs
        bfs_start = MPI_Wtime();
        max_level = make_bfs_mask(root, hg, &nvisited, h_send_buff,
                   h_recv_buff, h_send_size, h_recv_size,
                   dg, d_pred, d_queue, d_queue_off, 
                   d_queue_deg, d_next_off,
                   d_send_buff, d_send_size, d_recv_size,
                   d_mask_1, d_mask_2, d_mask_size, d_buffer32, d_recv_buffer32,
                   d_bitmask);
        bfs_stop = MPI_Wtime();
        bfs_times[bfs_root_idx] = bfs_stop - bfs_start;
        PRINT_TIME("run_make_bfs_mask:bfs_time_i", fp_bfs, bfs_times[bfs_root_idx]);

        // Copy predecessor array back to host
        cudaMemcpy(h_pred, d_pred, nverts*sizeof(int64_t), cudaMemcpyDeviceToHost);
        checkCUDAError("run_make_bfs_mask: d_pred->h_pred");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
        print_array_64t(h_pred, nverts, fp_bfs, "run_make_bfs_multi: h_pred");
        print_device_array32(d_bitmask->mask,d_bitmask->m_nelems, fp_bfs, "run_make_bfs_multi: mask");
#endif

        MPI_Barrier(MPI_COMM_WORLD);

        if (!validation) {
           validation_passed_one = 1;
        }
        else {
            // Validate BFS
            validate_start = MPI_Wtime();
            validation_passed_one = validate_bfs_result(&cg, root, h_pred, nvisited);
            validate_stop = MPI_Wtime();
            validate_times[bfs_root_idx] = validate_stop - validate_start;
        }
    
        // Check validate
        if (!validation_passed_one) {
            validation_passed = 0;
            if (rank == 0) 
                fprintf(stderr, "Validation failed for this"
                    " BFS root; skipping rest.\n");
            fprintf(stdout, "RANK %d, max vertex received:                %"PRI64"\n", rank, max_recv_vertex);

            break;
        }

        // Calculate number of input edges visited. 
        edge_visit_count = 0;
        count_visited_edges(&cg, edge_counts, &edge_visit_count, bfs_root_idx, h_pred);
    }

    // Print graph500 output
    if (rank == 0) 
    {
        if (!validation_passed) {
            fprintf(stdout, "No results printed for invalid run.\n");
        } 
        else 
        {
            int i;
            fprintf(stdout, "global_scale:                   %d\n", global_scale);
            fprintf(stdout, "global_edgefactor:              %.2g\n", global_edgefactor);
            fprintf(stdout, "NBFS:                           %d\n", num_bfs_roots);
            fprintf(stdout, "num_mpi_processes:              %d\n", size);
            fprintf(stdout, "max bfs level:                  %d\n", max_level);
            double stats[s_LAST];
            get_statistics(bfs_times, num_bfs_roots, stats);
            fprintf(stdout, "min_time:                       %g s\n", stats[s_minimum]);
            fprintf(stdout, "firstquartile_time:             %g s\n", stats[s_firstquartile]);
            fprintf(stdout, "median_time:                    %g s\n", stats[s_median]);
            fprintf(stdout, "thirdquartile_time:             %g s\n", stats[s_thirdquartile]);
            fprintf(stdout, "max_time:                       %g s\n", stats[s_maximum]);
            fprintf(stdout, "mean_time:                      %g s\n", stats[s_mean]);
            fprintf(stdout, "stddev_time:                    %g\n", stats[s_std]);
            get_statistics(edge_counts, num_bfs_roots, stats);
            fprintf(stdout, "min_nedge:                      %.11g\n", stats[s_minimum]);
            fprintf(stdout, "firstquartile_nedge:            %.11g\n", stats[s_firstquartile]);
            fprintf(stdout, "median_nedge:                   %.11g\n", stats[s_median]);
            fprintf(stdout, "thirdquartile_nedge:            %.11g\n", stats[s_thirdquartile]);
            fprintf(stdout, "max_nedge:                      %.11g\n", stats[s_maximum]);
            fprintf(stdout, "mean_nedge:                     %.11g\n", stats[s_mean]);
            fprintf(stdout, "stddev_nedge:                   %.11g\n", stats[s_std]);
            double* secs_per_edge = (double*)callmalloc(num_bfs_roots * sizeof(double), 
                        "PRINT_RESULTS");
            for (i = 0; i < num_bfs_roots; ++i) secs_per_edge[i] = bfs_times[i] / edge_counts[i];
            get_statistics(secs_per_edge, num_bfs_roots, stats);
            fprintf(stdout, "min_TEPS:                       %g TEPS\n", 1. / stats[s_maximum]);
            fprintf(stdout, "firstquartile_TEPS:             %g TEPS\n", 1. / stats[s_thirdquartile]);
            fprintf(stdout, "median_TEPS:                    %g TEPS\n", 1. / stats[s_median]);
            fprintf(stdout, "thirdquartile_TEPS:             %g TEPS\n", 1. / stats[s_firstquartile]);
            fprintf(stdout, "max_TEPS:                       %g TEPS\n", 1. / stats[s_minimum]);
            fprintf(stdout, "harmonic_mean_TEPS:             %g TEPS\n", 1. / stats[s_mean]);
            fprintf(stdout, "harmonic_stddev_TEPS:           %g\n", 
                stats[s_std] / (stats[s_mean] * stats[s_mean] * sqrt(num_bfs_roots - 1)));
            free(secs_per_edge); secs_per_edge = NULL;
            get_statistics(validate_times, num_bfs_roots, stats);
            fprintf(stdout, "min_validate:                   %g s\n", stats[s_minimum]);
            fprintf(stdout, "firstquartile_validate:         %g s\n", stats[s_firstquartile]);
            fprintf(stdout, "median_validate:                %g s\n", stats[s_median]);
            fprintf(stdout, "thirdquartile_validate:         %g s\n", stats[s_thirdquartile]);
            fprintf(stdout, "max_validate:                   %g s\n", stats[s_maximum]);
            fprintf(stdout, "mean_validate:                  %g s\n", stats[s_mean]);
            fprintf(stdout, "stddev_validate:                %g\n", stats[s_std]);

        }
    }

    NET_FREE();

    // Free
    free(bfs_roots);
    free(bfs_times); 
    free(validate_times); 
    free(edge_counts); edge_counts = NULL;
    free_csr_graph(&cg);
    free(h_pred);
    cudaFreeHost(h_send_buff);
    cudaFreeHost(h_recv_buff);

    cudaFree(d_pred); d_pred = NULL;
    cudaFree(d_queue); d_queue = NULL;
    cudaFree(d_queue_off); d_queue_off = NULL;
    cudaFree(d_queue_deg); d_queue_deg = NULL;

    cudaFree(d_next_off); d_next_off = NULL;
    cudaFree(d_send_buff); d_send_buff = NULL;
    cudaFree(d_buffer32); d_buffer32 = NULL;
    cudaFree(d_mask_1); d_mask_1 = NULL;
    cudaFree(d_mask_2); d_mask_2 = NULL;

    cudaFree(d_bitmask->pedges);
    cudaFree(d_bitmask->pverts);
    cudaFree(d_bitmask->mask);
    cudaFree(d_bitmask->unique_edges);
    cudaFree(d_bitmask->proc_offset);

#if NET_HAS_RX_P2P
    cudaFree(d_recv_buffer32); d_recv_buffer32 = NULL;
#endif
    return 0;
}

//   h_recv_buff is the buffer in HOST memory used to receive edges from other nodes. This becomes INT32_T
//   h_send_buff is the buffer in HOST memory used to send edges to other nodes. This becomes INT32_T
//   d_recv_buff is the buffer in DEVICE memory used to manipulate the edges that are received from other nodes. This CANNOT become INT32_T
//   d_send_buff is the buffer in DEVICE memory used to prepare the edges that will be sent to other nodes. This CANNOT become INT32_T
//   we need a new INT32_T buffer in DEVICE memory used copy to/from HOST memory send or recv buff d_buffer32

int make_bfs_mask(INT_T root, adjlist *hg, INT_T *nvisited, 
           INT32_T *h_send_buff, INT32_T *h_recv_buff,
           INT_T h_send_size, INT_T h_recv_size,
           adjlist *dg, int64_t *d_pred, INT_T *d_queue, 
           INT_T * d_queue_off, INT_T *d_queue_deg, 
           INT_T *d_next_off,
           INT_T *d_send_buff, INT_T d_send_size, INT_T d_recv_size,
           INT_T *d_mask_1, INT_T *d_mask_2, INT_T d_mask_size,
           INT32_T* d_buffer32, INT32_T* d_recv_buffer32,
           mask *d_bitmask)
{
    INT_T nverts = hg->nverts;      // The number of my vertices
    INT_T queue_count = 0;          // The number of vertices in the queue
    INT_T new_queue_count = 0;      // The number of vertices in the 
                        // next level queue
    INT_T global_new_queue_count = 0;   // Sum of all new_queue_count
    INT_T next_level_vertices = 0;      // The number of elements in the next frontier  

    int bfs_level = 0;          // Bfs level
    int max_bfs_level = 0;
    INT_T *d_count_per_proc = NULL;     // Number of vertices per procs on device
    INT_T *h_count_per_proc = NULL;     // Number of vertices per procs on host
    INT_T *send_count_per_proc = NULL;  

    INT_T *send_offset_per_proc = NULL; 
    INT_T *recv_count_all_proc = NULL;  
    INT_T *recv_count_per_proc = NULL;
    INT_T *recv_offset_per_proc = NULL;

    INT_T *d_recv_offset_per_proc = NULL; //Receive buffer offset per procs on device

    // Support vars
    int i;
    INT_T myoff;
    INT_T mynverts;
    INT32_T *d_myedges;
    INT_T recv_count;
    INT_T non_local_count;
    INT_T *d_q_1=NULL, *d_q_2=NULL;
    double start=0, stop=0, t=0;

    START_TIMER(dbg_lvl, start);
    cudaMalloc((void**)&d_count_per_proc, 2*size*sizeof(INT_T)); 
    checkCUDAError("make_bfs: cudaMalloc d_count_per_proc");

    cudaMalloc((void**)&d_recv_offset_per_proc, (size+1)*sizeof(INT_T));
    checkCUDAError("make_bfs: cudaMalloc d_recv_offset_per_proc");


    send_count_per_proc = (INT_T*)callmalloc((size+1)*sizeof(INT_T), 
                "make_bfs: malloc send_count_per_proc");
    send_offset_per_proc = (INT_T*)callmalloc((size+1)*sizeof(INT_T), 
                "make_bfs: malloc send_offset_per_proc");
    h_count_per_proc = (INT_T*)callmalloc((2*size+1)*sizeof(INT_T), 
                "make_bfs: malloc h_count_per_proc");
    recv_count_all_proc = (INT_T*)callmalloc((size*size+1)*sizeof(INT_T),
                "make_bfs: malloc recv_count_all_proc");
    recv_count_per_proc = (INT_T*)callmalloc((size+1)*sizeof(INT_T),
                "make_bfs: malloc recv_count_per_proc");
    recv_offset_per_proc = (INT_T*)callmalloc((size+1)*sizeof(INT_T),
                "make_bfs: malloc recv_offset_per_proc");

    memset(send_count_per_proc, 0, (size+1)*sizeof(INT_T));
    memset(send_offset_per_proc, 0, (size+1)*sizeof(INT_T));
    memset(recv_count_per_proc, 0, (size+1)*sizeof(INT_T));
    memset(recv_offset_per_proc, 0, (size+1)*sizeof(INT_T));
    memset(h_count_per_proc, 0, (2*size+1)*sizeof(INT_T));
    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cpy_malloc_set");

    INT32_T *d_bitmask_pverts = d_bitmask->pverts;
    INT32_T *d_bitmask_mask = d_bitmask->mask;
    INT32_T label;
    INT_T m_nelems = (INT_T)(d_bitmask->m_nelems);
    INT32_T local_root32;
    // Number of visited vertex calculated counting d_pred array elements

    START_TIMER(dbg_lvl, start);
    if (rank == VERTEX_OWNER(root)) {
        //Vertex in d_queue are LOCAL
        INT_T lroot = VERTEX_LOCAL(root);  // Get root vertex in local value
        local_root32 = (INT32_T)lroot;     // copy into a 32bit
        INT_T pred_root;
        cudaMemcpy(d_queue, &lroot, 1*sizeof(INT_T), cudaMemcpyHostToDevice); // Enqueue root vertex
        checkCUDAError("make_bfs_multi: root->d_queue");

        //Vertex in d_pred are GLOBAL
        cudaMemcpy (&label, &d_bitmask_pverts[local_root32], 1*sizeof(INT32_T), cudaMemcpyDeviceToHost);
        checkCUDAError("make_bfs_multi: d_bitmask_pverts->label");

        if (label != NO_CONNECTIONS) {
           cudaMemcpy(&d_bitmask_mask[label], &local_root32, 1*sizeof(INT32_T), cudaMemcpyHostToDevice); //Update bitmask
           checkCUDAError("make_bfs_multi: local_root->d_mask");
           pred_root = rank;
        } else {
            fprintf(stdout,"[rank %d] WARNING ROOT VERTEX NO LOCAL CONNECTIONS!!!!", rank);
            pred_root = ROOT_VERTEX;
        }

       cudaMemcpy(&d_pred[local_root32], &pred_root, 1*sizeof(int64_t), cudaMemcpyHostToDevice);
       checkCUDAError("make_bfs_multi: pred_root->root");

        queue_count++;  
    }
    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cpy_root");

    INIT_MPI_VAR;  // Init variables used in all MPI communications

    while(1) {
        LOG(dbg_lvl, fp_bfs, "\nTIME SPACE:*** *** bfs_level_start *** ***\n");
        PRINT_SPACE(__func__, fp_bfs, "bfs_level", bfs_level);

        bfs_level += 1;
        if (queue_count > 0) {
            /*  Step A of the paper */
            nvisited[0] += queue_count;
            CHECK_SIZE("queue_count", queue_count, "nverts", nverts, __func__);
            // Vertices in d_queue are LOCAL
            make_queue_deg(d_queue, queue_count, d_queue_off, d_queue_deg, dg->offset, nverts);
            make_queue_offset(d_queue_deg, queue_count, d_next_off, &next_level_vertices);
            /* In the beginning, with only the root, queue_count is one and next_level_vertices is the number of root's neighbors. */
            /* d_queue_off contains the offset of the root in the adjency list. d_next_off has a single element equal to zero? */
        } 

        int LARGE=0;

        /* Step B of the paper */
        // Vertices in d_queue are LOCAL
        /* Use a mask to track visited vertices */
        if (next_level_vertices > 0) {
            if (next_level_vertices > m_nelems) {
               LARGE=1;
               START_TIMER(dbg_lvl, start);
               cudaMemset(d_mask_1, VALUE_TO_REMOVE_BY_MASK, m_nelems*sizeof(INT_T));
               STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cudaMemset d_mask_1");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
        print_device_array(d_next_off, queue_count, fp_bfs, "make_bfs_multi: D_NEXTOFF");
#endif

               binary_expand_queue_mask_large(next_level_vertices, dg->edges, d_queue, d_next_off, d_queue_off,
                                              queue_count, d_mask_1, d_bitmask);
               next_level_vertices = m_nelems;

            } else {
               binary_expand_queue_mask(next_level_vertices, dg->edges, d_queue, d_next_off, d_queue_off,
                                        queue_count, d_mask_1, d_bitmask);
            }

            /* Compact send_array to remove already seen vertices */
            INT_T nelems_removed_by_mask;
            START_TIMER(dbg_lvl, start);
            call_thrust_remove_copy(d_mask_1, next_level_vertices, d_send_buff,  &nelems_removed_by_mask, VALUE_TO_REMOVE_BY_MASK);
            STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "call_thrust_remove_copy");
            next_level_vertices = nelems_removed_by_mask;
        }

        /* Step D of the paper */
        if (next_level_vertices > 0) {
            CHECK_SIZE("next_level_vertices", next_level_vertices, "size of d_buffer32", d_send_size, __func__);
            CHECK_SIZE("next_level_vertices", next_level_vertices, "size of d_mask", d_mask_size, __func__);

            // Calculate owners of Next level vertices
            bfs_owners(d_send_buff, next_level_vertices, d_mask_1, d_mask_2);
            sort_owners_bfs(d_mask_1, d_mask_2, next_level_vertices);
            bfs_count_vertices(d_mask_1, next_level_vertices, d_count_per_proc, send_count_per_proc, h_count_per_proc);

            // Use exclusive scan to count how many vertices to send to each node
            exclusive_scan(send_count_per_proc, send_offset_per_proc, size, size+1);
            // Vertices in d_send_buff are reordered according to d_mask_2, converted into LOCAL 32bit and copied into d_buffer32
            bfs_back_vertices32(d_send_buff, next_level_vertices, d_mask_2, d_buffer32);
        }

        START_TIMER(dbg_lvl, start);
        myoff = send_offset_per_proc[rank];     // Offset within the array of the vertices that remain local
        mynverts =  send_count_per_proc[rank];  // Number of vertices that remain local
        d_myedges = d_buffer32 + myoff;         // Pointer to the vertices that remain local
        send_count_per_proc[rank] = 0;          // Zero the value to send for local vertices
        non_local_count = (send_offset_per_proc[size] > mynverts ) ?
                          (send_offset_per_proc[size] - mynverts) : 0;

        MPI_Allgather(send_count_per_proc, size, MPI_INT_T, recv_count_all_proc, size, MPI_INT_T, MPI_COMM_WORLD);
        
        bfs_count_verts_to_recv(recv_count_all_proc, recv_count_per_proc);
        exclusive_scan(recv_count_per_proc, recv_offset_per_proc, size, size+1);
        recv_count = recv_offset_per_proc[size];
        
        //DR
        // the Allgather is an implicit barrier
        // QUESTION: can this be made an all-local computation? seems we can!!!
        START_TIMER(dbg_lvl, start);
        // Check if the buffer allocated is large enough to receive all data from other processors
        CHECK_SIZE("recv_count", recv_count, "h_recv_size", h_recv_size, __func__);
        //MPI RECV using 32bits arrays
        POST_IRECV();
        STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_irecv");

        if (mynverts > 0) {
           if (LARGE == 1) {
               atomic_enqueue_local(d_myedges, mynverts, d_queue, d_pred, nverts, &new_queue_count);
               // Added code to count predecessors
               COUNT_VISITED(d_mask_1, (new_queue_count+nvisited[0]), "atomic_enqueue_local");
           } else {
               // Reuse d_mask_1 as support array
               unique_and_atomic_enqueue_local(d_myedges, mynverts, d_mask_1, d_queue, d_pred, nverts, &new_queue_count);
               // Added code to count predecessors
               COUNT_VISITED(d_mask_1, (new_queue_count+nvisited[0]), "unique_and_atomic_enqueue_local");
           }
        }
        queue_count = new_queue_count;
        new_queue_count = 0;

        START_TIMER(dbg_lvl, start);
        if (non_local_count > 0) {
#if !NET_HAS_TX_P2P      
            //Copy vertices array from Device to Host using 32bits array
            cudaMemcpy(h_send_buff, d_buffer32, next_level_vertices*sizeof(INT32_T), cudaMemcpyDeviceToHost);
            checkCUDAError("make_bfs_multi: d_buffer32->h_send_buff");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            print_array_32t(h_send_buff, next_level_vertices, fp_bfs, "H_SEND_BUFF");
#endif
#else  //NET_HAS_TX_P2P
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            cudaMemcpy(h_send_buff, d_buffer32, next_level_vertices*sizeof(INT32_T), cudaMemcpyDeviceToHost);
            checkCUDAError("make_bfs_multi: d_buffer32->h_send_buff");
            print_array_32t(h_send_buff, next_level_vertices, fp_bfs, "H_SEND_BUFF");
#endif
#endif //NET_HAS_TX_P2P
        }
        STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cpy_send_buff");

        START_TIMER(dbg_lvl, start);
        //MPI Send using 32bits
        POST_SEND(send_count_per_proc);
        WAIT_IRECV();
        STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_send_wait");

        if (recv_count > 0) {
            max_recv_vertex = (max_recv_vertex < recv_count ? recv_count : max_recv_vertex);
            START_TIMER(dbg_lvl, start);
                        
#if NET_HAS_RX_P2P
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            cudaMemcpy(h_recv_buff, d_recv_buffer32, recv_count*sizeof(INT32_T), cudaMemcpyDeviceToHost);
            checkCUDAError("make_bfs_multi: h_recv_buff<-d_recv_buff");
            print_array_32t(h_recv_buff, recv_count, fp_bfs, "H_RECV_BUFF<-d_buffer2");
#endif
#else // NET_HAS_RX_P2P
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            print_array_32t(h_recv_buff, recv_count, fp_bfs, "H_RECV_BUFF");
#endif
            CHECK_SIZE("recv_count", recv_count, "d_recv_size", d_recv_size, __func__);
            //Copy back received vertices from Host to Device using 32bits array
            cudaMemcpy(d_recv_buffer32, h_recv_buff, recv_count*sizeof(INT32_T), cudaMemcpyHostToDevice);
            checkCUDAError("make_bfs_multi: h_recv_buff->d_buffer32");
#endif // NET_HAS_RX_P2P            
            cudaMemcpy(d_recv_offset_per_proc, recv_offset_per_proc, (size+1)*sizeof(INT_T), cudaMemcpyHostToDevice);
            checkCUDAError("make_bfs_multi: recv_offset_per_proc->d_recv_offset_per_proc");
            STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cpy_recv_buff");

            d_q_1 = d_queue_deg;
            d_q_2 = d_queue_off;
            atomic_enqueue_recv(d_recv_buffer32, recv_count, d_recv_offset_per_proc,
                                d_q_1, d_q_2, &new_queue_count, d_pred, nverts, d_bitmask);
            // d_q_2 contains the new vertices received from other nodes, already in LOCAL form
            COUNT_VISITED(d_mask_1, (new_queue_count+nvisited[0]+queue_count), "atomic_enqueue_recv");
        } 

        START_TIMER(dbg_lvl, start);
        // Wait atomic_enqueue_local
        if (new_queue_count > 0) {
            cudaMemcpy(d_queue + queue_count, d_q_2, new_queue_count*sizeof(INT_T), cudaMemcpyDeviceToDevice);
            queue_count += new_queue_count;
        }
        STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cpy_devTodev");

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
        print_device_array(d_pred, nverts, fp_bfs, "make_bfs_multi: D_PRED");
        print_device_array32(d_bitmask_mask, (INT_T)m_nelems,  fp_bfs, "make_bfs_multi: D_MASK");
#endif

        START_TIMER(dbg_lvl, start);
        MPI_Allreduce(&queue_count, &global_new_queue_count, 1, MPI_INT_T, MPI_SUM, MPI_COMM_WORLD);
        STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_allreduce");
        LOG(dbg_lvl, fp_bfs, "TIME SPACE:*** *** bfs_level_end *** ***\n\n");

        if (global_new_queue_count == 0) {
            break;
        }       
        new_queue_count = 0;
        next_level_vertices = 0;
        recv_count = 0;
        non_local_count = 0;
        mynverts = 0;
        memset(send_count_per_proc, 0, (size+1)*sizeof(INT_T));
        memset(send_offset_per_proc, 0, (size+1)*sizeof(INT_T));
        memset(recv_count_per_proc, 0, (size+1)*sizeof(INT_T));
        memset(recv_offset_per_proc, 0, (size+1)*sizeof(INT_T));
        memset(h_count_per_proc, 0, (2*size+1)*sizeof(INT_T));
#ifdef NET_HAS_RDMA
        memset(recv_addr_per_proc, 0, (size+1)*sizeof(uint64_t));
        memset(recv_addr_all_proc, 0, (size*size+1)*sizeof(uint64_t));
#endif
    }  // While(1)

    MPI_Allreduce(MPI_IN_PLACE, nvisited, 1, MPI_INT_T, MPI_SUM, MPI_COMM_WORLD);
    LOG(dbg_lvl, fp_bfs, "nvisited=%"PRI64"\n", nvisited[0]);

    if (rank==0) {
        max_bfs_level = ((max_bfs_level < bfs_level) ? bfs_level : max_bfs_level);
    }


    // Graph has been visited but now we need to send/receive predecessors
    recv_count = 0;
    non_local_count = 0;
    memset(send_count_per_proc, 0, (size+1)*sizeof(INT_T));
    memset(send_offset_per_proc, 0, (size+1)*sizeof(INT_T));
    memset(recv_count_per_proc, 0, (size+1)*sizeof(INT_T));
    memset(recv_offset_per_proc, 0, (size+1)*sizeof(INT_T));
    memset(h_count_per_proc, 0, (2*size+1)*sizeof(INT_T));

    INT_T next_nverts = 0; // Vertex for which I need predecessor information (both local and remote)
    INT_T *d_verts_owners = d_queue_deg;  // Array of vertex owners (we reuse an existing device vector)
    INT_T *d_vert_ids     = d_queue_off;  // Array of vertex ids (we reuse an existing device vector)

    // Remove vertex not visited. Copy vertex owner into d_verts_owners and vertex id into d_vert_ids ordered by vertex owner
    // Using d_mask_1 and d_mask_2 as temporary buffers
    bfs_remove_pred(d_pred, d_mask_1, d_mask_2, nverts, d_verts_owners, d_vert_ids, &next_nverts);

    // Reset d_pred
    cudaMemset(d_pred, NO_PREDECESSOR, nverts*sizeof(int64_t));
    //d_mask_1 contains the rank holding predecessors info
    //d_mask_2 contains vertex id for which I need predecessors info

    //Count how many predecessors each processor should provide to this node
    bfs_count_vertices(d_verts_owners, next_nverts, d_count_per_proc, send_count_per_proc, h_count_per_proc);
    exclusive_scan(send_count_per_proc, send_offset_per_proc, size, size+1);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
    print_device_array(d_mask_1, next_nverts,  fp_bfs, "make_bfs_multi: d_mask_1");
    print_device_array(d_mask_2, next_nverts,  fp_bfs, "make_bfs_multi: d_mask_2");
    print_array_64t(send_offset_per_proc, size+1, fp_bfs, "make_bfs_multi: send_count_per_proc");
#endif

    // Copy vertex for which I need predecessor into 32 bits buffer
    bfs_copy32(d_vert_ids, next_nverts, d_buffer32);

    // This is the number of local vertices whose predecessor was found by this node
    mynverts =  send_count_per_proc[rank];
    // This is the offset pointer to the first predecessor
    myoff = send_offset_per_proc[rank];
    d_myedges = d_buffer32 + myoff;
    send_count_per_proc[rank] = 0;

    // Processors for which I need to ask others to provide predecessors
    non_local_count = (send_offset_per_proc[size] > mynverts ) ?
                      (send_offset_per_proc[size] - mynverts) : 0;

    // Gather from all nodes how many predecessors I need to provide
    MPI_Allgather(send_count_per_proc, size, MPI_INT_T, recv_count_all_proc, size, MPI_INT_T, MPI_COMM_WORLD);

    // Processors for which I need to provide others the predecessors
    bfs_count_verts_to_recv(recv_count_all_proc, recv_count_per_proc);
    exclusive_scan(recv_count_per_proc, recv_offset_per_proc, size, size+1);
    recv_count = recv_offset_per_proc[size];
    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_allgather");

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
    print_array(send_count_per_proc, size, fp_bfs, "SEND_COUNT_PER_PROC");
    print_array(recv_count_per_proc, size, fp_bfs, "RECV_COUNT_PER_PROC");
#endif

    START_TIMER(dbg_lvl, start);
    CHECK_SIZE("recv_count", recv_count, "h_recv_size", h_recv_size, __func__);
    //Receive processors for which I need to provide predecessors
    //POST_IRECV();
    senderc= 0;                                 \
    memset(recv_req,0,size*sizeof(MPI_Request)); \
    for (i = 0; i < size; ++i){                 \
        if (recv_count_per_proc[i] > 0){            \
            MPI_Irecv((h_recv_buff + recv_offset_per_proc[i]), \
                  recv_count_per_proc[i], MPI_INT32_T,  \
                  i, RECV_BFS_TAG+rank, MPI_COMM_WORLD, &recv_req[senderc]);    \
                  senderc++;                \
        }                           \
    }

    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_irecv");

    if (mynverts > 0) {
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
       print_device_array(d_pred, nverts,  fp_bfs, "make_bfs_multi: d_pred before bfs_pred_local");
       print_device_array32(d_bitmask_mask, (INT_T)m_nelems,  fp_bfs, "make_bfs_multi: d_mask before bfs_pred_local");
       print_device_array32(d_myedges, mynverts,  fp_bfs, "make_bfs_multi: d_myedges before bfs_pred_local");
       print_device_array32(d_bitmask_pverts, nverts,  fp_bfs, "make_bfs_multi: d_bitmask_pverts before bfs_pred_local");
#endif
        // Update d_pred using local information
       bfs_pred_local(d_myedges, mynverts, d_bitmask, d_pred);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
       print_device_array(d_pred, nverts,  fp_bfs, "make_bfs_multi: d_pred after bfs_pred_local");
#endif
    }

    // I need to ask for predecessors
    START_TIMER(dbg_lvl, start);
    if (non_local_count > 0) {
        CHECK_SIZE("next_nverts", next_nverts,  "h_send_size", h_send_size, __func__);
        //Copy  array from Device to Host using 32bits array
        cudaMemcpy(h_send_buff, d_buffer32, next_nverts*sizeof(INT32_T), cudaMemcpyDeviceToHost);
        checkCUDAError("make_bfs_multi: d_buffer32->h_send_buff");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
        print_array_32t(h_send_buff, next_nverts, fp_bfs, "H_SEND_BUFF Vertex with missing PRED");
#endif
    }
    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cpy_send_buff");

    START_TIMER(dbg_lvl, start);
    //Send vertices for which I need predecessors
    //POST_SEND(send_count_per_proc);
    for (i = 0; i < size; ++i){                 \
        if (send_count_per_proc[i] > 0){            \
            MPI_Send((h_send_buff + send_offset_per_proc[i]), \
                    send_count_per_proc[i], MPI_INT32_T, i, \
                 RECV_BFS_TAG+i, MPI_COMM_WORLD );          \
        }                           \
    }

    //Complete receive vertices for which I need to provide predecessors
    //WAIT_IRECV();
    MPI_Waitall(senderc, recv_req, MPI_STATUSES_IGNORE);
    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_send_wait");

    INT_T *h_tmp_count, *h_tmp_offset; // Support variable needed to swap send/receive counts and offsets

    //If I need to provide predecessors to other nodes
    if (recv_count > 0) {
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
        print_array_32t(h_recv_buff, recv_count, fp_bfs, "H_RECV_BUFF");
#endif
        CHECK_SIZE("recv_count", recv_count, "d_recv_size", d_recv_size, __func__);
        //Received vertex for which I need to provide predecessors
        cudaMemcpy(d_buffer32, h_recv_buff, recv_count*sizeof(INT32_T), cudaMemcpyHostToDevice);
        checkCUDAError("make_bfs_multi: h_recv_buff->d_buffer32");
        cudaMemcpy(d_recv_offset_per_proc, recv_offset_per_proc, (size+1)*sizeof(INT_T), cudaMemcpyHostToDevice);
        checkCUDAError("make_bfs_multi: recv_offset_per_proc->d_recv_offset_per_proc");
    }

    // Swap offsets and counts between send and receive
    h_tmp_offset = recv_offset_per_proc; recv_offset_per_proc = send_offset_per_proc; send_offset_per_proc=h_tmp_offset;
    h_tmp_count = recv_count_per_proc; recv_count_per_proc = send_count_per_proc; send_count_per_proc=h_tmp_count;

    // Prepare to receive predecessors I need
    senderc= 0;                                 \
    memset(recv_req,0,size*sizeof(MPI_Request)); \
    for (i = 0; i < size; ++i){                 \
        if (recv_count_per_proc[i] > 0){            \
            MPI_Irecv((h_recv_buff + recv_offset_per_proc[i]), \
                  recv_count_per_proc[i], MPI_INT32_T,  \
                  i, RECV_BFS_TAG+rank, MPI_COMM_WORLD, &recv_req[senderc]);    \
                  senderc++;                \
        }                           \
    }

    // Calculate predecessors requested by other nodes
    START_TIMER(dbg_lvl, start);
    if (recv_count > 0) {
        max_recv_vertex = (max_recv_vertex < recv_count ? recv_count : max_recv_vertex);
        // Prepare predecessors requested by other nodes and put them into d_buffer32
        bfs_pred_recv(d_buffer32, recv_count, d_bitmask, d_recv_offset_per_proc);
        // Predecessors which needs to be sent back are copied into host buffer
        cudaMemcpy(h_send_buff, d_buffer32, recv_count*sizeof(INT32_T), cudaMemcpyDeviceToHost);
        checkCUDAError("make_bfs_multi: d_buffer32->h_send_buff");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
        print_array_32t(h_send_buff, recv_count, fp_bfs, "H_SEND_BUFF Predecessors");
        print_array_64t(send_count_per_proc, size, fp_bfs, "send_count_per_proc Predecessors");
        print_array_64t(send_offset_per_proc, size+1, fp_bfs, "send_offset_per_proc Predecessors");
        print_array_64t(recv_count_per_proc, size, fp_bfs, "recv_count_per_proc Predecessors");
        print_array_64t(recv_offset_per_proc, size+1, fp_bfs, "recv_offset_per_proc Predecessors");
#endif
    }
    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "bfs_pred_recv");

    // Send back predecessors
    START_TIMER(dbg_lvl, start);
    //MPI Send using 32bits
    //POST_SEND(send_count_per_proc);     // Send predecessors requested by other nodes
    for (i = 0; i < size; ++i){                 \
        if (send_count_per_proc[i] > 0){            \
            MPI_Send((h_send_buff + send_offset_per_proc[i]), \
                     send_count_per_proc[i], MPI_INT32_T, i,    \
                     RECV_BFS_TAG+i, MPI_COMM_WORLD );          \
        }                           \
    }

    //WAIT_IRECV();    // Receive predecessors requested to other nodes
    MPI_Waitall(senderc, recv_req, MPI_STATUSES_IGNORE);
    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_send_wait");

    START_TIMER(dbg_lvl, start);
    if (non_local_count > 0) {
       cudaMemcpy(d_buffer32, h_recv_buff, next_nverts*sizeof(INT32_T), cudaMemcpyHostToDevice);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
    print_array_32t(h_recv_buff, next_nverts, fp_bfs, "H_RECV_BUFF LAST");
#endif
       // Process received predecessors and update pred array
       bfs_pred_remote(d_buffer32, next_nverts, d_verts_owners, d_vert_ids, d_pred);
    }
    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "bfs_pred_remote");

    if (rank == VERTEX_OWNER(root)) { // This node owns root vertex
        //Copy back root vertex to predecessor
       cudaMemcpy(&d_pred[local_root32], &root, 1*sizeof(int64_t), cudaMemcpyHostToDevice);
       checkCUDAError("make_bfs_multi: pred_root->root");
    }

    MPI_Barrier(MPI_COMM_WORLD); // Wait all nodes to complete

    // Count local visited vertices using d_pred
    COUNT_VISITED(d_mask_1, local_nvisited, "END");
    cudaFree(d_count_per_proc); d_count_per_proc = NULL;
    free(h_count_per_proc); h_count_per_proc = NULL;
    free(send_offset_per_proc); send_offset_per_proc = NULL;
    free(send_count_per_proc); send_count_per_proc = NULL;
    free(recv_count_per_proc); recv_count_per_proc = NULL;
    free(recv_count_all_proc); recv_count_all_proc = NULL;
    free(recv_offset_per_proc); recv_offset_per_proc = NULL;

    return max_bfs_level;
}


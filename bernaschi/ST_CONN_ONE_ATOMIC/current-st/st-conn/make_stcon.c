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
#include "make_stcon.h"
#include "make_bfs_func.h"
#include "make_bfs.h"
#include "make_stcon_func.h"
#include "cputils.h"
#include "gputils.h"
#include "adj_func.h"
#include "cputestfunc.h"
#include "defines.h"
#include "mythrustlib.h"

extern FILE *fp_bfs;
extern FILE *fp_stats;

extern int nthreads, maxblocks;
extern int rank, size;
extern int64_t MaxLabel;
extern int64_t MaxGlobalLabel;
extern int global_scale;
extern double global_edgefactor;
extern int dbg_lvl;
extern int num_bfs_roots;		// The number of bfs to perform
extern size_t current_allocated_size;
extern size_t freed_memory_size;
extern int validation;
extern double d_send_size_factor;
extern double d_recv_size_factor;
extern double d_mask_size_factor;
extern unsigned int green_exe_time;

INT_T stcon_max_recv_vertex = 0;

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

#define INIT_MPI_VAR							\
		MPI_Request recv_req[size];					\
		int senderc=0;

#define POST_IRECV()							\
	senderc= 0;									\
	memset(recv_req,0,size*sizeof(MPI_Request)); \
	for (i = 0; i < size; ++i){					\
		if (recv_count_per_proc[i] > 0){			\
			MPI_Irecv((h_recv_buff + recv_offset_per_proc[i]), \
				  recv_count_per_proc[i], MPI_INT32_T,	\
				  i, RECV_BFS_TAG+rank, MPI_COMM_WORLD, &recv_req[senderc]);	\
				  senderc++;				\
		}							\
	}


#define WAIT_IRECV()							\
    {			\
       MPI_Waitall(senderc, recv_req, MPI_STATUSES_IGNORE);\
    }

#define POST_SEND(COUNT_PER_PROC)							\
	for (i = 0; i < size; ++i){					\
		if (COUNT_PER_PROC[i] > 0){			\
			MPI_Send((h_send_buff + send_offset_per_proc[i]), \
					COUNT_PER_PROC[i], MPI_INT32_T, i,	\
				 RECV_BFS_TAG+i, MPI_COMM_WORLD );			\
		}							\
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

int run_make_stcon(adjlist *dg, adjlist *hg, mask *d_bitmask)
{
	// Vars
	int bfs_root_idx;		// The counter of actual bfs
	int64_t* bfs_roots = NULL;	// The roots of bfs for source node
    int64_t* bfs_roots_t = NULL;     // The roots of bfs for target node
	double* edge_counts = NULL;	// Number of edges visited in each bfs
	//int validation_passed; 		// Flag for validate step
	uint64_t seed1, seed2;		// Seeds to init the random num. gen.

	int validation_passed_one;	// Result of validate function
	int64_t edge_visit_count; 	// The number of visited edges 
	int64_t nglobalverts; 		// Total number of vertices
	csr_graph cg;			// Reference data struct
	int64_t *h_pred = NULL;		// The array of predecessors
	int64_t *h_bfs_pred = NULL; // The array of predecessors for BFS Validation
	INT_T root_s = -1;		// root of st-con (source node for st-conn)
        INT_T root_t = -1;      // root of st-con (target node for st-conn)
        INT_T root = -1;		// root of bfs = st-con matching node
        INT_T nvisited;
        INT_T stcon_nvisited;

	// Timing
	double* bfs_times = NULL;	// Execution time of each bfs
	double bfs_start, bfs_stop;
        double stcon_start, stcon_stop;
        double* stcon_times = NULL; 
        double *stcon_nodes_visited = NULL;



	int stcon_max_level = 0;
        int bfs_max_level = 0;
        int global_mn_found = 0, local_mn_found = 0;
        int mn_to_check = 0;

	double* validate_times = NULL;	// Execution time of validate step 
	double validate_start; 
	double validate_stop; 

	// Device arrays
	int64_t *d_pred = NULL;		// Predecessor array
	INT_T *d_queue = NULL;		// The current queue
	INT_T *d_queue_off = NULL;	// Offset of vertices in the queue
	INT_T *d_queue_deg = NULL;	// Degree of vertices in the queue
	INT_T *d_next_off = NULL;	// Offset to make next level frontier

	INT_T nverts = hg->nverts;	// Number of my vertices
	INT_T nedges = hg->nedges;	// Number of my edges
	
	INT_T *d_send_buff = NULL;	// The buffer for vertices to send 
	INT_T d_send_size = 0;		// Size of the buffer to send
	INT_T d_recv_size = 0;		// Size of the buffer to recv
	INT_T h_send_size = 0;
	INT_T h_recv_size = 0;
	INT_T d_mask_size = 0;			
    if (global_scale <= MIN_GLOBAL_SCALE) {
		d_send_size = 8*nedges;
		d_recv_size = 8*nedges;
		d_mask_size = 8*nedges;
	} else {
		d_send_size = d_send_size_factor * (INT_T)d_bitmask->m_nelems;
		d_recv_size = d_recv_size_factor * (INT_T)d_bitmask->m_nelems;
		d_mask_size = d_mask_size_factor * (INT_T)d_bitmask->m_nelems;
	}
    if (d_mask_size < nverts) d_mask_size = nverts+2; // we need these when running on 1 node

	h_send_size = d_send_size; // We have only vertex (no more predecessors and vertex)
	h_recv_size = d_recv_size;

	//h_send_buff and h_recv_buff: host buffers to send and receive vertices
	INT32_T *h_send_buff = NULL;
	INT32_T *h_recv_buff = NULL;

	INT32_T *d_buffer32 = NULL;  // Buffer used to copy 32bits Local Vertex from/to HOST Memory
	INT32_T *d_recv_buffer32 = NULL;

	INT_T *d_mask_1 = NULL;		// Support arrays
	INT_T *d_mask_2 = NULL;

	INT_T *h_st_rank = NULL; //[ST_RANK_SIZE] = {-1,-1,-1,-1, -1,-1};
        INT_T * h_st_pred_global = NULL;
        INT_T * h_bfs_pred_global = NULL;
        INT32_T * h_nverts_global = NULL;
        INT_T h_pred_global_size = 0;
        INT_T * h_st_rank_all = NULL;
        INT_T * all_mn_found = (INT_T *) malloc (size * sizeof(INT_T));
        int i,j = 0;
        INT32_T jj;
        INT_T local_visited = 0, all_visited = 0; 
        int all_stcon_max_levels = 0;

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
	cudaMalloc((void**)&d_queue_off, (nverts+2)*sizeof(INT_T)); // ATTENZIONE MODIFICATO PER ESTENDERE PER ALTRE INFO
	checkCUDAError("run_make_bfs_multi: malloc d_queue_off");
	cudaMalloc((void**)&d_queue_deg, (nverts+2)*sizeof(INT_T)); // ATTENZIONE MODIFICATO PER ESTENDERE PER ALTRE INFO
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
		print_device_array32(d_bitmask->pedges,d_bitmask->p_nelems, fp_bfs, "run_make_bfs_multi: ST-pedges");
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
        int num_bfs_roots2 = 2*num_bfs_roots;
	bfs_roots = (int64_t*)callmalloc(2*num_bfs_roots*sizeof(int64_t), 
					 "run_make_bfs_multi: bfs_roots");
        //bfs_roots_t = (int64_t*)callmalloc(num_bfs_roots*sizeof(int64_t),
                    //                                 "run_make_bfs_multi: bfs_roots_t");
	edge_counts = (double*)callmalloc(num_bfs_roots*sizeof(double), 
					  "run_make_bfs_multi: edge_counts");
	bfs_times = (double*)callmalloc(num_bfs_roots*sizeof(double), 
					"run_make_bfs_multi: bfs_times");
	validate_times = (double*)callmalloc(num_bfs_roots*sizeof(double), 
					     "run_make_bfs_multi: validate_times");
	h_pred = (int64_t*)callmalloc(nverts*sizeof(int64_t), 
				      "run_make_bfs_multi: h_pred");

	h_bfs_pred = (int64_t*)callmalloc(nverts*sizeof(int64_t),
					      "run_make_bfs_multi: h_pred");

	h_st_rank = (int64_t*)callmalloc(ST_RANK_SIZE*sizeof(int64_t),
		      "run_make_bfs_multi: h_st_rank");

        stcon_times = (double*)callmalloc(num_bfs_roots*sizeof(double), "run_make_bfs_multi: bfs_times");

	int tot = 0;
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
	//find_stcon_roots(&num_bfs_roots, &cg, seed1, seed2, bfs_roots);
	//find_bfs_roots(&num_bfs_roots, &cg, seed2, seed1, bfs_roots_t); //to do add other seed. 

	// Perform bfs
	// Init the validate flag to 1 (not passed)
        validation_passed_one = 1;
	PRINT_SPACE(__func__, fp_bfs, "SCALE", global_scale);
	PRINT_SPACE(__func__, fp_bfs, "nverts", nverts);
	PRINT_SPACE(__func__, fp_bfs, "nedges", nedges);

    NET_INIT();	
	INT32_T *d_mask = d_bitmask->mask;
    INT32_T tmp_empty;
	cudaMemcpy (&tmp_empty, d_mask, 1*sizeof(INT32_T), cudaMemcpyDeviceToHost);
	checkCUDAError("make_bfs_multi: fake cuda Memcpy");
        h_nverts_global = (INT32_T*)callmalloc (size * sizeof(INT32_T),"make_bfs_multi: h_nverts_global");
        h_st_rank_all =  (INT_T*) callmalloc(size*ST_RANK_SIZE*(sizeof(INT_T)),"make_bfs_multi: h_nverts_global");
        INT32_T nverts32 = (INT32_T) nverts;

        find_stcon_roots(&num_bfs_roots2, &cg, hg, seed1, seed2, bfs_roots);

        MPI_Gather(&nverts32, 1, MPI_INT32_T, h_nverts_global, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
        INT32_T * displs = NULL;
        if (rank == 0){
                displs = (INT32_T*)malloc((size+1)*sizeof(INT32_T));
                stcon_nodes_visited = (double*)callmalloc(num_bfs_roots*sizeof(double), "run_make_bfs_multi: stcon_nodes_visited");
                exclusive_scan_INT32(h_nverts_global, displs, size, size+1);
                h_st_pred_global = (INT_T*)malloc(sizeof(INT_T) * displs[size]);
                h_bfs_pred_global =(INT_T*)malloc(sizeof(INT_T) * displs[size]);
                h_pred_global_size = displs[size];
        } 
        LOG_STATS("ST-CON-RUN RED-V BLUE-V DEPTH MN-FOUND TIME(Sec) TEPS\n")

	for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {
                memset(h_st_rank, NO_PREDECESSOR, ST_RANK_SIZE*sizeof(int64_t));
		root_s = bfs_roots[bfs_root_idx];
		root_t = bfs_roots[bfs_root_idx+ num_bfs_roots];
                
                if (rank==0) {
                    fprintf(stdout,"\nSTART %d ST-CON , RED = %"PRI64", BLUE = %"PRI64"\n", bfs_root_idx, root_s, root_t);
                }

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
                fprintf(fp_bfs,"################## START - STCON ################\n");
                if (rank == 0)fprintf(stdout,"################## START - STCON ################\n");
                if (rank == 0)printf("[%d] ROOT source %ld - ROOT target %ld\n",bfs_root_idx, root_s, root_t);
#endif
		// Zero the number of visited vertices 
		stcon_nvisited = 0;

		// Set -1 to all predecessors
		cudaMemset(d_pred, NO_PREDECESSOR, nverts*sizeof(int64_t));
		checkCUDAError("run_run_make_bfs_maskbfs_multi: memset d_pred");
		cudaMemset(d_mask, NO_PREDECESSOR, sizeof(INT32_T) * (INT_T)d_bitmask->m_nelems);
		checkCUDAError("run_make_bfs_mask: memset d_bitmask->mask");

		// Output bfs idx and root
		LOG(dbg_lvl, fp_bfs, "\nRunning ST-CONN %d, root red vertex %"PRI64"\n", bfs_root_idx, root_s)
                LOG(dbg_lvl, fp_bfs, "\nRunning ST-CONN %d, root blue vertex %"PRI64"\n", bfs_root_idx, root_t)
		MPI_Barrier(MPI_COMM_WORLD);
		// make_stcon
                stcon_start = MPI_Wtime();

		stcon_max_level = make_stcon(root_s, root_t, hg, &stcon_nvisited, h_send_buff,
			       h_recv_buff, h_send_size, h_recv_size,
			       dg, d_pred, d_queue, d_queue_off, 
			       d_queue_deg, d_next_off,
			       d_send_buff, d_send_size, d_recv_size,
			       d_mask_1, d_mask_2, d_mask_size, d_buffer32, d_recv_buffer32,
                   d_bitmask, h_pred, h_st_rank, &global_mn_found, &local_mn_found); 
		stcon_stop = MPI_Wtime();
		stcon_times[bfs_root_idx] = stcon_stop - stcon_start;
		PRINT_TIME("run_make_stcon_mask:bfs_time_i", fp_bfs, stcon_times[bfs_root_idx]);
                all_stcon_max_levels = ((all_stcon_max_levels < stcon_max_level) ? stcon_max_level : all_stcon_max_levels);
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	fprintf(fp_bfs,"################## END - STCON ################  LEV %d\n",stcon_max_level);
#endif

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
		print_array_64t(h_pred, nverts, fp_bfs, "run_make_bfs_multi: h_pred END - STCON ");
#endif
                
                 local_visited = 0; all_visited = 0;
                 for ( jj=0; jj<nverts; jj++) {
                        if (h_pred[jj] != NO_PREDECESSOR) local_visited++;
                 }             
                 // Send rank 0 the visited count
                MPI_Reduce(&local_visited, &all_visited, 1, MPI_INT_T, MPI_SUM, 0, MPI_COMM_WORLD);

                if (rank==0) stcon_nodes_visited[bfs_root_idx] = all_visited;
                MPI_Allgather(h_st_rank, ST_RANK_SIZE, MPI_INT_T, h_st_rank_all, ST_RANK_SIZE, MPI_INT_T, MPI_COMM_WORLD);
                // store all matching found with raise condition bfs version the maximum number of matching nodes will be equal to size (number of mpi process) 
                j = 0; // j will be minor than global_mn_found
                tot = 0; 
                for (i = 0; i < size; i++){
                    if (h_st_rank_all[(i*ST_RANK_SIZE)+2] != NO_PREDECESSOR){
                      root = h_st_rank_all[(i*ST_RANK_SIZE)+2];
                      all_mn_found[j] = h_st_rank_all[i*ST_RANK_SIZE+2];
                      j++;
                    }
                }
                tot = j;

                // Calculate number of input edges visited.
                edge_visit_count = 0;
                count_visited_edges(&cg, edge_counts, &edge_visit_count, bfs_root_idx, h_pred);

                if (rank==0) {
                    double teps = edge_counts[bfs_root_idx] / (stcon_times[bfs_root_idx]);
                    LOG(dbg_lvl, fp_bfs, "\nEND ST-CONN %d, root red vertex %"PRI64" - root blue vertex %"PRI64"\n", bfs_root_idx, root_s, root_t)
                    if (tot == 0){
                             fprintf(stdout,"END %d ST-CON: NO MATCHING NODE FOUND\n", bfs_root_idx);
                             LOG_STATS("%d %"PRI64" %"PRI64" %d %d %g %g\n", bfs_root_idx, root_s, root_t,
                                                                       stcon_max_level, tot, stcon_times[bfs_root_idx], teps)
                    } else {

                            LOG_STATS("%d %"PRI64" %"PRI64" %d %d %g %g\n", bfs_root_idx, root_s, root_t,
                                               stcon_max_level, tot, stcon_times[bfs_root_idx], teps)

                            fprintf(stdout,"ST-CON RUN %d: Depth = %d Global Matching Nodes = %ld\n", bfs_root_idx, stcon_max_level, tot);
                    }
                }

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
                                print_array_64t(h_st_rank_all, size*ST_RANK_SIZE, fp_bfs, "run_make_stcon: h_st_rank_all ");
#endif
                LOG(dbg_lvl, fp_bfs, "\nEND ST-CONN %d, root red vertex %"PRI64" - root blue vertex %"PRI64"\n", bfs_root_idx, root_s, root_t)

                if (rank == 0)fprintf(stdout,"END STCON [%d]\n", bfs_root_idx);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
                print_array_64t(h_pred, nverts, fp_bfs, "run_make_bfs_multi: h_pred END - STCON ");
#endif

                MPI_Barrier(MPI_COMM_WORLD);
                mn_to_check = ((validation < tot) ? validation : tot);
                for (j = 0; j < mn_to_check; j++){
                        root = all_mn_found[j];
                        if (tot != 0 && mn_to_check > 0) {
                                nvisited = 0;
                                cudaMemset(d_pred, NO_PREDECESSOR, nverts*sizeof(int64_t));
                                checkCUDAError("run_run_make_bfs_maskbfs_multi: memset d_pred");
                                cudaMemset(d_mask, NO_PREDECESSOR, sizeof(INT32_T) * (INT_T)d_bitmask->m_nelems);
                                checkCUDAError("run_make_bfs_mask: memset d_bitmask->mask");
                                // Output bfs idx and root
                                LOG(dbg_lvl, fp_bfs, "\nRunning BFS with root vertex %"PRI64"\n", root)
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
                                fprintf(stdout,"START BFS[%d] TEST root = %ld\n", bfs_root_idx, root);
#endif
                                MPI_Barrier(MPI_COMM_WORLD);
                                bfs_start = MPI_Wtime();
                                bfs_max_level = make_bfs_mask(root, hg, &nvisited, h_send_buff,
                                                   h_recv_buff, h_send_size, h_recv_size,
                                                   dg, d_pred, d_queue, d_queue_off,
                                                   d_queue_deg, d_next_off,
                                                   d_send_buff, d_send_size, d_recv_size,
                                                   d_mask_1, d_mask_2, d_mask_size, d_buffer32, d_recv_buffer32,
                                                   d_bitmask);
                                bfs_stop = MPI_Wtime(); bfs_times[0] = bfs_stop - bfs_start; 
                                //PRINT_TIME("run_make_bfs_mask:bfs_time_i", fp_bfs, bfs_times[0]);
                                
                                // Copy predecessor array back to host
                                cudaMemcpy(h_bfs_pred, d_pred, nverts*sizeof(int64_t), cudaMemcpyDeviceToHost);
                                checkCUDAError("run_make_bfs_mask: d_pred->h_bfs_pred");
                        #if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
                                print_array_64t(h_bfs_pred, nverts, fp_bfs, "run_make_bfs_multi: h_bfs_pred");
                                print_device_array32(d_bitmask->mask,d_bitmask->m_nelems, fp_bfs, "run_make_bfs_multi: mask");
                        #endif


                        
                                /***** END ST-CON BFS VALIDATION ******/
                                /***** START VALIDATION SECTION *****/
                                MPI_Barrier(MPI_COMM_WORLD);
                                MPI_Gatherv(h_bfs_pred, nverts, MPI_INT_T, h_bfs_pred_global, h_nverts_global, displs, MPI_INT_T, 0, MPI_COMM_WORLD);
                                if (rank == 0){                     
                                    if (validate_stcon_bfs (stcon_max_level, root, root_s, root_t, h_bfs_pred_global, displs) > 0) {
                                                LOG(dbg_lvl, fp_bfs, "\nST-CON validation error at %d execution, root red vertex %"PRI64"\n", bfs_root_idx, root_s)
                                                LOG(dbg_lvl, fp_bfs, "\nST-CON validation error at %d, root blue vertex %"PRI64"\n", bfs_root_idx, root_t)
                                                fprintf(stdout, "ST-CON Validation %d ERROR with Matching node %"PRI64"\n",  j, root);
                                                fprintf(stdout, "seed1=%"PRIu64", seed2=%"PRIu64"\n", seed1, seed2);
                                                //print_stcon_path(stcon_max_level, root, root_s, root_t, h_all_st_pred, st_all_2verts, st_all_2verts_offset);
                                                //print_bfs_path(root, root_s, root_t, h_all_bfs_pred, all_verts_offset); printf("\n\n\t\tSTAMPO il path [ERRORE]\n\n");
                                                validation_passed_one = 0;
                                        }
                                        else fprintf(stdout, "ST-CON Validation %d OK with Matching node %"PRI64"\n", j, root);
                                }
                                MPI_Bcast(&validation_passed_one, 1, MPI_INT, 0, MPI_COMM_WORLD);
                                if (validation_passed_one == 0)break;
                        }//end if 
                }//end for TEST each mn
                if (validation_passed_one == 0) break;
        		/******* END VALIDATION SECTION ******/
        }//END FOR
        /*
         *
         *
         *  PRINT MULTI-NODE Section
         *
         *
         * */

         /*if (size > 1 && global_mn_found > 0){ //&& global_mn_found > 0){
           //alloco
               MPI_Gatherv(h_pred, nverts, MPI_INT_T, h_st_pred_global, h_nverts_global, displs, MPI_INT_T, 0, MPI_COMM_WORLD);
                if (rank == 0){
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0) 
                                        print_array_64t(h_st_pred_global,  h_pred_global_size, fp_bfs, "run_make_bfs_multi: h_all_bfs_pred"); 
#endif
                  //  printf("global_pred_size: %d \n",h_pred_global_size);
                  //  for (i = 0; i < h_pred_global_size; i++)
                  //      printf("%ld ", h_st_pred_global[i]);
                       //libero
               }
               printf("\n");
         
          

         }*/

        // STATISTICS
        //
        if (rank == 0)
        {
                if (validation_passed_one == 0) {
                        fprintf(stdout, "No results printed for invalid run.\n");
                }
                else
                {
                        int i;
                        fprintf(stdout,"GRAPH STATS\n");
                        fprintf(stdout, "global_scale:                   %d\n",   global_scale);
                        fprintf(stdout, "global_edgefactor:              %.2g\n", global_edgefactor);
                        fprintf(stdout, "num_mpi_processes:              %d\n", size);

                        fprintf(stdout,"STCON STATS\n");
                        fprintf(stdout, "N ST-CON:                       %d\n",   num_bfs_roots);
                        fprintf(stdout, "max ST-CON level:               %d\n", all_stcon_max_levels);
                        double stats[s_LAST];

                        LOG_STATS("max-ST-CON-level %d\n", all_stcon_max_levels)

                        get_statistics(stcon_times, num_bfs_roots, stats);
                        fprintf(stdout, "min_time:                       %g s\n", stats[s_minimum]);
                        fprintf(stdout, "max_time:                       %g s\n", stats[s_maximum]);
                        fprintf(stdout, "mean_time:                      %g s\n", stats[s_mean]);
                        fprintf(stdout, "stddev_time:                    %g\n",   stats[s_std]);

                        LOG_STATS("min_time %g s\n", stats[s_minimum])
                        LOG_STATS("max_time %g s\n", stats[s_maximum])
                        LOG_STATS("mean_time %g s\n", stats[s_mean])
                        LOG_STATS("stddev_time %g\n",   stats[s_std])
                        double mean_time = stats[s_mean];
                         
                        double* secs_per_edge = (double*)callmalloc(num_bfs_roots * sizeof(double), "PRINT_RESULTS");
                        for (i = 0; i < num_bfs_roots; ++i) secs_per_edge[i] = 1 / stcon_times[i];
                        get_statistics(secs_per_edge, num_bfs_roots, stats);
                        fprintf(stdout, "min_CONPS:                       %g \n", stats[s_minimum]);
                        fprintf(stdout, "max_CONPS:                       %g \n", stats[s_maximum]);
                        fprintf(stdout, "mean_CONPS:                      %g \n", stats[s_mean]);
                        fprintf(stdout, "stddev_CONPS:                    %g \n",  stats[s_std]);

                        LOG_STATS("min_CONPS                   %g \n", stats[s_minimum])
                        LOG_STATS("max_CONPS                   %g \n", stats[s_maximum])
                        LOG_STATS("mean_CONPS                  %g \n", stats[s_mean])
                        LOG_STATS("stddev_CONPS                %g \n",  stats[s_std])
                        LOG_STATS("mean_ST, mean_time, stddev_ST: %g, %g, %g \n",  stats[s_mean], mean_time, stats[s_std])
                        
                        fprintf(stdout,"-----ST_CON_VISITED_NODES-----\n");
                        get_statistics(stcon_nodes_visited, num_bfs_roots, stats);
                        fprintf(stdout, "min_visited:                       %g \n", stats[s_minimum]);
                        fprintf(stdout, "max_visited:                       %g \n", stats[s_maximum]);
                        fprintf(stdout, "mean_visited:                      %g \n", stats[s_mean]);
                        fprintf(stdout, "stddev_visited:                    %g \n", stats[s_std]);
                        LOG_STATS("-----ST_CON_VISITED_NODES-----\n");
                        LOG_STATS("min_visited, max_visited, mean_visited, stddev_visited: %g, %g, %g, %g\n", stats[s_minimum], stats[s_maximum], stats[s_mean], stats[s_std])

                         
                        free(secs_per_edge); secs_per_edge = NULL;
                        
                }
        }



	NET_FREE();

	// Free
        
	free(bfs_roots);
        free(bfs_roots_t);
	free(bfs_times); 
	free(validate_times); 
	free(edge_counts); edge_counts = NULL;
	free_csr_graph(&cg);
	free(h_pred);
	free(h_bfs_pred);
        free(stcon_times);
        if (rank == 0){
            free(h_st_pred_global);
            free(h_bfs_pred_global);
            free(displs);
        }
        free(h_nverts_global); 
        free(h_st_rank_all);
        free(all_mn_found);

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



/*
 * Passare a make_bfs_mask root_t aka node target e rinominare root come root_s
 * Cambiare il prototipo della funzione
 */


int make_stcon(INT_T root_s, INT_T root_t, adjlist *hg, INT_T *nvisited,
		   INT32_T *h_send_buff, INT32_T *h_recv_buff,
		   INT_T h_send_size, INT_T h_recv_size,
		   adjlist *dg, int64_t *d_pred, INT_T *d_queue, 
		   INT_T * d_queue_off, INT_T *d_queue_deg, 
		   INT_T *d_next_off,
		   INT_T *d_send_buff, INT_T d_send_size, INT_T d_recv_size,
		   INT_T *d_mask_1, INT_T *d_mask_2, INT_T d_mask_size,
		   INT32_T* d_buffer32, INT32_T* d_recv_buffer32,
		   mask *d_bitmask, INT_T *h_pred, INT_T *h_st_rank, int * global_mn_found, int *local_mn_found)
{
	INT_T nverts = hg->nverts;		// The number of my vertices
	INT_T queue_count = 0;			// The number of vertices in the queue
	INT_T new_queue_count = 0;		// The number of vertices in the 
						// next level queue
	INT_T global_new_queue_count = 0;	// Sum of all new_queue_count
        INT_T global_stop = 0;
        INT_T local_stop = 0;
	INT_T next_level_vertices = 0;		// The number of elements in the next frontier	

	int bfs_level = 0;			// Bfs level
	int max_bfs_level = 0;
	INT_T *d_count_per_proc = NULL;		// Number of vertices per procs on device
	INT_T *h_count_per_proc = NULL;		// Number of vertices per procs on host
	INT_T *send_count_per_proc = NULL;	

	INT_T *send_offset_per_proc = NULL;	
	INT_T *recv_count_all_proc = NULL;	
	INT_T *recv_count_per_proc = NULL;
	INT_T *recv_offset_per_proc = NULL;

    //INT_T h_st[3] = {-1,-1,-1};// host

	INT_T *d_recv_offset_per_proc = NULL; //Receive buffer offset per procs on device

	// Support vars
	int i;
	INT_T myoff;
	INT_T mynverts;
	INT32_T *d_myedges;
	INT_T recv_count;
	INT_T non_local_count;
	INT_T *d_q_1=NULL, *d_q_2=NULL;
	/*Device arrays for peterson's algorithm*/

	INT_T * d_st_rank =NULL; //st_rank[0]: red_pred_rank; st_rank[1]: blue_pred_rank; st_rank[2] st_rank[3] matching node; st_rank[4] = pred_red; st_rank[5] = pred_blue
	double start=0, stop=0, t=0;
	START_TIMER(dbg_lvl, start);
	cudaMalloc((void**)&d_count_per_proc, 2*size*sizeof(INT_T)); 
	checkCUDAError("make_bfs: cudaMalloc d_count_per_proc");

	cudaMalloc((void**)&d_recv_offset_per_proc, (size+1)*sizeof(INT_T));
	checkCUDAError("make_bfs: cudaMalloc d_recv_offset_per_proc");
	cudaMalloc((void**)&d_st_rank, ST_RANK_SIZE*sizeof(INT_T));
	checkCUDAError("make_bfs_mask: cudamalloc d_st_rank");

	cudaMemset(d_st_rank, NO_PREDECESSOR, ST_RANK_SIZE*sizeof(INT_T)); // set default value
        cudaMemset(d_queue, NO_PREDECESSOR, sizeof(INT_T)*2*nverts);
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
	INT32_T label_s;
        INT32_T label_t;
	INT_T m_nelems = (INT_T)(d_bitmask->m_nelems);
	INT32_T local_root_s32;
        INT32_T local_root_t32;
	INT32_T *h_pverts = NULL;
        INT32_T *h_mask = NULL;
        INT_T *h_verts_owners = (INT_T*)(malloc((nverts+2)*sizeof(INT_T)));
        INT_T *h_vert_ids = (INT_T*)(malloc((nverts+2)*sizeof(INT_T)));
        h_pverts = (INT32_T*)malloc(nverts*sizeof(INT32_T)); //nverts
        h_mask = (INT32_T*)malloc((m_nelems+1)*sizeof(INT32_T)); //nverts

	// Number of visited vertex calculated counting d_pred array elements

/*
 *
 *  roots enqueue
 *  Some consideration. If (rank == vertx_owner(s) && rank == vertx_owner(t) ) enqueue both roots in the same node.
 *  Otherwise replace the previous code for each root.
 *  To mark root_s with red mask and root_t blue mask
 *
 *
 * */
	START_TIMER(dbg_lvl, start);
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
		fprintf(stdout ,"root_s: %ld root_t %ld\n", root_s, root_t);
#endif

	if ((VERTEX_OWNER(root_s) == VERTEX_OWNER(root_t)) && rank == VERTEX_OWNER(root_t)) {

		INT_T pred_root_s;
		INT_T pred_root_t;
		INT_T root_st[2] = {root_s, root_t};

		// Use color_mask for 32bit data. D_queue is used as 32bit data.
		INT_T lroot_st[2] = {VERTEX_LOCAL(root_s), (VERTEX_LOCAL(root_t) | COLOR_MASK_64) };
		local_root_s32 = (INT32_T)lroot_st[0];
		local_root_t32 = (INT32_T)(lroot_st[1] & (~COLOR_MASK_64)); //gli tolgo il colore e lo casto

		cudaMemcpy(d_queue, &lroot_st[0], 2*sizeof(INT_T), cudaMemcpyHostToDevice);
		cudaMemcpy (&label_s, &d_bitmask_pverts[local_root_s32], 1*sizeof(INT32_T), cudaMemcpyDeviceToHost);
		cudaMemcpy (&label_t, &d_bitmask_pverts[local_root_t32], 1*sizeof(INT32_T), cudaMemcpyDeviceToHost);
		checkCUDAError("make_bfs_multi: d_bitmask_pverts->label");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
		fprintf(fp_bfs,"root_s: %ld root_t %ld\n", root_st[0], root_st[1]);
		fprintf(fp_bfs,"lroot_s =  %ld lroot_t %ld \n", lroot_st[0], lroot_st[1]);

		fprintf(fp_bfs, "LABEL_s value: %d\n", label_s);
		fprintf(fp_bfs, "LABEL_t value: %d\n", label_t);
#endif

		if (label_s != NO_CONNECTIONS) {
			cudaMemcpy(&d_bitmask_mask[label_s], &local_root_s32, 1*sizeof(INT32_T), cudaMemcpyHostToDevice); //Update bitmask
			checkCUDAError("make_bfs_multi: local_root_s->d_mask");
			pred_root_s = rank;
		}
		else{
			fprintf(stdout,"[rank %d] WARNING ROOT VERTEX NO LOCAL CONNECTIONS!!!!", rank);
			pred_root_s = ROOT_VERTEX;
			//pred_root_t = ROOT_VERTEX; dipende!!! Va messo qua solamente se i roots vanno processati nello stesso nodo.
		}

		if (label_t != NO_CONNECTIONS) {
			local_root_t32 = local_root_t32 | COLOR_MASK;
			cudaMemcpy(&d_bitmask_mask[label_t], &local_root_t32, 1*sizeof(INT32_T), cudaMemcpyHostToDevice); //Update bitmask
			checkCUDAError("make_bfs_multi: local_root_t->d_mask");
                        local_root_t32 = local_root_t32 &(~ COLOR_MASK);
			pred_root_t = rank | COLOR_MASK_64;
		}
		else{
			fprintf(stdout,"[rank %d] WARNING ROOT VERTEX NO LOCAL CONNECTIONS!!!!", rank);
			pred_root_t = ROOT_VERTEX;
			//pred_root_t = ROOT_VERTEX; dipende!!! Va messo qua solamente se i roots vanno processati nello stesso nodo.
		}
		cudaMemcpy(&d_pred[local_root_s32], &pred_root_s, 1*sizeof(int64_t), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_pred[local_root_t32], &pred_root_t, 1*sizeof(int64_t), cudaMemcpyHostToDevice);
		checkCUDAError("make_bfs_multi: pred_root->root");
		queue_count = 2;
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
		print_device_array(d_queue, queue_count, fp_bfs, "D_QUEUE");
#endif
	}

	/*MULTI-NODE*/
	else{
		/*RED ENQUEUE no changes respect to BFS*/
		if (rank == VERTEX_OWNER(root_s)) {
		//Vertex in d_queue are LOCAL
		//Enqueue red root
			INT_T lroot_s = VERTEX_LOCAL(root_s);  // Get root vertex in local value
			local_root_s32 = (INT32_T)lroot_s;     // copy into a 32bit
			INT_T pred_root_s;
			//viene messo in coda un vertice a 64bit quindi serve la maschera a 64.
			cudaMemcpy(d_queue, &lroot_s, 1*sizeof(INT_T), cudaMemcpyHostToDevice); // Enqueue root vertex
			checkCUDAError("make_bfs_multi: root_s->d_queue");
			///portare il colore su d_pred => sul rank
			//Vertex in d_pred are GLOBAL. Indeed I need to use MASK_COLOR_64
			cudaMemcpy (&label_s, &d_bitmask_pverts[local_root_s32], 1*sizeof(INT32_T), cudaMemcpyDeviceToHost);
			checkCUDAError("make_bfs_multi: d_bitmask_pverts->label");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
			fprintf(fp_bfs, "INFO: root_s %ld local_root_s %ld\n", root_s, lroot_s);
			fprintf(fp_bfs, "LABEL_s value: %d\n", label_s); // mi dice se ho connessioni quindi devo averlo anche per il root_t
#endif
			if (label_s != NO_CONNECTIONS) {
			   cudaMemcpy(&d_bitmask_mask[label_s], &local_root_s32, 1*sizeof(INT32_T), cudaMemcpyHostToDevice); //Update bitmask
			   checkCUDAError("make_bfs_multi: local_root_s->d_mask");
			   pred_root_s = rank;
			} else {
					fprintf(stdout,"[rank %d] WARNING ROOT VERTEX NO LOCAL CONNECTIONS!!!!", rank);
					pred_root_s = ROOT_VERTEX;
					//pred_root_t = ROOT_VERTEX; dipende!!! Va messo qua solamente se i roots vanno processati nello stesso nodo.
			}

			cudaMemcpy(&d_pred[local_root_s32], &pred_root_s, 1*sizeof(int64_t), cudaMemcpyHostToDevice);
			checkCUDAError("make_bfs_multi: pred_root->root");
			queue_count++;
		}
		//enqueue BLUE root
		else if (rank == VERTEX_OWNER(root_t)) {
		//Vertex in d_queue are LOCAL
			INT_T lroot_t = VERTEX_LOCAL(root_t);  // Get root vertex in local value
			local_root_t32 = (INT32_T)(lroot_t);     // copy into a 32bit without color
			lroot_t = lroot_t | COLOR_MASK_64;
			//local_root_t32 = local_root_t32 | COLOR_MASK; // indirizzamento  non va la maschera
			INT_T pred_root_t;
			//viene messo in coda un vertice a 64bit quindi serve la maschera a 64.
			cudaMemcpy(d_queue, &lroot_t, 1*sizeof(INT_T), cudaMemcpyHostToDevice); // Enqueue root vertex
			checkCUDAError("make_bfs_multi: root_s->d_queue");
			//Vertex in d_pred are GLOBAL. Indeed I need to use MASK_COLOR_64
			cudaMemcpy (&label_t, &d_bitmask_pverts[local_root_t32], 1*sizeof(INT32_T), cudaMemcpyDeviceToHost);
			checkCUDAError("make_bfs_multi: d_bitmask_pverts->label");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
			fprintf(fp_bfs,"LABEL_t value: %d\n", label_t); // mi dice se ho connessioni quindi devo averlo anche per il root_t
			fprintf(fp_bfs, "INFO: root_t %ld vertex_local %ld\n", root_t, VERTEX_LOCAL(root_t) );
#endif
			if (label_t != NO_CONNECTIONS) {
				local_root_t32 = local_root_t32 | COLOR_MASK;
				cudaMemcpy(&d_bitmask_mask[label_t], &local_root_t32, 1*sizeof(INT32_T), cudaMemcpyHostToDevice); //Update bitmask
				checkCUDAError("make_bfs_multi: local_root_t->d_mask");
                                local_root_t32 = local_root_t32 & ( ~COLOR_MASK); 
				pred_root_t = rank | COLOR_MASK_64;
			} else {
				fprintf(stdout,"[rank %d] WARNING ROOT VERTEX NO LOCAL CONNECTIONS!!!!", rank);
				pred_root_t = ROOT_VERTEX;
			}

			cudaMemcpy(&d_pred[local_root_t32], &pred_root_t, 1*sizeof(int64_t), cudaMemcpyHostToDevice);
			checkCUDAError("make_bfs_multi: pred_root->root");
			queue_count++;
		}
	}
	STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cpy_root");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
        print_device_array(d_queue, queue_count, fp_bfs, "D_QUEUE AFTER ROOTs ENQUEUE #####");
//        print_device_array32(d_bitmask_mask, (INT32_T)d_bitmask-> m_nelems,  fp_bfs, "d_mask LA ODIO");
//        print_device_array32(&d_bitmask_mask[label_s], (INT_T)1,  fp_bfs, "DMASK di LABEL_s DOPO ENQUEUE");
#endif
 	INIT_MPI_VAR;  // Init variables used in all MPI communicationsi
	while(1) {
		LOG(dbg_lvl, fp_bfs, "\nTIME SPACE:*** *** bfs_level_start *** ***\n");
		PRINT_SPACE(__func__, fp_bfs, "bfs_level", bfs_level);
		bfs_level += 1;
		if (queue_count > 0) {
	        /*  Step A of the paper */
			nvisited[0] += queue_count;
			CHECK_SIZE("queue_count", queue_count, "nverts", nverts, __func__);
			// Vertices in d_queue are LOCAL and colored
			stcon_make_queue_deg(d_queue, queue_count, d_queue_off, d_queue_deg, dg->offset, nverts);
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            print_device_array(d_queue_deg, queue_count, fp_bfs, "make_bfs_multi: d_queue_deg");
#endif
			make_queue_offset(d_queue_deg, queue_count, d_next_off, &next_level_vertices);
	        /* In the beginning, with only the root, queue_count is one and next_level_vertices is the number of root's neighbors. */
	        /* d_queue_off contains the offset of the root in the adjency list. d_next_off has a single element equal to zero? */
		} 
                // always large program flow
		/* Step B of the paper */

		// Vertices in d_queue are LOCAL
		/* Use a mask to track visited vertices */
		if (next_level_vertices > 0) {

		   START_TIMER(dbg_lvl, start);
		   cudaMemset(d_mask_1, VALUE_TO_REMOVE_BY_MASK, m_nelems*sizeof(INT_T));
		   STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cudaMemset d_mask_1");
            
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            print_device_array(d_next_off, queue_count, fp_bfs, "make_bfs_multi: D_NEXTOFF");
            print_device_array(d_queue, queue_count,fp_bfs, "D_QUEUE before binary#####");
            print_device_array32(d_bitmask->mask, d_bitmask->m_nelems, fp_bfs, "make_bfs_multi: d_bitmask->mask BEFORE binary expande large");
#endif

	    stcon_binary_expand_queue_mask_large(next_level_vertices, dg->edges, d_queue, d_next_off, d_queue_off,
			                                       queue_count, d_mask_1, d_bitmask,
			                                       d_st_rank, d_pred);

	    next_level_vertices = m_nelems;

            cudaMemcpy(h_st_rank, d_st_rank, ST_RANK_SIZE*sizeof(INT_T), cudaMemcpyDeviceToHost);
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            print_array_64t(h_st_rank, ST_RANK_SIZE, fp_bfs, "make_bfs_multi after binary expande large: h_st_rank");
          	print_device_array32(d_bitmask->mask, d_bitmask->m_nelems, fp_bfs, "make_bfs_multi: d_bitmask->mask AFTER binary expande large");
          	print_device_array(d_mask_1, next_level_vertices,  fp_bfs, "d_mask_1 ST");
          	fprintf(fp_bfs, "d_send_size = %ld\n",d_send_size);
#endif
                if (h_st_rank[3] != NO_PREDECESSOR){
                    local_stop = 1;
                }

	        /* Compact send_array to remove already seen vertices */
			INT_T nelems_removed_by_mask;
			START_TIMER(dbg_lvl, start);
			call_thrust_remove_copy(d_mask_1, next_level_vertices, d_send_buff,  &nelems_removed_by_mask, VALUE_TO_REMOVE_BY_MASK);
			STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "call_thrust_remove_copy");
			next_level_vertices = nelems_removed_by_mask;
	         }

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            print_device_array(d_send_buff, next_level_vertices,  fp_bfs, "d_send_buff ST");
#endif

            MPI_Allreduce(&local_stop, &global_stop, 1, MPI_INT_T, MPI_SUM, MPI_COMM_WORLD);
            if (global_stop > 0){
                break;
            }

            /* Step D of the paper */
            if (next_level_vertices > 0) {
                        CHECK_SIZE("next_level_vertices", next_level_vertices, "size of d_buffer32", d_send_size, __func__);
                        CHECK_SIZE("next_level_vertices", next_level_vertices, "size of d_mask", d_mask_size, __func__);
                        // Calculate owners of Next level vertices stored in d_send_buff
                        stcon_owners(d_send_buff, next_level_vertices, d_mask_1, d_mask_2);
                        sort_owners_bfs(d_mask_1, d_mask_2, next_level_vertices);
                        bfs_count_vertices(d_mask_1, next_level_vertices, d_count_per_proc, send_count_per_proc, h_count_per_proc);

                        // Use exclusive scan to count how many vertices to send to each node
                        exclusive_scan(send_count_per_proc, send_offset_per_proc, size, size+1);
                        // Vertices in d_send_buff are reordered according to d_mask_2, converted into LOCAL 32bit and copied into d_buffer3
                        stcon_back_vertices32(d_send_buff, next_level_vertices, d_mask_2, d_buffer32);
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            print_device_array32(d_buffer32, next_level_vertices,  fp_bfs, "d_buff32 ST");
#endif
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
                    stcon_atomic_enqueue_local(d_myedges, mynverts, d_queue, d_pred, nverts, &new_queue_count);
                    COUNT_VISITED(d_mask_1, (new_queue_count+nvisited[0]), "atomic_enqueue_local");

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
		   print_device_array(d_queue, new_queue_count, fp_bfs, "D_QUEUE after atomic_enqueue_local");
		   print_device_array(d_pred, nverts, fp_bfs, "D_PRED after atomic_enqueue_local");
		   print_device_array32(d_bitmask->mask, d_bitmask->m_nelems, fp_bfs, "make_bfs_multi: mask atomic_enqueue_local");
#endif


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
		POST_SEND(send_count_per_proc); //vedo qui
		WAIT_IRECV();
		STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_send_wait");
		if (recv_count > 0) {
			stcon_max_recv_vertex = (stcon_max_recv_vertex < recv_count ? recv_count : stcon_max_recv_vertex);
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
			stcon_atomic_enqueue_recv(d_recv_buffer32, recv_count, d_recv_offset_per_proc,
					            d_q_1, d_q_2, &new_queue_count, d_pred, nverts, d_bitmask, d_st_rank);

			// d_q_2 contains the new vertices received from other nodes, already in LOCAL form
			COUNT_VISITED(d_mask_1, (new_queue_count+nvisited[0]+queue_count), "atomic_enqueue_recv");

			cudaMemcpy(h_st_rank, d_st_rank, ST_RANK_SIZE*sizeof(INT_T), cudaMemcpyDeviceToHost);
                        if (h_st_rank[3] != NO_PREDECESSOR){
                           local_stop = 1;
                        }
		} 
		START_TIMER(dbg_lvl, start);
		// Wait atomic_enqueue_local
		if (new_queue_count > 0) {
			cudaMemcpy(d_queue + queue_count, d_q_2, new_queue_count*sizeof(INT_T), cudaMemcpyDeviceToDevice);
			queue_count += new_queue_count;
		}
		STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cpy_devTodev");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
		print_device_array(d_queue, queue_count, fp_bfs, "make_bfs_multi: D_QUEUE after atomic_enqueue_recv");
		print_device_array(d_pred, nverts, fp_bfs, "make_bfs_multi: D_PRED atomic_enqueue_recv");
		print_array_64t(h_st_rank, ST_RANK_SIZE, fp_bfs, "make_bfs_multi: h_st_rank atomic_enqueue_recv");
		print_device_array32(d_bitmask->mask, d_bitmask->m_nelems, fp_bfs, "make_bfs_multi: mask atomic_enqueue_recv");
//		print_device_array32(d_bitmask_mask, (INT_T)m_nelems,  fp_bfs, "make_bfs_multi: D_MASK line 986");
//        	print_device_array(d_bitmask_mask, (INT_T)m_nelems,  fp_bfs, "make_bfs_multi: D_MASK line 986");
#endif

		START_TIMER(dbg_lvl, start);
		MPI_Allreduce(&queue_count, &global_new_queue_count, 1, MPI_INT_T, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&local_stop, &global_stop, 1, MPI_INT_T, MPI_SUM, MPI_COMM_WORLD);

		STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_allreduce");
		LOG(dbg_lvl, fp_bfs, "TIME SPACE:*** *** bfs_level_end *** ***\n\n");

		if (global_new_queue_count == 0 || global_stop > 0) {
                        //exit loop while
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

	}  // While(1)

	MPI_Allreduce(MPI_IN_PLACE, nvisited, 1, MPI_INT_T, MPI_SUM, MPI_COMM_WORLD);
	LOG(dbg_lvl, fp_bfs, "nvisited=%"PRI64"\n", nvisited[0]);
	max_bfs_level = ((max_bfs_level < bfs_level) ? bfs_level : max_bfs_level);
	// Graph has been visited but now we need to send/receive predecessors

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	fprintf(fp_bfs, "------------ START Predecessors Exchange -----------------\n");
	//print_device_array(d_pred, nverts, fp_bfs, "make_bfs_multi: D_PRED");
//	print_device_array32(d_bitmask->mask, d_bitmask->m_nelems, fp_bfs, "make_bfs_multi: mask");
#endif

	recv_count = 0;
	non_local_count = 0;
	memset(send_count_per_proc, 0, (size+1)*sizeof(INT_T));
	memset(send_offset_per_proc, 0, (size+1)*sizeof(INT_T));
	memset(recv_count_per_proc, 0, (size+1)*sizeof(INT_T));
	memset(recv_offset_per_proc, 0, (size+1)*sizeof(INT_T));
	memset(h_count_per_proc, 0, (2*size+1)*sizeof(INT_T));
        memset(h_pred, NO_PREDECESSOR, sizeof(INT_T) *nverts);

	INT_T next_nverts = 0; // Vertex for which I need predecessor information (both local and remote)
	INT_T *d_verts_owners = d_queue_deg;  // Array of vertex owners (we reuse an existing device vector)
	INT_T *d_vert_ids     = d_queue_off;  // Array of vertex ids (we reuse an existing device vector)

	// Remove vertex not visited. Copy vertex owner into d_verts_owners and vertex id into d_vert_ids ordered by vertex owner
	// Using d_mask_1 and d_mask_2 as temporary buffers
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
        fprintf(fp_bfs, "------------ Stcon_remove_pred  %d %d-----------------\n", next_nverts, nverts);
        print_device_array(d_verts_owners, next_nverts,  fp_bfs, "make_bfs_multi: d_verts_owners BEFORE stcon_remove_pred");
        print_device_array(d_vert_ids, next_nverts,  fp_bfs, "make_bfs_multi: d_vert_ids BEFORE stcon_remove_pred");
        print_device_array(d_mask_1, next_level_vertices,  fp_bfs, "make_bfs_multi: d_mask_1 BEFORE stcon_remove_pred");
        print_device_array(d_mask_2, next_level_vertices,  fp_bfs, "make_bfs_multi: d_mask_2 BEFORE stcon_remove_pred");
#endif
        stcon_remove_pred(d_pred, d_mask_1, d_mask_2, nverts, d_verts_owners, d_vert_ids, &next_nverts, h_st_rank, local_stop);
if (next_nverts > 0){
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
    fprintf(fp_bfs, "------------ Stcon_remove_pred END-----------------\n");
    print_device_array(d_verts_owners, next_nverts,  fp_bfs, "make_bfs_multi: d_verts_owners AFTER stcon_remove_pred");
    print_device_array(d_vert_ids, next_nverts,  fp_bfs, "make_bfs_multi: d_vert_ids AFTER stcon_remove_pred");
    //print_array_64t(send_offset_per_proc, size+1, fp_bfs, "make_bfs_multi: send_count_per_proc");
#endif

    //d_mask_1 contains the rank holding predecessors info
    //d_mask_2 contains vertex id for which I need predecessors info
    //Count how many predecessors each processor should provide to this node

        bfs_count_vertices(d_verts_owners, next_nverts, d_count_per_proc, send_count_per_proc, h_count_per_proc);
        exclusive_scan(send_count_per_proc, send_offset_per_proc, size, size+1);
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
    //print_device_array(d_mask_1, next_nverts,  fp_bfs, "make_bfs_multi: d_mask_1");
    //print_device_array(d_mask_2, next_nverts,  fp_bfs, "make_bfs_multi: d_mask_2");
    print_array_64t(send_offset_per_proc, size+1, fp_bfs, "make_bfs_multi: send_count_per_proc");
#endif
    // Copy vertex for which I need predecessor into 32 bits buffer
        bfs_copy32(d_vert_ids, next_nverts, d_buffer32);
    }
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
//    printf("Gather from all nodes how many predecessors I need to provide \n");
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
  //  printf("Receive processors for which I need to provide predecessors\n");
    //POST_IRECV();
    senderc= 0;									
    memset(recv_req,0,size*sizeof(MPI_Request)); 
    for (i = 0; i < size; ++i){					
            if (recv_count_per_proc[i] > 0){			
                    MPI_Irecv((h_recv_buff + recv_offset_per_proc[i]), 
                              recv_count_per_proc[i], MPI_INT32_T,	
                              i, RECV_BFS_TAG+rank, MPI_COMM_WORLD, &recv_req[senderc]);	
                              senderc++;				
            }							
    }
    //printf("Receive processors for which I need to provide predecessors END***\n");
    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_irecv");
        // I need to ask for predecessors
        START_TIMER(dbg_lvl, start);
        if (next_nverts > 0) {
            CHECK_SIZE("next_nverts", next_nverts,  "h_send_size", h_send_size, __func__);
            //Copy  array from Device to Host using 32bits array
            cudaMemcpy(h_send_buff, d_buffer32, next_nverts*sizeof(INT32_T), cudaMemcpyDeviceToHost);
            checkCUDAError("make_bfs_multi: d_buffer32->h_send_buff");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
            print_array_32t(h_send_buff, next_nverts, fp_bfs, "H_SEND_BUFF Vertex with missing PRED");
#endif
        }
        STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "cpy_send_buff");
//QUI L'ALTRO SCAMBIO
    if (mynverts > 0) {
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	   //print_device_array(d_pred, nverts,  fp_bfs, "make_bfs_multi: d_pred before bfs_pred_local");
	   print_device_array32(d_bitmask_mask, (INT_T)m_nelems,  fp_bfs, "make_bfs_multi: ################# DMASK before bfs_pred_local");
	   print_device_array32(d_myedges, mynverts,  fp_bfs, "make_bfs_multi: d_myedges before bfs_pred_local");
	   print_device_array32(d_bitmask_pverts, nverts,  fp_bfs, "make_bfs_multi: d_bitmask_pverts before bfs_pred_local");
#endif

       INT32_T *h_myedges;
       h_myedges = h_send_buff + myoff;
	   
       cudaMemcpy(h_pverts, d_bitmask->pverts, nverts*sizeof(INT32_T), cudaMemcpyDeviceToHost);
       checkCUDAError("make_bfs_multi: d_bitmask->pverts->h_pverts");

       cudaMemcpy(h_mask, d_bitmask->mask, (m_nelems)*sizeof(INT32_T), cudaMemcpyDeviceToHost);
       checkCUDAError("make_bfs_multi: d_bitmask->mask->h_mask");
        
       // set h_st_rank in 0 and 1. h_st_rank[0] and [1] will contain the pred of h_st_rank[2] (matching node)
       stcon_pred_local_host(h_myedges, mynverts, h_mask, h_pverts, h_pred, h_st_rank);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	   print_array_64t(h_pred, nverts,  fp_bfs, "make_bfs_multi: h_pred after bfs_pred_local");
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
//	printf("Send vertices for which I need predecessors %d\n", rank);
	//POST_SEND(send_count_per_proc);
	for (i = 0; i < size; ++i){					\
		if (send_count_per_proc[i] > 0){			\
			MPI_Send((h_send_buff + send_offset_per_proc[i]), \
					send_count_per_proc[i], MPI_INT32_T, i,	\
				 RECV_BFS_TAG+i, MPI_COMM_WORLD );			\
		}							\
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
	senderc= 0;									\
	memset(recv_req,0,size*sizeof(MPI_Request)); 
	for (i = 0; i < size; ++i){					\
		if (recv_count_per_proc[i] > 0){			\
			MPI_Irecv((h_recv_buff + recv_offset_per_proc[i]), \
				  recv_count_per_proc[i], MPI_INT32_T,	\
				  i, RECV_BFS_TAG+rank, MPI_COMM_WORLD, &recv_req[senderc]);	\
				  senderc++;				\
		}							\
	}

	// Calculate predecessors requested by other nodes
	START_TIMER(dbg_lvl, start);
	if (recv_count > 0) {
		stcon_max_recv_vertex = (stcon_max_recv_vertex < recv_count ? recv_count : stcon_max_recv_vertex);
        // Prepare predecessors requested by other nodes and put them into d_buffer32  //IMPORTANTE STconn
        //
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
		print_device_array32(d_buffer32, recv_count, fp_bfs, "D_SEND_BUFF before PRED_RECV");
		print_device_array(d_bitmask->unique_edges, d_bitmask->m_nelems, fp_bfs, "UNIQUE_EDGES before PRED_RECV");
        print_device_array32(d_bitmask_mask, (INT_T)m_nelems,  fp_bfs, "################## DMASK before PRED_RECV");
#endif
	
		stcon_pred_recv(d_buffer32, recv_count, d_bitmask, d_recv_offset_per_proc);
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
    STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "stcon_pred_recv");

      // Send back predecessors
	START_TIMER(dbg_lvl, start);
	//MPI Send using 32bits
	//POST_SEND(send_count_per_proc);     // Send predecessors requested by other nodes
	for (i = 0; i < size; ++i){					\
		if (send_count_per_proc[i] > 0){			\
			MPI_Send((h_send_buff + send_offset_per_proc[i]), \
		  			 send_count_per_proc[i], MPI_INT32_T, i,	\
					 RECV_BFS_TAG+i, MPI_COMM_WORLD );			\
		}							\
	}

	//WAIT_IRECV();    // Receive predecessors requested to other nodes
	MPI_Waitall(senderc, recv_req, MPI_STATUSES_IGNORE);
	STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "mpi_send_wait");

	START_TIMER(dbg_lvl, start);
	if (non_local_count > 0) {

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	print_array_32t(h_recv_buff, next_nverts, fp_bfs, "H_RECV_BUFF LAST");
#endif
       // Process received predecessors and update pred array  da vedere per ST-conn lavorare in host copiare tutti tranne d_bueffer32 d_pred
       // h_verts_owners e h_vert_id
       cudaMemcpy(h_verts_owners, d_verts_owners, (nverts+2)*sizeof(INT_T), cudaMemcpyDeviceToHost);
       cudaMemcpy(h_vert_ids, d_vert_ids, (nverts+2)*sizeof(INT_T), cudaMemcpyDeviceToHost);

       stcon_pred_remote_host(h_recv_buff, next_nverts, h_verts_owners, h_vert_ids, h_pred, h_st_rank);
            
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	   print_array_64t(h_pred, nverts,  fp_bfs, "make_bfs_multi: h_pred after stcon_pred_remote_host");
#endif
            
	}
	STOP_TIMER(dbg_lvl, start, stop, t, fp_bfs, "bfs_pred_remote");

	if (rank == VERTEX_OWNER(root_s)) { // This node owns root vertex
		//Copy back root vertex to predecessor
                //VEDERE ST_alternativa
		h_pred[local_root_s32] = root_s;
 	}

	if (rank == VERTEX_OWNER(root_t)) { // This node owns root vertex
		//Copy back root vertex to predecessor local_root_t32 is not colored
		h_pred[local_root_t32 & (~COLOR_MASK)] = root_t;
 	}

	MPI_Barrier(MPI_COMM_WORLD); // Wait all nodes to complete
        global_mn_found[0] = (int)global_stop; 
        local_mn_found[0] = local_stop;
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)

	if (h_st_rank[2] != NO_PREDECESSOR) {
		printf("##################################\n");
		printf("RANK(%d) FOUND MATCHING NODE!!\n"
				" RED ROOT NODE = %ld\n"
				" BLUE ROOT NODE = %ld\n"
				" MATCHING NODE = %ld \n"
				"RED PATH PREDECESSOR = %ld \n"
				"BLUE PATH PREDECESSOR = %ld \n"
                                "NUM OF MNs = %ld \n"
				,rank,
				root_s, root_t
				, h_st_rank[2], h_st_rank[4], h_st_rank[5]  ,global_stop);

		printf("##################################\n");
	}
#endif
	// Count local visited vertices using d_pred
    
	//COUNT_VISITED(d_mask_1, local_nvisited, "END");

	cudaFree(d_count_per_proc); d_count_per_proc = NULL;
        cudaFree(d_recv_offset_per_proc); d_recv_offset_per_proc = NULL;
        cudaFree(d_st_rank);
        free(h_pverts);
        free(h_mask);
	free(h_count_per_proc); h_count_per_proc = NULL;
	free(send_offset_per_proc); send_offset_per_proc = NULL;
	free(send_count_per_proc); send_count_per_proc = NULL;
	free(recv_count_per_proc); recv_count_per_proc = NULL;
	free(recv_count_all_proc); recv_count_all_proc = NULL;
	free(recv_offset_per_proc); recv_offset_per_proc = NULL;
        free(h_verts_owners);
        free(h_vert_ids);
        
	return max_bfs_level;
}





void print_stcon_path (int max_level, INT_T root_s, INT_T root_t, INT_T mn, INT_T red_pred, INT_T blue_pred, INT_T *h_pred) {

        INT_T V, predV;
        predV = root_t;

    red_pred  = red_pred  & (~COLOR_MASK_64);

    fprintf(stdout, "<- ********** ST-CON Result ********* ->\n");
    fprintf (stdout, "MAX DEPTH = %d\n",max_level);
    fprintf (stdout, "ROOT RED VERTEX = %ld\n",root_s);
    fprintf (stdout, "ROOT BLUE VERTEX = %ld\n",root_t);

    fprintf(stdout,"\nRED PATH = %ld -> ", mn);
    int i = 0;
    predV =  red_pred  & (~COLOR_MASK_64);
        do {
                fprintf(stdout, " %ld -> ", predV);
                V = predV;
                predV = h_pred[V] & (~COLOR_MASK_64);
                i++;
        } while (V != predV);
        fprintf(stdout," (DEPTH = %d)\n", i);

    fprintf(stdout,"BLUE PATH = %ld -> ", mn);
    predV =  blue_pred  & (~COLOR_MASK_64);
    i = 0;
        do {
                fprintf(stdout, " %ld -> ", predV);
                V = predV;
                predV = h_pred[V] & (~COLOR_MASK_64);
                i++;
        } while (V != predV);
        fprintf(stdout," (DEPTH = %d)\n", i);

        fprintf(stdout, "<- ********** ST-CON Result DONE ********* ->\n");

}

void print_bfs_path (INT_T root_s, INT_T root_t, INT_T mn, INT_T *h_pred) {

        fprintf(stdout, "<- ********** BFS Result ********* ->\n");

        int i = 0;
        INT_T V, predV;

        // Blue Path
        predV = root_t;
        fprintf(stdout, "BLUE PATH =  ");

        do {
                fprintf(stdout, " %ld -> ", predV);
                V = predV;
                predV = h_pred[V];
                i++;
        } while (V != predV);

        // Red Path
        predV = root_s;
        fprintf(stdout, "\nRED PATH =  ");
        int j = 0;
        do {
                fprintf(stdout, " %ld -> ", predV);
                V = predV;
                predV = h_pred[V];
                j++;

        } while (V != predV);

        fprintf(stdout, "\n MAX LEVEL %d \n", ((j > i ? j : i)));

        fprintf(stdout, "<- ********** BFS Result DONE ********* ->\n");

}


int validate_stcon_bfs (int max_level, INT_T mn, INT_T root_s, INT_T root_t, INT_T *h_all_bfs_pred, INT32_T *all_verts_offset)
{

        int red_depth = -1, blue_depth = -1, err = 0;
        INT_T V, predV, V_local;
        int V_owner, V_offset;

        // Blue Path
        predV = root_t;

        do {
                V = predV;  // GLOBAL VERTEX
                V_owner = VERTEX_OWNER(V);
                V_local = VERTEX_LOCAL(V);

                V_offset = all_verts_offset[V_owner];

                predV = h_all_bfs_pred[V_local + V_offset];    // predV is global

                blue_depth++;
        } while (V != predV);

       // Check that V == matching node and blue_depth <= max_level
        if (V != mn) {
                fprintf(stderr, "WRONG MATCHING NODE: BLUE Path leaf = %ld, Matching Node = %ld\n", V, mn);
                err++;
        }
        if (blue_depth > max_level) {
                fprintf(stderr, "WRONG ST-CON depth on BLUE: ST-CON depth = %d, bfs depth = %d\n", max_level, blue_depth);
                err++;
        }

        // Red Path
        predV = root_s;
        do {
                V = predV;  // GLOBAL VERTEX
                V_owner = VERTEX_OWNER(V);
                V_local = VERTEX_LOCAL(V);

                V_offset = all_verts_offset[V_owner];

                predV = h_all_bfs_pred[V_local + V_offset];    // predV is global

                red_depth++;
        } while (V != predV);

        // Check that V == matching node and blue_depth <= max_level
        if (V != mn) {
                fprintf(stderr, "WRONG MATCHING NODE: RED Path leaf = %ld, Matching Node = %ld\n", V, mn);
                err++;
        }
        if (red_depth > max_level) {
                fprintf(stderr, "WRONG ST-CON depth on RED: ST-CON depth = %d, bfs depth = %d\n", max_level, red_depth);
                err++;
        }

        int max_bfs_depth = ((red_depth > blue_depth) ? red_depth : blue_depth);
        if ( max_bfs_depth != max_level) {
                fprintf(stderr, "WRONG ST-CON MAX depth: ST-CON MAX depth = %d, bfs MAX depth = %d\n", max_level, max_bfs_depth);
                err++;
        }

        return err;
}









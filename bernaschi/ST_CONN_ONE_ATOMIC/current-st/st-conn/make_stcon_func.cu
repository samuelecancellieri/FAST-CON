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
#include <cuda.h>
#include <mpi.h>

#include "header.h"
#include "defines.h"
#include "gputils.h"
#include "cputils.h"
#include "make_struct_gpufunc.h"
#include "make_stcon_func.h"
#include "reference_common.h"
#include "cputestfunc.h"
#include "gputestfunc.h"
#include "cudakernel/mykernels.h"
#include "mythrustlib.h"

extern FILE *fp_bfs;
extern FILE *fp_time;
extern int nthreads, nblocks, maxblocks;
extern int rank, size, lgsize;
extern int dbg_lvl;

// Build the queue offset and queue degree arrays on Device
int stcon_make_queue_deg(INT_T *d_queue, INT_T queue_count,
		   INT_T *d_queue_off, INT_T *d_queue_deg,
		   INT_T *dg_off, INT_T nverts)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);
	CHECK_INPUT(d_queue, queue_count, __func__);
	// Each thread read a vertex in the queue
	int nblocks = MIN((INT_T)maxblocks, (queue_count + (INT_T)nthreads - 1)
		    / (INT_T)nthreads);

	// Kernel
	k_stcon_make_queue_deg<<<nblocks, nthreads>>>(d_queue, queue_count, d_queue_off, d_queue_deg, dg_off, nverts,
			rank, size, lgsize);
	checkCUDAError("make_queue_deg: kernel launch");
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	PRINT_SPACE(__func__, fp_bfs, "queue_count", queue_count);

	// Check queue
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	//check_queue_deg(d_queue, d_queue_deg, queue_count);
#endif
	return 0;
}

int stcon_binary_expand_queue_mask_large(INT_T next_level_vertices, INT_T *dg_edges,
			      INT_T *d_queue, INT_T *d_next_off, 
			      INT_T *d_queue_off, INT_T queue_count, 
			      INT_T *d_send_buff,
			      mask *d_bitmask,
				  INT_T *d_st_rank,
				  INT_T *d_pred)
{
	double tstart=0, tstop, t=0;
	CHECK_INPUT(dg_edges, next_level_vertices, __func__);
	CHECK_INPUT(d_queue, queue_count, __func__);
//  int nblocks = MIN((INT_T)maxblocks, ((next_level_vertices/2) + (INT_T)nthreads - 1)/ (INT_T)nthreads);
	int nblocks = MIN((INT_T)maxblocks, (next_level_vertices + (INT_T)nthreads - 1)/ (INT_T)nthreads);

	INT32_T *d_mask = d_bitmask->mask;
	INT32_T *d_bitmask_pedges = d_bitmask->pedges;
	//INT32_T umask_size = d_bitmask->m_nelems;

	INT_T *d_unique_edges = d_bitmask->unique_edges;

	INT32_T *d_mn_found = 0;
	cudaMalloc((void**)&d_mn_found, 1*sizeof(INT32_T));
	cudaMemset(d_mn_found, 0, 1*sizeof(INT32_T));

	START_TIMER(dbg_lvl, tstart);
	//if (nblocks == 0) nblocks =1;
	k_stcon_binary_mask_unique_large<<<nblocks, nthreads>>>(next_level_vertices, dg_edges,
										  d_queue, d_next_off,
										  d_queue_off, queue_count,
										  d_send_buff,
										  d_bitmask_pedges, d_mask,
										  rank,size,lgsize,
										   d_st_rank, d_pred, d_unique_edges, d_mn_found 
                                                                                  );

/*
	print_device_array(d_queue, queue_count, fp_bfs, "HALF THREADs: CURRENT_QUEUE");
	print_device_array(d_send_buff, d_bitmask->m_nelems, fp_bfs, "HALF BINARY THREADs: NLFS");
	printf("END k_bin\n");
*/
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);

	checkCUDAError("k_binary_mask_unique: kernel");
//	PRINT_SPACE(__func__, fp_bfs, "next_level_vertices", next_level_vertices);
//	PRINT_SPACE(__func__, fp_bfs, "queue_ratio", ((double)queue_count/(double)next_level_vertices));
	PRINT_TIME(__func__, fp_time, t);
        cudaFree(d_mn_found);
	return 0;
}

//  Fill d_mask_1 and d_mask_2:
//  d_mask_1 = VERTEX_OWNER(V)
//  d_mask_2 = gid
int stcon_owners(INT_T* d_sendbuff, INT_T next_level_vertices, INT_T* d_mask_1, INT_T* d_mask_2)
{
	CHECK_INPUT(d_sendbuff, next_level_vertices, __func__);
	CHECK_INPUT(d_mask_1, next_level_vertices, __func__);
	CHECK_INPUT(d_mask_2, next_level_vertices, __func__);

	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	int nblocks = MIN((INT_T)maxblocks, (next_level_vertices + (INT_T)nthreads - 1)/(INT_T)nthreads);

	k_stcon_owner<<<nblocks, nthreads>>>(d_sendbuff, next_level_vertices, d_mask_1, d_mask_2,
					                   rank, size, lgsize);
	checkCUDAError("stcon_owners: k_stcon_owner");
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	//check_owners(d_mask_1, d_mask_2, next_level_vertices, d_sendbuff, __func__, fp_bfs);
#endif
	return 0;
}

int stcon_back_vertices32(INT_T* d_array_u, INT_T nelems, INT_T* d_idx, INT32_T* d_support)
{
	CHECK_INPUT(d_array_u, nelems, __func__);
	CHECK_INPUT32(d_support, nelems, __func__);
	CHECK_INPUT(d_idx, nelems, __func__);

	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	int nblocks;
	nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));
	k_stcon_back_kernel32<<<nblocks, nthreads>>>(d_array_u, nelems, d_idx, d_support, rank, size, lgsize);
	checkCUDAError("stcon_back_vertices32: kernel 1");
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);

	return 0;
}

int stcon_atomic_enqueue_local(INT32_T *d_myedges, INT_T mynverts, INT_T *d_new_queue, int64_t *d_pred, INT_T nverts, INT_T *new_queue_count)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	int nblocks = MIN((INT_T)maxblocks, (mynverts + (INT_T)nthreads - 1) / (INT_T)nthreads);

 	new_queue_count[0] = 0;

	ATOMIC_T *d_qcount, h_qcount = 0;
	cudaMalloc((void**)&d_qcount, 1*sizeof(ATOMIC_T));
	checkCUDAError("atomic_enqueue_local: malloc d_qcount");
	cudaMemset(d_qcount, 0, 1*sizeof(ATOMIC_T));
	checkCUDAError("atomic_enqueue_local: memset d_qcount");

	k_stcon_dequeue_step_8_local<<<nblocks, nthreads>>>(d_myedges, mynverts, d_new_queue, d_pred, d_qcount,
						                          rank, size, lgsize);
  	checkCUDAError("atomic_enqueue_local: k1");

	cudaMemcpy(&h_qcount, d_qcount, 1*sizeof(ATOMIC_T), cudaMemcpyDeviceToHost);
	checkCUDAError("atomic_enqueue_local: d_qcount -> h_qcount");
	new_queue_count[0] = (INT_T)h_qcount;

	cudaFree(d_qcount);
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	PRINT_SPACE(__func__, fp_bfs, "mynverts", mynverts);
	PRINT_SPACE(__func__, fp_bfs, "new_queue_count", new_queue_count[0]);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	assert(new_queue_count[0] <= nverts);
	//bfs_check_new_queue_local(d_new_queue, new_queue_count[0], d_pred, nverts, __func__,fp_bfs);
#endif
	return 0;
}

int stcon_atomic_enqueue_recv(INT32_T *d_recv_buff, INT_T recv_count, INT_T* d_recv_offset_per_proc,
		                INT_T *d_q_1, INT_T *d_q_2, INT_T *new_queue_count, int64_t *d_pred, INT_T nverts, mask *d_bitmask, INT_T * d_st_rank)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	INT32_T *d_pverts = d_bitmask->pverts;
	INT32_T *d_mask = d_bitmask->mask;
	//INT32_T umask_size = d_bitmask->m_nelems;
	int nblocks = MIN((INT_T)maxblocks, (recv_count  + (INT_T)nthreads - 1) / (INT_T)nthreads);

	INT32_T *d_mn_found = 0;
	cudaMalloc((void**)&d_mn_found, 1*sizeof(INT32_T));
	cudaMemset(d_mn_found, 0, 1*sizeof(INT32_T));

	k_stcon_dequeue_step_9_recv_1 <<<nblocks, nthreads>>>(d_recv_buff, recv_count, d_recv_offset_per_proc, d_q_1, d_pred,
			                                              rank, size, lgsize,
			                                               d_st_rank, d_mn_found);
	checkCUDAError("atomic_enqueue_recv: k1");
	
	nblocks = MIN((INT_T)maxblocks, (nverts + (INT_T)nthreads - 1) / (INT_T)nthreads);
 	new_queue_count[0] = 0; 

	ATOMIC_T *d_qcount, h_qcount = 0;
	cudaMalloc((void**)&d_qcount, 1*sizeof(ATOMIC_T));
	checkCUDAError("atomic_enqueue_recv: malloc d_qcount");
	cudaMemset(d_qcount, 0, 1*sizeof(ATOMIC_T));
	checkCUDAError("atomic_enqueue_recv: memset d_qcount");

	k_stcon_dequeue_step_9_recv_2 <<<nblocks, nthreads>>>(d_q_1, d_q_2, d_pred, nverts, d_pverts, d_mask, d_qcount,
			                                        rank, size, lgsize, d_st_rank);
	checkCUDAError("atomic_enqueue_recv: k2");

	cudaMemcpy(&h_qcount, d_qcount, 1*sizeof(ATOMIC_T), cudaMemcpyDeviceToHost);
	checkCUDAError("atomic_enqueue_local: d_qcount -> h_qcount");

	new_queue_count[0] = (INT_T)h_qcount;

	cudaFree(d_qcount);
        cudaFree(d_mn_found);
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	PRINT_SPACE(__func__, fp_bfs, "recv_count", recv_count);
	PRINT_SPACE(__func__, fp_bfs, "new_queue_count", new_queue_count[0]);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	assert(new_queue_count[0] <= nverts);
	//bfs_check_new_queue(d_q_2, new_queue_count[0], d_pred, nverts, __func__,fp_bfs);
#endif
	return 0;
}

int stcon_remove_pred(int64_t *d_pred, INT_T *d_mask_1, INT_T *d_mask_2, INT_T nelems,
		            INT_T *d_out_1, INT_T *d_out_2, INT_T *new_nelems, INT_T *h_st_rank, SHORT_INT flag_add_mn)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);
	int nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));
	k_stcon_remove_pred<<<nblocks, nthreads>>>(d_pred, d_mask_1, d_mask_2, nelems,  rank, size, lgsize);
	checkCUDAError("bfs_remove_pred: k_remove_pred");
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
    print_device_array(d_mask_1, nelems , fp_bfs, "bfs_remove_pred: d_mask_1");
    print_device_array(d_mask_2, nelems , fp_bfs, "bfs_remove_pred: d_mask_2");
#endif
	call_thrust_remove_copy(d_mask_1, nelems, d_out_1, new_nelems, NO_PREDECESSOR);
	call_thrust_remove_copy(d_mask_2, nelems, d_out_2, new_nelems, NO_PREDECESSOR);
/*
	add to d_out_1 rank_1 e rank_2 at position new_nelems and new_nelems+1
	add to d_out_2 two times the matching node
	new_elemes[0] += 2;
*/
    //Leave only those vertices found by other processors and sort by processor
        
	if (flag_add_mn){
                INT_T local_mn[2] = {VERTEX_LOCAL(h_st_rank[2]), VERTEX_LOCAL(h_st_rank[2])};

                //printf("entro qua %d(cpu-id) %ld %ld----- rank %ld %ld\n",rank, local_mn[0], local_mn[1], h_st_rank[0], h_st_rank[1]);
		// Vertices in d_out_2 are LOCAL while vertices in d_st_rank are global!!!
/*	        if ( h_st_rank[0] == -1){
                     h_st_rank[0] = 1;
                }if ( h_st_rank[1] == -1){
                     h_st_rank[1] = 1;
                }*/
                cudaMemcpy(&d_out_2[new_nelems[0]], local_mn, 2*sizeof(INT_T), cudaMemcpyHostToDevice);

		cudaMemcpy(&d_out_1[new_nelems[0]], h_st_rank, 2*sizeof(INT_T), cudaMemcpyHostToDevice);
//                printf("new_nelems %ld\n", new_nelems[0]); 
                new_nelems[0] = new_nelems[0] + 2;
                
	}

	call_thrust_sort_by_key(d_out_1, d_out_2, new_nelems[0]);
         
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	return 0;
}

int stcon_pred_recv(INT32_T *d_buffer32, INT_T nelems, mask *d_bitmask, INT_T* d_recv_offset_per_proc)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	INT32_T *d_pedges = d_bitmask->pedges;
	INT_T *d_unique_edges = d_bitmask->unique_edges;
	INT_T *d_proc_offset = d_bitmask->proc_offset;
	INT32_T *d_mask = d_bitmask->mask;

	int nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));

	k_stcon_pred_recv<<<nblocks, nthreads>>>(d_buffer32, nelems, d_recv_offset_per_proc,
			                               d_unique_edges, d_proc_offset, d_pedges, d_mask,
			                               rank, size, lgsize);
	checkCUDAError("stcon_pred_recv: k_stcon_pred_recv");

	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	return 0;

}

int stcon_pred_local_host(INT32_T *h_buffer32, INT_T nelems, INT32_T *h_mask, INT32_T *h_pverts, INT_T *h_pred, INT_T *h_st_rank)
{
	double tstart=0, tstop, t=0;
	unsigned int gid = 0;

	INT32_T V, label, predVL;
	//INT_T color[2] = {0, COLOR_MASK_64};
	INT32_T predVL_color;
        // Vertices in h_buffer32 are LOCAL while in h_st_rank matching node is global indeed we need to match against local matching node
	INT_T local_mn = VERTEX_LOCAL(h_st_rank[2]);
        short predVL_color_idx;
	START_TIMER(dbg_lvl, tstart);
	for (gid = 0; gid < nelems; gid++) {
		V = h_buffer32[gid];   // LOCAL
		V = V & (~COLOR_MASK); //tolgo il colore perche' uso V come indirizzo
		label = h_pverts[V];
		predVL = h_mask[label]; // Predecessor (local)      su d_mask il colore e' presente
	        predVL_color = predVL & COLOR_MASK;
                predVL_color_idx = (predVL_color == COLOR_MASK);
		predVL = predVL & (~COLOR_MASK); // tolgo il colore

		//h_pred[V] = VERTEX_TO_GLOBAL(predVL) | color[(predVL_color == COLOR_MASK)]; //su di pred il colore lo dobbiamo avere si

		// PER ORA NON METTIAMO IL COLORE
		h_pred[V] = VERTEX_TO_GLOBAL(predVL);
                
		if (local_mn == V) {
                        //printf("h_st_rank[4]= %ld h_st_rank[5] = %ld\n", h_st_rank[4], h_st_rank[5]);
			if ( (h_st_rank[4+predVL_color_idx] == NO_PREDECESSOR)   || (h_st_rank[4+predVL_color_idx] == OTHER_RANK) )  {
				h_st_rank[4+predVL_color_idx] = h_pred[V] & (~COLOR_MASK_64);
			}

		}
	}

        print_array_64t(h_st_rank, ST_RANK_SIZE , fp_bfs, "bfs_pred_local_host: h_st_rank");

	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	return 0;
}

int stcon_pred_remote_host(INT32_T *h_buffer32, INT_T nelems, INT_T *h_mask_1, INT_T *h_mask_2, INT_T *h_pred, INT_T *h_st_rank)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);
	unsigned int gid = 0;
	INT32_T V, predVL;
	INT_T node_rank;
	//IINT_T color[2] = { 0, COLOR_MASK_64 };
	INT32_T predVL_color;
        short predVL_color_idx;
    // Vertices in h_buffer32 are LOCAL while in h_st_rank matching node is global indeed we need to match against local matching node
	INT_T local_mn = VERTEX_LOCAL(h_st_rank[2]);

	for (gid = 0; gid < nelems; gid++) {
		node_rank = h_mask_1[gid];
		//node_rank is without color
		//node_rank = node_rank & (~COLOR_MASK);
		if (node_rank != rank) {
			predVL = h_buffer32[gid]; // qui c'e' il colore lo devo portare anche su d_pre with color
			predVL_color = predVL & COLOR_MASK;
			predVL = predVL & (~COLOR_MASK); //erase color to right use of VERTEX_2_GLOBAL
                        predVL_color_idx = (predVL_color == COLOR_MASK);
			V = (INT32_T) (h_mask_2[gid] & (~COLOR_MASK)); // qui non ricordo d_mask cosa contiene ma V deve essere senza colore perche' indice

			//h_pred[V] = VERTEX_2_GLOBAL(predVL, node_rank) | color[(predVL_color == COLOR_MASK)];
			// PER ORA NON METTIAMO IL COLORE
			h_pred[V] = VERTEX_2_GLOBAL(predVL, node_rank);
                        
			if (local_mn == V) {
				if ((h_st_rank[4+predVL_color_idx] == NO_PREDECESSOR) || (h_st_rank[4+predVL_color_idx] == OTHER_RANK)) {
					h_st_rank[4+predVL_color_idx] = h_pred[V] & (~COLOR_MASK_64);
				}
			}

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
			//fprintf(fp_bfs, "V: %d h_pred[V]: %ld\n", V, h_pred[V]);
#endif
		}
	}

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
    print_array_64t(h_st_rank, ST_RANK_SIZE , fp_bfs, "bfs_pred_local_host: h_st_rank");
#endif
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	return 0;

}


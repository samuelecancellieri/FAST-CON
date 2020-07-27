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
#include "make_bfs_func.h"
#include "reference_common.h"
#include "cputestfunc.h"
#include "gputestfunc.h"
#include "cudakernel/mykernels.h"
#include "mythrustlib.h"

extern FILE *fp_bfs;
extern FILE *fp_time;
extern int nthreads, nblocks, maxblocks;
extern int rank, size, lgsize;
extern int64_t MaxLabel;
extern int64_t MaxGlobalLabel;
extern int dbg_lvl;

// Build the queue offset and queue degree arrays on Device
int make_queue_deg(INT_T *d_queue, INT_T queue_count, 
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
	k_make_queue_deg<<<nblocks, nthreads>>>(d_queue, queue_count, d_queue_off,
						d_queue_deg, dg_off, nverts, rank,
						size, lgsize);
	checkCUDAError("make_queue_deg: kernel launch");
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	PRINT_SPACE(__func__, fp_bfs, "queue_count", queue_count);
	
	// Check queue
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_queue_deg(d_queue, d_queue_deg, queue_count);
#endif
	return 0;
}

int make_queue_offset(INT_T* d_queue_deg, INT_T queue_count, 
		      INT_T *d_next_offset, INT_T *next_level_vertices)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);
	CHECK_INPUT(d_queue_deg, queue_count, __func__);
		
	// make the offset via thrust exclusive scan
	call_thrust_exclusive_scan(d_queue_deg, next_level_vertices, queue_count, d_next_offset);
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_queue_off(d_queue_deg, d_next_offset, queue_count);
#endif
	return 0;
}

int binary_expand_queue_mask_large(INT_T next_level_vertices, INT_T *dg_edges,
			      INT_T *d_queue, INT_T *d_next_off, 
			      INT_T *d_queue_off, INT_T queue_count, 
			      INT_T *d_send_buff,
			      mask *d_bitmask)
{
	double tstart=0, tstop, t=0;
	CHECK_INPUT(dg_edges, next_level_vertices, __func__);
	CHECK_INPUT(d_queue, queue_count, __func__);
        int nblocks = MIN((INT_T)maxblocks, ((next_level_vertices/2) + (INT_T)nthreads - 1)/ (INT_T)nthreads);
        if (nblocks == 0){
            nblocks = 1;
        }
	INT32_T *d_mask = d_bitmask->mask;
	INT32_T *d_bitmask_pedges = d_bitmask->pedges;
        START_TIMER(dbg_lvl, tstart);
	
	k_binary_mask_unique_large<<<nblocks, nthreads>>>(next_level_vertices, dg_edges,
					       d_queue, d_next_off, 
					       d_queue_off, queue_count, 
					       d_send_buff,
					       d_bitmask_pedges, d_mask,
					       rank,size,lgsize);
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);

	checkCUDAError("k_binary_mask_unique: kernel");
	PRINT_SPACE(__func__, fp_bfs, "next_level_vertices", next_level_vertices);
	PRINT_SPACE(__func__, fp_bfs, "queue_ratio", ((double)queue_count/(double)next_level_vertices));
	PRINT_TIME(__func__, fp_time, t);

	return 0;
}

int binary_expand_queue_mask(INT_T next_level_vertices, INT_T *dg_edges, INT_T *d_queue, INT_T *d_next_off,
			                   INT_T *d_queue_off, INT_T queue_count, INT_T *d_send_buff, mask *d_bitmask)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);
	CHECK_INPUT(dg_edges, next_level_vertices, __func__);
	CHECK_INPUT(d_queue, queue_count, __func__);

	int nblocks = MIN((INT_T)maxblocks, (next_level_vertices + (INT_T)nthreads - 1)/ (INT_T)nthreads);
	
	INT32_T *d_mask = d_bitmask->mask;
	INT32_T *d_bitmask_pedges = d_bitmask->pedges;
	
	k_binary_mask_unique<<<nblocks, nthreads>>>(next_level_vertices, dg_edges, d_queue, d_next_off, d_queue_off, queue_count,
						                        d_send_buff, d_bitmask_pedges, d_mask,
						                        rank,size,lgsize);

	checkCUDAError("k_binary_mask_unique: kernel");
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	PRINT_SPACE(__func__, fp_bfs, "next_level_vertices", next_level_vertices);
        PRINT_TIME(__func__, fp_time, t);
	return 0;
}

//  Fill d_mask_1 and d_mask_2:
//  d_mask_1 = VERTEX_OWNER(V)
//  d_mask_2 = gid
int bfs_owners(INT_T* d_sendbuff, INT_T next_level_vertices, INT_T* d_mask_1, INT_T* d_mask_2)
{
	CHECK_INPUT(d_sendbuff, next_level_vertices, __func__);
	CHECK_INPUT(d_mask_1, next_level_vertices, __func__);
	CHECK_INPUT(d_mask_2, next_level_vertices, __func__);

	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	int nblocks = MIN((INT_T)maxblocks, (next_level_vertices + (INT_T)nthreads - 1)/(INT_T)nthreads);
	
	k_bfs_owner<<<nblocks, nthreads>>>(d_sendbuff, next_level_vertices, d_mask_1, d_mask_2,
					                   rank, size, lgsize);
	checkCUDAError("bfs_owners: k_bfs_owner");
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_owners(d_mask_1, d_mask_2, next_level_vertices, d_sendbuff,
							__func__, fp_bfs);	
#endif	
	return 0;
}

// Sort d_mask_1 via thrust, d_mask_2 is a payload.
int sort_owners_bfs(INT_T* d_mask_1, INT_T* d_mask_2, INT_T next_level_vertices)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);
	CHECK_INPUT(d_mask_1, next_level_vertices, __func__);
	CHECK_INPUT(d_mask_2, next_level_vertices, __func__);
	
	// Sort array via thrust
	call_thrust_sort_by_key(d_mask_1, d_mask_2, next_level_vertices);

	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_sort(d_mask_1, next_level_vertices, __func__, fp_bfs);
#endif
	return 0;
}

int bfs_count_vertices(INT_T* d_array, INT_T nelems, INT_T* d_count_per_proc,
		               INT_T* send_count_per_proc, INT_T* host_count_per_proc)
{
	//CHECK_INPUT(d_array, nelems, __func__);
	CHECK_INPUT(d_count_per_proc, 1, __func__);

	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	int nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));
	if (nelems > 1) {
		cudaMemset(d_count_per_proc, 0, 2*size*sizeof(INT_T));
		k_bfs_count_proc<<<nblocks, nthreads>>>(d_array, nelems, d_count_per_proc);
		checkCUDAError("bfs_count_vertices: kernel");
	
		cudaMemcpy(host_count_per_proc, d_count_per_proc, 2*size*sizeof(INT_T), cudaMemcpyDeviceToHost);
		checkCUDAError("bfs_count_vertices: d_count_per_proc -> host_count_per_proc");
		int i;
		for(i=0; i < size; ++i) {
			send_count_per_proc[i] = (host_count_per_proc[2*i+1] - host_count_per_proc[2*i]);
		}

	} else {
		INT_T p0;
		cudaMemcpy(&p0, &d_array[0], 1*sizeof(INT_T), cudaMemcpyDeviceToHost);
		send_count_per_proc[p0] = nelems;
	}
	
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	// Yet to be defined!
	//check_count(d_array, nelems, send_count_per_proc, __func__, fp_bfs);
#endif
	return 0;
}

// Fill d_support with vertices in the d_array_u ordered according to indices in d_idx.
int bfs_back_vertices32(INT_T* d_array_u, INT_T nelems, INT_T* d_idx, INT32_T* d_support)
{
	CHECK_INPUT(d_array_u, nelems, __func__);
	CHECK_INPUT32(d_support, nelems, __func__);
	CHECK_INPUT(d_idx, nelems, __func__);

	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	int nblocks;
	nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));
	bfs_back_kernel32<<<nblocks, nthreads>>>(d_array_u, nelems, d_idx, d_support, rank, size, lgsize);
	checkCUDAError("bfs_back_vertices32: kernel 1");
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);

	return 0;
}

int bfs_count_verts_to_recv(INT_T *recv_count_all_proc, INT_T *recv_count_per_proc)
{
	int i;
	for (i = 0; i < size; ++i)
	{
		int p;
		p = rank + i*size;
		recv_count_per_proc[i] = recv_count_all_proc[p];
		recv_count_per_proc[size] += recv_count_all_proc[p];
	}
	PRINT_SPACE(__func__, fp_bfs, "torecv", (double)recv_count_per_proc[size]);

	return 0;
}

int atomic_enqueue_local(INT32_T *d_myedges, INT_T mynverts, INT_T *d_new_queue, int64_t *d_pred, INT_T nverts, INT_T *new_queue_count)
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

	k_dequeue_step_8_local<<<nblocks, nthreads>>>(d_myedges, mynverts, d_new_queue, d_pred, d_qcount,
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
	bfs_check_new_queue_local(d_new_queue, new_queue_count[0], d_pred, nverts, __func__,fp_bfs);
#endif
	return 0;
}

int atomic_enqueue_recv(INT32_T *d_recv_buff, INT_T recv_count, INT_T* d_recv_offset_per_proc,
		                INT_T *d_q_1, INT_T *d_q_2, INT_T *new_queue_count, int64_t *d_pred, INT_T nverts, mask *d_bitmask)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	INT32_T *d_pverts = d_bitmask->pverts;
	INT32_T *d_mask = d_bitmask->mask;

	int nblocks = MIN((INT_T)maxblocks, (recv_count  + (INT_T)nthreads - 1) / (INT_T)nthreads);

	k_dequeue_step_9_recv_1 <<<nblocks, nthreads>>>(d_recv_buff, recv_count, d_recv_offset_per_proc, d_q_1, d_pred,
			                                        rank, size, lgsize);
	checkCUDAError("atomic_enqueue_recv: k1");
	
	nblocks = MIN((INT_T)maxblocks, (nverts + (INT_T)nthreads - 1) / (INT_T)nthreads);
 	new_queue_count[0] = 0; 

	ATOMIC_T *d_qcount, h_qcount = 0;
	cudaMalloc((void**)&d_qcount, 1*sizeof(ATOMIC_T));
	checkCUDAError("atomic_enqueue_recv: malloc d_qcount");
	cudaMemset(d_qcount, 0, 1*sizeof(ATOMIC_T));
	checkCUDAError("atomic_enqueue_recv: memset d_qcount");

	k_dequeue_step_9_recv_2 <<<nblocks, nthreads>>>(d_q_1, d_q_2, d_pred, nverts, d_pverts, d_mask, d_qcount,
			                                        rank, size, lgsize);
	checkCUDAError("atomic_enqueue_recv: k2");

	cudaMemcpy(&h_qcount, d_qcount, 1*sizeof(ATOMIC_T), cudaMemcpyDeviceToHost);
	checkCUDAError("atomic_enqueue_local: d_qcount -> h_qcount");

	new_queue_count[0] = (INT_T)h_qcount;

	cudaFree(d_qcount);

	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	PRINT_SPACE(__func__, fp_bfs, "recv_count", recv_count);
	PRINT_SPACE(__func__, fp_bfs, "new_queue_count", new_queue_count[0]);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	assert(new_queue_count[0] <= nverts);
	bfs_check_new_queue(d_q_2, new_queue_count[0], d_pred, nverts, __func__,fp_bfs);
#endif
	return 0;
}

// d_new_queue is used to unique vertices in d_myedges, then d_new_queue is compacted to d_queue
// via atomic add
int unique_and_atomic_enqueue_local(INT32_T *d_myedges, INT_T mynverts, INT_T *d_new_queue, INT_T *d_queue,
				                    int64_t *d_pred, INT_T nverts, INT_T *new_queue_count)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	int nblocks = MIN((INT_T)maxblocks, (mynverts + (INT_T)nthreads - 1) / (INT_T)nthreads);
	
	k_unique_local<<<nblocks, nthreads>>>(d_myedges, mynverts, d_new_queue, d_pred,
					                      rank, size, lgsize);
  	checkCUDAError("unique_and_atomic_enqueue_local: k1");

  	nblocks = MIN((INT_T)maxblocks, (nverts + (INT_T)nthreads - 1) / (INT_T)nthreads);

 	new_queue_count[0] = 0; 

	ATOMIC_T *d_qcount, h_qcount = 0;
	cudaMalloc((void**)&d_qcount, 1*sizeof(ATOMIC_T));
	checkCUDAError("unique_and_atomic_enqueue_local: malloc d_qcount");
	cudaMemset(d_qcount, 0, 1*sizeof(ATOMIC_T));
	checkCUDAError("unique_and_atomic_enqueue_local: memset d_qcount");

 	k_atomic_enqueue_local<<<nblocks, nthreads>>>(d_new_queue, d_queue, d_pred, nverts, d_qcount,
						                          rank, size, lgsize);
  	checkCUDAError("unique_and_atomic_enqueue_local: k2");

	cudaMemcpy(&h_qcount, d_qcount, 1*sizeof(ATOMIC_T), cudaMemcpyDeviceToHost);
	checkCUDAError("unique_and_atomic_enqueue_local: d_qcount -> h_qcount");

	new_queue_count[0] = (INT_T)h_qcount;

	cudaFree(d_qcount);
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	PRINT_SPACE(__func__, fp_bfs, "mynverts", mynverts);
	PRINT_SPACE(__func__, fp_bfs, "new_queue_count", new_queue_count[0]);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	assert(new_queue_count[0] <= nverts);
	bfs_check_new_queue_local(d_queue, new_queue_count[0], d_pred, nverts, __func__,fp_bfs);
#endif
	return 0;
}

int bfs_remove_pred(int64_t *d_pred, INT_T *d_mask_1, INT_T *d_mask_2, INT_T nelems,
		            INT_T *d_out_1, INT_T *d_out_2, INT_T *new_nelems)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	int nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));

	k_remove_pred<<<nblocks, nthreads>>>(d_pred, d_mask_1, d_mask_2, nelems,  rank, size, lgsize);
	checkCUDAError("bfs_remove_pred: k_remove_pred");

	call_thrust_remove_copy(d_mask_1, nelems, d_out_1, new_nelems, NO_PREDECESSOR);
	call_thrust_remove_copy(d_mask_2, nelems, d_out_2, new_nelems, NO_PREDECESSOR);

	//Leave only those vertices found by other processors and sort by processor
	call_thrust_sort_by_key(d_out_1, d_out_2, new_nelems[0]);

	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	return 0;
}

int bfs_copy32(INT_T *d_array, INT_T nelems, INT32_T  *d_buffer32)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);
	int nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));
        //printf("--------------->----------------> nelems %ld\n",nelems);
	k_bfs_copy32<<<nblocks, nthreads>>>(d_array, nelems, d_buffer32);
	checkCUDAError("bfs_copy32: k_bfs_copy32");

	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	return 0;

}


int bfs_pred_local(INT32_T *d_buffer32, INT_T nelems, mask *d_bitmask, int64_t *d_pred)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	INT32_T *d_pverts = d_bitmask->pverts;
	INT32_T *d_mask = d_bitmask->mask;

	int nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));

	k_bfs_pred_local<<<nblocks, nthreads>>>(d_buffer32, nelems, d_pverts, d_mask, d_pred,
			                                rank, size, lgsize);
	checkCUDAError("bfs_pred_local: k_bfs_pred_local");

	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	return 0;
}

int bfs_pred_recv(INT32_T *d_buffer32, INT_T nelems, mask *d_bitmask, INT_T* d_recv_offset_per_proc)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);

	INT32_T *d_pedges = d_bitmask->pedges;
	INT_T *d_unique_edges = d_bitmask->unique_edges;
	INT_T *d_proc_offset = d_bitmask->proc_offset;
	INT32_T *d_mask = d_bitmask->mask;

	int nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));

	k_bfs_pred_recv<<<nblocks, nthreads>>>(d_buffer32, nelems, d_recv_offset_per_proc,
			                               d_unique_edges, d_proc_offset, d_pedges, d_mask,
			                               rank, size, lgsize);
	checkCUDAError("bfs_pred_recv: k_bfs_pred_recv");

	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	return 0;

}

int bfs_pred_remote(INT32_T *d_buffer32, INT_T nelems, INT_T *d_mask_1, INT_T *d_mask_2, int64_t *d_pred)
{
	double tstart=0, tstop, t=0;
	START_TIMER(dbg_lvl, tstart);
	int nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));

	k_bfs_pred_remote<<<nblocks, nthreads>>>(d_buffer32, nelems, d_mask_1, d_mask_2, d_pred,
			                                rank, size, lgsize);
	checkCUDAError("bfs_pred_local: k_bfs_pred_remote");

	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
	return 0;

}


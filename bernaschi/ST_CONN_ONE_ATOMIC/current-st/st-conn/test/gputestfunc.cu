#ifndef COMMON_H
/* This is to make PRId64 working with c++ compiler */
#ifdef __cplusplus
#define __STDC_FORMAT_MACROS
#endif
/* header of int64_t and PRId64 */
#include <inttypes.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include "defines.h"
#include "header.h"
#include "mythrustlib.h"
#include "reference_common.h"
#include "cputils.h"
#include "gputils.h"

#include "./gputestfunc.h"

extern FILE *fp_struct;
extern FILE *fp_bfs;
extern int rank, size;
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
extern int lgsize;
#endif
extern int64_t MaxLabel;
extern int64_t MaxGlobalLabel;
extern int global_scale;
extern double global_edgefactor;

void CHECK_LOCAL_VERTEX(INT_T *pointer, INT_T nelems, 
			const char *fcaller, FILE *fout) 
{
	if (pointer == NULL) {
		fprintf(fout, "%s: in %s input array is NULL\n",
			fcaller, __func__);
		fflush(fout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nelems < 0) {
		fprintf(fout, "%s: in %s nelems < 0\n",
			fcaller, __func__);
		fflush(fout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nelems == 0) {
		fprintf(fout, "\n*** WARNING ***\n");
		fprintf(fout, "%s: in %s nelems = 0\n",
			fcaller, __func__);
		fprintf(fout, "*** WARNING ***\n\n");
		fflush(fout);
	}
	INT_T i;
	for (i=0; i < nelems; ++i) {
		if (pointer[i] < 0) {
			fprintf(fout, "%s: in %s,"
				" local vertex = %"PRI64" <= 0\n",
				fcaller, __func__, pointer[i]);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (VERTEX_LOCAL(pointer[i]) > MaxLabel) {
			fprintf(fout, "%s: in %s, pointer[i] >= MaxLabel\n",
				fcaller, __func__);
			fprintf(fout, "%s: in %s, pointer=%"PRI64"\n", 
				fcaller, __func__, pointer[i]);
			fprintf(fout, "%s: in %s, MaxLabel=%"PRId64"\n", 
				fcaller, __func__, MaxLabel);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (rank != VERTEX_OWNER(pointer[i])) {
			fprintf(fout, "%s: in %s, rank != VERTEX_OWNER(pointer[i])\n",
				fcaller, __func__);
			fprintf(fout, "%s: in %s, VERTEX_OWNER(pointer)=%d ", 
				fcaller, __func__, VERTEX_OWNER(pointer[i]));
			fprintf(fout, "%s: in %s, rank=%d\n", 
				fcaller, __func__, rank);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

}
void CHECK_GLOBAL_VERTEX(INT_T *pointer, INT_T nelems, 
			 const char *fcaller, FILE *fout) 
{
	if (pointer == NULL) {
		fprintf(fout, "%s: in %s input array is NULL\n",
			fcaller, __func__);
		fflush(fout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nelems <= 0) {
		fprintf(fout, "%s: in %s nelems = 0\n",
			fcaller, __func__);
		fflush(fout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	INT_T i;
	for (i=0; i < nelems; ++i) {
		if (pointer[i] < 0) {
			fprintf(fout, "%s: in %s,"
				" vertex = %"PRI64" <= 0\n",
				fcaller, __func__, pointer[i]);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (pointer[i] > MaxGlobalLabel) {
			fprintf(fout, "%s: in %s, pointer[i] >= MaxGlobalLabel\n",
				fcaller, __func__);
			fprintf(fout, "%s: in %s, pointer=%"PRI64"\n", 
				fcaller, __func__, pointer[i]);
			fprintf(fout, "%s: in %s, MaxGlobalLabel=%"PRId64"\n", 
				fcaller, __func__, MaxGlobalLabel);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (VERTEX_OWNER(pointer[i]) > size) {
			fprintf(fout, "%s: in %s,  VERTEX_OWNER(pointer[i]) > size\n",
				fcaller, __func__);
			fprintf(fout, "%s: in %s, VERTEX_OWNER(pointer)=%d ", 
				fcaller, __func__, VERTEX_OWNER(pointer[i]));
			fprintf(fout, "%s: in %s, size=%d\n", 
				fcaller, __func__, size);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

}

// Vertex equal to -1 are allowed!
void CHECK_GLOBAL_VERTEX_MINUSONE(INT_T *pointer, INT_T nelems, 
			 	  const char *fcaller, FILE *fout) 
{
	if (pointer == NULL) {
		fprintf(fout, "%s: in %s input array is NULL\n",
			fcaller, __func__);
		fflush(fout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nelems <= 0) {
		fprintf(fout, "%s: in %s nelems = 0\n",
			fcaller, __func__);
		fflush(fout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	INT_T i;
	for (i=0; i < nelems; ++i) {
		if (pointer[i] < 0) {
			if (pointer[i] != -1) {
				fprintf(fout, "%s: in %s,"
					" vertex = %"PRI64" <= 0\n",
					fcaller, __func__, pointer[i]);
				fflush(fout);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
		if (pointer[i] > MaxGlobalLabel) {
			fprintf(fout, "%s: in %s, pointer[i] >= MaxGlobalLabel\n",
				fcaller, __func__);
			fprintf(fout, "%s: in %s, pointer=%"PRI64"\n", 
				fcaller, __func__, pointer[i]);
			fprintf(fout, "%s: in %s, MaxGlobalLabel=%"PRId64"\n", 
				fcaller, __func__, MaxGlobalLabel);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (VERTEX_OWNER(pointer[i]) > size) {
			fprintf(fout, "%s: in %s,  VERTEX_OWNER(pointer[i]) > size\n",
				fcaller, __func__);
			fprintf(fout, "%s: in %s, VERTEX_OWNER(pointer)=%d ", 
				fcaller, __func__, VERTEX_OWNER(pointer[i]));
			fprintf(fout, "%s: in %s, size=%d\n", 
				fcaller, __func__, size);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

}

// *** *** *** BFS check functions *** *** *** /
void check_queue_deg(INT_T *d_queue, INT_T *d_queue_deg, INT_T queue_count)
{
	INT_T *h_queue = NULL;
	INT_T *h_queue_deg = NULL;
	h_queue = (INT_T*)callmalloc(queue_count*sizeof(INT_T), __func__);
	h_queue_deg = (INT_T*)callmalloc(queue_count*sizeof(INT_T), __func__);

	cudaMemcpy(h_queue, d_queue, queue_count*sizeof(INT_T),
		   cudaMemcpyDeviceToHost);
	checkCUDAError("make_queue_deg: d_queue->h_queue");
	cudaMemcpy(h_queue_deg, d_queue_deg, queue_count*sizeof(INT_T),
		   cudaMemcpyDeviceToHost);
	checkCUDAError("make_queue_deg: d_queue_deg->h_queue_deg");

	if (queue_count < PRINT_MAX_NVERTS) {
		print_array(h_queue, queue_count, fp_bfs, "QUEUE");
		print_array(h_queue_deg, queue_count, fp_bfs, "QUEUE_DEGREE");
	}

	//CHECK_LOCAL_VERTEX(h_queue, queue_count, __func__, fp_bfs);
		
	INT_T maxdegree = (INT_T)global_edgefactor * (2<<global_scale);
	INT_T j;
	for (j=0; j < queue_count; ++j) {
		if (h_queue_deg[j] > maxdegree) {
			fprintf(stderr, "ERROR\n");
			fprintf(stderr, "rank %d in %s: Offset greater than edgefactor*2^SCALE at element j=%"PRI64"!\n", 
				rank, __func__, j);
			fprintf(stderr, "h_queue_deg[j]=%"PRI64"\n", h_queue_deg[j]);
			fprintf(stderr, "edgefactor*2^SCALE=%"PRI64"\n", maxdegree);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	free(h_queue); h_queue = NULL;
	free(h_queue_deg); h_queue_deg = NULL;
}

void check_queue_off(INT_T *d_queue_deg, INT_T *d_next_off, INT_T queue_count)
{
	INT_T *h_next_off = NULL;
	INT_T *h_queue_deg = NULL;
	h_next_off = (INT_T*)callmalloc((queue_count+1)*sizeof(INT_T),
		     __func__);
	h_queue_deg = (INT_T*)callmalloc(queue_count*sizeof(INT_T), __func__);
	cudaMemcpy(h_next_off, d_next_off, (queue_count+1)*sizeof(INT_T),
		   cudaMemcpyDeviceToHost);
	checkCUDAError("make_queue_offset: d_next_off->h_next_off");
	cudaMemcpy(h_queue_deg, d_queue_deg, queue_count*sizeof(INT_T),
		   cudaMemcpyDeviceToHost);
	checkCUDAError("make_queue_offset: d_queue_deg->h_queue_deg");

	if (queue_count < PRINT_MAX_NVERTS){
		print_array(h_next_off, queue_count, fp_bfs, "NEXT_OFFSET");
	}
	INT_T i;
	INT_T *appo_next_off = NULL;
	appo_next_off = (INT_T*)callmalloc((queue_count+1)*sizeof(INT_T), 
					   __func__);
	exclusive_scan(h_queue_deg, appo_next_off, queue_count, (queue_count+1)); 

	for (i=0; i < queue_count; ++i) {
		assert(appo_next_off[i] == h_next_off[i]);
	}

	free(appo_next_off); appo_next_off = NULL;
	free(h_next_off); h_next_off = NULL;
	free(h_queue_deg); h_queue_deg = NULL;
}

// After the binary expansion:
// h_send_buff = V (where V is a neighbor of a vertex U in the queue)
// h_recv_buff = predecessor of V
void check_binary_expand(INT_T *d_send_buff, INT_T *d_recv_buff, 
			 INT_T next_level_vertices)
{
	INT_T *h_send_buff = NULL;
	INT_T *h_recv_buff = NULL;
	h_send_buff = (INT_T*)callmalloc(next_level_vertices*sizeof(INT_T),
			__func__);
	h_recv_buff = (INT_T*)callmalloc(next_level_vertices*sizeof(INT_T),
			__func__);
	cudaMemcpy(h_send_buff, d_send_buff,
		   next_level_vertices*sizeof(INT_T),
		   cudaMemcpyDeviceToHost);
	checkCUDAError("binary_expand_queue: d_send_buff->h_send_buff");
	cudaMemcpy(h_recv_buff, d_recv_buff,
		   next_level_vertices*sizeof(INT_T),
		   cudaMemcpyDeviceToHost);
	checkCUDAError("binary_expand_queue: d_recv_buff->h_recv_buff");

	if (next_level_vertices < PRINT_MAX_NEDGES) {
		print_array(h_send_buff, next_level_vertices, 
			    fp_bfs, "check_binary_expand: H_SEND_BUFF");
		print_array(h_recv_buff, next_level_vertices, 
			    fp_bfs, "check_binary_expand: H_RECV_BUFF");
	}
	// V must be less than MaxGlobalLabel
	// predV == U must be less than MaxLabel
	CHECK_LOCAL_VERTEX(h_recv_buff, next_level_vertices, __func__, fp_bfs);
	CHECK_GLOBAL_VERTEX(h_send_buff, next_level_vertices, __func__, fp_bfs);

	free(h_send_buff); h_send_buff = NULL;
	free(h_recv_buff); h_recv_buff = NULL;
}


void check_unique(INT_T *d_array, INT_T nelems,  
		  const char *fcaller, FILE *fout)
{
	INT_T *h_array; h_array = NULL;
	h_array = (INT_T*)callmalloc(nelems*sizeof(INT_T),
				     __func__);
	cudaMemcpy(h_array, d_array, nelems*sizeof(INT_T),
		   cudaMemcpyDeviceToHost);
	checkCUDAError("sort_unique_next_level_frontier: d_array->h_array");

	INT_T j;
	if (nelems < PRINT_MAX_NEDGES){
		print_array(h_array, nelems, fout, "check_unique, UNIQUE ARRAY");
	}
	for(j=0; j < (nelems-1); ++j){
		assert(h_array[j] >= 0);
		assert(h_array[j] <= MaxGlobalLabel);
		assert(h_array[j] != h_array[j+1]);
	}
	fflush(fout);
	free(h_array); h_array = NULL;
}

void bfs_check_new_queue(INT_T *d_new_queue, INT_T new_queue_count,
			 int64_t *d_pred, INT_T nverts,
			 const char *fcaller,FILE *fout)
{
        if (new_queue_count <= 0) return;

        INT_T *h_new_queue = NULL;
        int64_t *h_pred = NULL;
        h_new_queue = (INT_T*)callmalloc(new_queue_count*sizeof(INT_T), __func__);
        h_pred = (int64_t*)callmalloc(nverts*sizeof(int64_t), __func__);
        cudaMemcpy(h_new_queue, d_new_queue, new_queue_count*sizeof(INT_T),
                   cudaMemcpyDeviceToHost);
        checkCUDAError("atomic_enqueue_local: d_new_queue->h_new_queue");
        cudaMemcpy(h_pred, d_pred, nverts*sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
        checkCUDAError("atomic_enqueue_local: d_pred->h_pred");

    	fprintf(fout, "%s starting %s: new_queue_count=%"PRId64"; PRINT_MAX_NVERTS=%d\n", fcaller, __func__,
    			new_queue_count,PRINT_MAX_NVERTS);
	if (new_queue_count < PRINT_MAX_NVERTS){
		print_array(h_new_queue, new_queue_count, fout, "NEW_QUEUE");
	}
	//CHECK_LOCAL_VERTEX(h_new_queue, new_queue_count, __func__, fp_bfs);
	/*
	INT_T i, U;
	for (i=0; i < new_queue_count; ++i) {
		//U = VERTEX_LOCAL(h_new_queue[i]);
		U = (h_new_queue[i]);
		if (h_pred[U] == -1) {
			fprintf(fout, "%s: h_pred[U] == -1\n",__func__);
			fprintf(fout, "%s: U = %"PRI64"\n",	__func__, U);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		} else {
			if (new_queue_count < PRINT_MAX_NEDGES){
			fprintf(fout, "%s: U = %"PRI64"\n",	__func__, U);
			fprintf(fout, "%s: predU = %"PRI64"\n",	__func__, h_pred[U]);
			fflush(fout);
			}
		}
	}
	*/
	fflush(fout);
        free(h_new_queue); h_new_queue = NULL;
        free(h_pred); h_pred = NULL;
}


void bfs_check_new_queue_local(INT_T *d_new_queue, INT_T new_queue_count,
			 int64_t *d_pred, INT_T nverts, 
			 const char *fcaller,FILE *fout)
{
        if (new_queue_count <= 0) return;
 
        INT_T *h_new_queue = NULL;
        int64_t *h_pred = NULL;
        h_new_queue = (INT_T*)callmalloc(new_queue_count*sizeof(INT_T), __func__);
        h_pred = (int64_t*)callmalloc(nverts*sizeof(int64_t), __func__);
        cudaMemcpy(h_new_queue, d_new_queue, new_queue_count*sizeof(INT_T),
                   cudaMemcpyDeviceToHost);
        checkCUDAError("atomic_enqueue_local: d_new_queue->h_new_queue");
        cudaMemcpy(h_pred, d_pred, nverts*sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
        checkCUDAError("atomic_enqueue_local: d_pred->h_pred");

	fprintf(fout, "%s starting %s: new_queue_count=%"PRId64"; PRINT_MAX_NVERTS=%d\n", fcaller, __func__,
			new_queue_count,PRINT_MAX_NVERTS);
	if (new_queue_count < PRINT_MAX_NVERTS){
		print_array(h_new_queue, new_queue_count, fout, "NEW_QUEUE");
	}
	//CHECK_LOCAL_VERTEX(h_new_queue, new_queue_count, __func__, fp_bfs);
	/*
	INT_T i, U;
	for (i=0; i < new_queue_count; ++i) {
		//U = VERTEX_LOCAL(h_new_queue[i]);
		U = h_new_queue[i];
		if (h_pred[U] == -1) {
			fprintf(fout, "%s: h_pred[U] == -1\n",
				__func__);
			fprintf(fout, "%s: U = %"PRI64"\n",
				__func__, U);
			fprintf(fout, "%s: predU = %"PRI64"\n",
				__func__, h_pred[U]);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		} else {
			if (new_queue_count < PRINT_MAX_NEDGES){
			fprintf(fout, "%s: U = %"PRI64"\n",	__func__, U);
			fprintf(fout, "%s: predU = %"PRI64"\n",	__func__, h_pred[U]);
			fflush(fout);}
		}
	}
	*/
	fflush(fout);
        free(h_new_queue); h_new_queue = NULL;
        free(h_pred); h_pred = NULL;
}

/************************** Make struct func ***********************************/
int print_checkpassed = 1;
void check_add_vu(INT_T *d_edges, INT_T nedges, const char *fcaller, FILE *fout)
{
	// check data type
	INT_T X = -1;
	if (X != -1) {
		fprintf(stderr, "ERROR\n");
		fprintf(stderr, "%s: data type error, unsigned is not supported!\n", __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// The direct graph has nedges ---> 2*nedges elements
	// The undirect will have 2*nedges ----> 4*nedges elements
	INT_T *h_edges = NULL;
	INT_T *host_copy_of_d_edges = NULL;
	h_edges = (INT_T*)callmalloc(4*nedges*sizeof(INT_T), "check_add_vu: malloc h_edges");
	host_copy_of_d_edges = (INT_T*)callmalloc(4*nedges*sizeof(INT_T), "check_add_vu: malloc h_edges");
	cudaMemcpy(h_edges, d_edges, 2*nedges*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_add_vu: d_edges_u->h_edges");
	cudaMemcpy(host_copy_of_d_edges, d_edges, 4*nedges*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_add_vu: d_edges_u->host_copy_of_d_edges");
	
	// FIll the 2nd half of host array with (v,u) if u!=v or (-1,-1) if u==v 
	INT_T k,v,u;
	for (k=0; k < nedges; ++k) {
		u = h_edges[2*k];
		v = h_edges[2*k+1];
		if ( u != v ) {
			h_edges[2*nedges + 2*k] = v;
			h_edges[2*nedges + 2*k + 1] = u;
		} else if (u == v) {
			h_edges[2*nedges + 2*k] = -1;
			h_edges[2*nedges + 2*k + 1] = -1;
		}
	}

	// Check that the array on device and on host are equal 
	for(k=0; k < 4*nedges; ++k) {
		if (h_edges[k] != host_copy_of_d_edges[k]) {
			fprintf(stderr, "ERROR!\n");
			fprintf(stderr, "rank %d in %s in %s: the undirect graph is wrong!\n", rank, __func__, fcaller); 
			if (nedges < PRINT_MAX_NEDGES) {
				print_array(h_edges, 4*nedges, fout, "check_add_vu: HOST_EDGES");
				print_array(host_copy_of_d_edges, 4*nedges, fout, "check_add_vu: HOST_COPY_OF_D_EDGES");
        		}
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}	
	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	free(h_edges);
}

void check_add_undirect_edges(INT_T *d_edges, INT_T nedges, const char *fcaller, FILE *fout)
{
	// check data type
	INT_T X = -1;
	if (X != -1) {
		fprintf(stderr, "ERROR\n");
		fprintf(stderr, "%s: data type error, unsigned is not supported!\n", __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// The direct graph has nedges ---> 2*nedges elements
	// The undirect will have 2*nedges ----> 4*nedges elements
	INT_T *h_edges = NULL;
	h_edges = (INT_T*)callmalloc(4*nedges*sizeof(INT_T), "check_add_undirect_edges: malloc h_edges");
	cudaMemcpy(h_edges, d_edges, 4*nedges*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_add_undirect_edges: d_edges_u->h_edges");

	// Check that the graph is undirect
	INT_T k;
	INT_T u,v,uc,vc;
	for (k=0; k < nedges; ++k) {
		u = h_edges[2*k];
		v = h_edges[2*k+1];
		uc = h_edges[2*nedges + 2*k];
		vc = h_edges[2*nedges + 2*k + 1];
		if (u != v) {
			if (( u != vc ) || ( v != uc )) {
				fprintf(stderr, "ERROR\n");
				fprintf(stderr, "rank %d in %s in %s: direct edge and undirect edge differ!\n", 
					rank, fcaller, __func__);
				fprintf(stderr, "direct edge = (%"PRI64",%"PRI64")\n", u, v);
				fprintf(stderr, "undirect edge = (%"PRI64",%"PRI64")\n", uc, vc);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		} else if ( u == v ) {
			if (( uc != vc ) || ( uc != -1 ) || (vc != -1 )){
				fprintf(stderr, "ERROR\n");
				fprintf(stderr, "rank %d in %s in %s: undirect edge error!\n", 
					rank, fcaller, __func__);
				fprintf(stderr, "direct edge = (%"PRI64",%"PRI64")\n", u, v);
				fprintf(stderr, "undirect edge = (%"PRI64",%"PRI64")\n", uc, vc);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}

	// Check Labels	
	CHECK_GLOBAL_VERTEX_MINUSONE(h_edges, 4*nedges, __func__, fout);

	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	free(h_edges);
}

// The input edge list represent an undirect graph, here we check mainly
// if all the edges (-1,-1) are removed without errors.
// The input list had nelems edges before the remove operation
// and compact_nelems after, so now only compact_nelems are well defined.
void check_compact_edge_list(INT_T* d_edges, INT_T nelems, INT_T compact_nelems, const char* fcaller, FILE* fout) 
{
	// check data type
	INT_T X = -1;
	if (X != -1) {
		fprintf(stderr, "ERROR\n");
		fprintf(stderr, "%s in %s: data type error, unsigned is not supported!\n", __func__, fcaller);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (compact_nelems > nelems) {
		fprintf(stderr, "ERROR\n");
		fprintf(stderr, "rank %d in %s in %s: compact_nelems > input nelems!\n", 
			rank, __func__, fcaller);
		fprintf(stderr, "compact_nelems = %"PRI64"\n", compact_nelems);
		fprintf(stderr, "nelems before remove = %"PRI64"\n", nelems);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if ((compact_nelems%2) != 0) {
		fprintf(stderr, "ERROR\n");
		fprintf(stderr, "rank %d in %s in %s: the number of elems in the compact edge list are uneaven!\n", 
			rank, __func__, fcaller);
		fprintf(stderr, "compact_nelems = %"PRI64"\n", compact_nelems);
		fprintf(stderr, "nelems before remove= %"PRI64"\n", nelems);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	INT_T *h_edges = NULL;
	h_edges = (INT_T*)callmalloc(nelems*sizeof(INT_T), "check_compact: malloc h_edges");
	cudaMemcpy(h_edges, d_edges, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_compact: d_edges_u->h_edges");

	// Check Labels	
	CHECK_GLOBAL_VERTEX(h_edges, compact_nelems, __func__, fout);

	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	free(h_edges);
}

// Check the edge list after the split. After the split the array has the form:
// u0, u1, u2, ....,v0, v1, v2 .....
// If before the split the num of edges is odd I need to add an extra edges,
// The added edge is (U0, V0) at the end of the edge list. After the split
// the vertices of this edges are in the elements:
// d_edges[nedges_to_split - 1] = U0
// d_edges[2*nedges_to_split - 1] = V0
void check_split_edge_list(INT_T* d_edges, INT_T nedges, INT_T nedges_to_split, const char* fcaller, FILE* fout)
{

	CHECK_SIZE("zero", 0, "nedges", nedges, __func__);

	if (nedges_to_split < nedges) {
		fprintf(stderr, "ERROR\n");
		fprintf(stderr, "rank %d in %s in %s: the number of edges to split is less then the number of edges!\n", 
			rank, __func__, fcaller);
		fprintf(stderr, "nedges to split = %"PRI64"\n", nedges_to_split);
		fprintf(stderr, "nedges before split= %"PRI64"\n", nedges);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	INT_T *h_edges = NULL;
	h_edges = (INT_T*)callmalloc(2*nedges_to_split*sizeof(INT_T), "check_split_edge_list: malloc h_edges");
	cudaMemcpy(h_edges, d_edges, 2*nedges_to_split*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_split_edge_list: d_edges_u->h_edges");

	// If nedges was odd ..., here h_edges may be wrong because is printed before the test!
	if (nedges_to_split > nedges) {
		fprintf(fout, "rank %d in %s in %s, nedges was odd added the edge (%"PRI64",%"PRI64")\n",
			rank, __func__, fcaller, h_edges[0], h_edges[nedges_to_split]);
	}

	// If the num of edges is odd I can check that the last edge is (U0, V0)
	if (nedges_to_split > nedges) {
		if ((nedges%2) == 0) {
			fprintf(stderr, "ERROR\n");
			fprintf(stderr, "rank %d in %s in %s: the number of edges to split is greater then the number"
				" of edges but nedges is even!\n", rank, __func__, fcaller);
			fprintf(stderr, "nedges to split = %"PRI64"\n", nedges_to_split);
			fprintf(stderr, "nedges before split= %"PRI64"\n", nedges);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		// U0
		if(h_edges[0] != h_edges[nedges_to_split-1]){
			fprintf(stderr, "ERROR\n");
			fprintf(stderr, "rank %d in %s in %s h_edges[0] != h_edges[nedges_to_split-1]!\n", 
				rank, __func__, fcaller);
			fprintf(stderr, "h_edges[0] = %"PRI64"\n", h_edges[0]);
			fprintf(stderr, "h_edges[nedges_to_split -1]= %"PRI64"\n", h_edges[nedges_to_split-1]);
			fprintf(stderr, "nedges before split= %"PRI64"\n", nedges);
			fprintf(stderr, "nedges to split = %"PRI64"\n", nedges_to_split);
			print_array(h_edges, nedges_to_split, fout, "IN CHECK SPLIT: SPLITTED EDGE LIST U");
			print_array(h_edges+nedges_to_split, nedges_to_split, fout, "IN CHECK SPLIT: SPLITTED EDGE LIST V");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		// V0
		if(h_edges[nedges_to_split] != h_edges[2*nedges_to_split-1]){
			fprintf(stderr, "ERROR\n");
			fprintf(stderr, "rank %d in %s in %s h_edges[nedges_to_split] != h_edges[2*nedges_to_split-1]!\n", 
				rank, __func__, fcaller);
			fprintf(stderr, "h_edges[nedges_to_split] = %"PRI64"\n", h_edges[nedges_to_split]);
			fprintf(stderr, "h_edges[2*nedges_to_split-1]= %"PRI64"\n", h_edges[2*nedges_to_split-1]);
			fprintf(stderr, "nedges before split= %"PRI64"\n", nedges);
			fprintf(stderr, "nedges to split = %"PRI64"\n", nedges_to_split);
			fprintf(stderr, "2*nedges_to_split - 1= %"PRI64"\n", 2*nedges_to_split-1);
			print_array(h_edges, nedges_to_split, fout, "IN CHECK SPLIT: SPLITTED EDGE LIST U");
			print_array(h_edges+nedges_to_split, nedges_to_split, fout, "IN CHECK SPLIT: SPLITTED EDGE LIST V");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	// Check Labels	
	CHECK_GLOBAL_VERTEX(h_edges, 2*nedges_to_split, __func__, fout);

	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	free(h_edges);
}

void check_sort(INT_T *d_array, INT_T nelems, const char *fcaller, FILE *fout)
{
	CHECK_SIZE("zero", 0, "nelems", nelems, __func__);

	INT_T *h_array; h_array = NULL;
	h_array = (INT_T*)callmalloc(nelems*sizeof(INT_T), __func__);
	cudaMemcpy(h_array, d_array, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_sort: d_array->h_array");

	INT_T j;
	if (nelems < PRINT_MAX_NEDGES){
		print_array(h_array, nelems, fout, "check_sort, SORTED ARRAY");
	}

	for(j=0; j < nelems-1; ++j){
		if(h_array[j] > h_array[j+1]) {
			fprintf(stderr, "ERROR\n");
			fprintf(stderr, "rank %d in %s in %s h_array[j] > h_array[j+1] at element j=%"PRI64"!\n", 
				rank, __func__, fcaller, j);
			fprintf(stderr, "h_array[j]=%"PRI64"\n", h_array[j]);
			fprintf(stderr, "h_array[j+1]=%"PRI64"\n", h_array[j+1]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	// Check Labels	
	CHECK_GLOBAL_VERTEX(h_array, nelems, __func__, fout);

	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	fflush(fout);
	free(h_array); h_array = NULL;
}

void check_offset(INT_T *d_array, INT_T nelems, const char *fcaller, FILE *fout)
{
	CHECK_SIZE("zero", 0, "nelems", nelems, __func__);

	INT_T *h_array; h_array = NULL;
	h_array = (INT_T*)callmalloc(nelems*sizeof(INT_T), __func__);
	cudaMemcpy(h_array, d_array, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_offset: d_array->h_array");

	INT_T j;
	if (nelems < PRINT_MAX_NEDGES){
		print_array(h_array, nelems, fout, "check_offset, OFFSET ARRAY");
	}

	for(j=0; j < nelems-1; ++j){
		if(h_array[j] > h_array[j+1]) {
			if ((h_array[j+1] != 0) && (h_array[j+1] != -1)){ 
				fprintf(stderr, "ERROR\n");
				fprintf(stderr, "rank %d in %s in %s h_array[j] > h_array[j+1] at element j=%"PRI64"!\n", 
					rank, __func__, fcaller, j);
				fprintf(stderr, "h_array[j]=%"PRI64"\n", h_array[j]);
				fprintf(stderr, "h_array[j+1]=%"PRI64"\n", h_array[j+1]);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
		INT_T maxdegree = global_edgefactor * (2<<global_scale);
		if (h_array[j] > maxdegree) {
			fprintf(stderr, "ERROR\n");
			fprintf(stderr, "rank %d in %s in %s: Offset greater than edgefactor*2^SCALE at element j=%"PRI64"!\n", 
				rank, __func__, fcaller, j);
			fprintf(stderr, "h_array[j]=%"PRI64"\n", h_array[j]);
			fprintf(stderr, "edgefactor*2^SCALE=%"PRI64"\n", maxdegree);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	fflush(fout);
	free(h_array); h_array = NULL;
}

void check_degree(INT_T *d_array, INT_T nelems, const char *fcaller, FILE *fout)
{
	CHECK_SIZE("zero", 0, "nelems", nelems, __func__);

	INT_T *h_array; h_array = NULL;
	h_array = (INT_T*)callmalloc(nelems*sizeof(INT_T), __func__);
	cudaMemcpy(h_array, d_array, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_degree: d_array->h_array");

	INT_T j;
	if (nelems < PRINT_MAX_NEDGES){
		print_array(h_array, nelems, fout, "check_degree, OFFSET ARRAY");
	}

	for(j=0; j < nelems; ++j){
		INT_T maxdegree = 16 * (2<<global_scale);
		if (h_array[j] > maxdegree) {
			fprintf(stderr, "ERROR\n");
			fprintf(stderr, "rank %d in %s in %s: degree greater than 16*2^SCALE at element j=%"PRI64"!\n", 
				rank, __func__, fcaller, j);
			fprintf(stderr, "h_array[j]=%"PRI64"\n", h_array[j]);
			fprintf(stderr, "16*2^SCALE=%"PRI64"\n", maxdegree);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	fflush(fout);
	free(h_array); h_array = NULL;
}

//  d_owners = VERTEX_OWNER(V)
//  d_idx = gid
void check_owners(INT_T *d_owners, INT_T *d_idx, INT_T nelems, INT_T *d_vertices,
		  const char *fcaller, FILE *fout)
{
	INT_T *h_vertices = NULL;
	INT_T *h_owners = NULL;
	INT_T *h_idx = NULL;
	h_vertices = (INT_T*)callmalloc(nelems*sizeof(INT_T), __func__);
	h_owners = (INT_T*)callmalloc(nelems*sizeof(INT_T), __func__);
	h_idx = (INT_T*)callmalloc(nelems*sizeof(INT_T), __func__);
	cudaMemcpy(h_vertices, d_vertices, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("bfs_owners: d_send_buff->vertices");
	cudaMemcpy(h_owners, d_owners, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("bfs_owners: d_owners->h_owners");
	cudaMemcpy(h_idx, d_idx, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("bfs_owners: d_idx->h_idx");

	if (nelems < PRINT_MAX_NEDGES) {
		print_array(h_owners, nelems, fout, "check_owners_bfs: H_MASK_1");
		print_array(h_idx, nelems, fout, "check_owners_bfs: H_MASK_2");
	}
	INT_T i;
	for (i=0; i < nelems; ++i) {
		CHECK_SIZE("zero", 0, "h_owners", h_owners[i], __func__);
		CHECK_SIZE("zero", 0, "h_idx", h_idx[i], __func__);
		CHECK_SIZE("h_owners", h_owners[i], "size", size, __func__);
		CHECK_SIZE("h_idx", h_idx[i], "nelems", nelems, __func__);
		if (h_owners[i] != VERTEX_OWNER(h_vertices[i])) {
			fprintf(stderr, "ERROR\n");
			fprintf(stderr, "rank %d in %s in %s: h_owners[i] != VERTEX_OWNER(h_vertices[i]) at i=%"PRI64"\n",
				rank, __func__, fcaller, i);
			fprintf(stderr, "h_owners[i]=%"PRI64"\n", h_owners[i]);
			fprintf(stderr, "VERTEX_OWNER(h_vertices[i])=%d\n", VERTEX_OWNER(h_vertices[i]));
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

        free(h_vertices); h_vertices = NULL;
        free(h_owners); h_owners = NULL;
        free(h_idx); h_idx = NULL;
}

void check_count(INT_T *d_array, INT_T nelems, INT_T *send_count_per_proc,
		 const char *fcaller, FILE *fout)
{
        INT_T *h_array = NULL;
        h_array = (INT_T*)callmalloc(nelems*sizeof(INT_T), "bfs_count_vertices: " "malloc h_array");
        cudaMemcpy(h_array, d_array, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
        checkCUDAError("bfs_count_vertices: d_array->h_array");

	INT_T j;
	if (nelems < PRINT_MAX_NEDGES){
		print_array(h_array, nelems, fout, "check_count, H_EDGES");
		print_array(send_count_per_proc, size, fout, "check_count, SEND_COUNT_PER_PROC");
	}
	INT_T count_per_proc[size];
	for (j=0; j < size; ++j) {
		count_per_proc[j] = 0;
	}
	INT_T proc;
	for(j=0; j < nelems; ++j) {
		proc = h_array[j];
		if (proc >= size) {
			fprintf(stderr, "ERROR\n");
			fprintf(stderr, "rank %d in %s in %s: h_array[j] > size at j=%"PRI64"\n", 
				rank, __func__, fcaller, j);
			fprintf(stderr, "h_array[j]=%"PRI64"\n", h_array[j]);
			fprintf(stderr, "size=%d\n", size);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		count_per_proc[proc] += 2;
	}
	for (j=0; j < size; ++j) {
		if(count_per_proc[j] != send_count_per_proc[j]) {
			fprintf(stderr, "ERROR\n");
			fprintf(stderr, "count_per_proc[j] != send_count_per_proc[j]\n");
			print_array(count_per_proc, size, fout, "CHECK_COUNT: COUNT_PER_PROC");
			print_array(send_count_per_proc, size, fout, "CHECK_COUNT: SEND_COUNT_PER_PROC");
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}	

	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	fflush(fout);
        free(h_array); h_array = NULL;
}

// Check the array to send
void check_back(INT_T *d_array, INT_T nelems, const char *fcaller, FILE *fout)
{
	INT_T *h_array = NULL;
	h_array = (INT_T*)callmalloc(nelems*sizeof(INT_T), "check_back: malloc h_array");
	cudaMemcpy(h_array, d_array, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_back: d_array_u->h_array");

	if (nelems < PRINT_MAX_NEDGES) {
		print_array(h_array, nelems, fout, "check_back: H_ARRAY");
	}
	INT_T i;
	for (i=0; i < nelems; ++i) {
		if (rank == VERTEX_OWNER(h_array[i])) {
			if (h_array[i] < 0) {
				fprintf(stderr, "ERROR\n");
				fprintf(stderr, "rank %d in %s in %s," " local vertex = %"PRI64" <= 0\n",
					 rank,  __func__, fcaller,  h_array[i]);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			if (VERTEX_LOCAL(h_array[i]) > MaxLabel) {
				fprintf(stderr, "ERROR\n");
				fprintf(fout, "rank %d in %s in %s, h_array[i] >= MaxLabel at i=%"PRI64"\n", 
					rank, __func__, fcaller, i);
				fprintf(fout, "rank %d in %s in %s, h_array[i]=%"PRI64"\n", rank, __func__, 
					fcaller, h_array[i]);
				fprintf(fout, "rank %d in %s in %s, MaxLabel=%"PRId64"\n", rank, __func__, 
					fcaller, MaxLabel);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		} else {
			if (h_array[i] < 0) {
				fprintf(stderr, "ERROR\n");
				fprintf(stderr, "rank %d in %s in %s, local vertex = %"PRI64" <= 0 at i=%"PRI64"\n", 
					rank, __func__, fcaller, h_array[i], i);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			if (h_array[i] > MaxGlobalLabel) {
				fprintf(stderr, "ERROR\n");
				fprintf(stderr, "rank %d in %s in %s, h_array[i] >= MaxGlobalLabel at i=%"PRI64"\n", 
					rank, __func__, fcaller, i);
				fprintf(stderr, "rank %d in %s in %s, h_array[i]=%"PRI64"\n", 
					rank, __func__, fcaller, h_array[i]);
				fprintf(stderr, "rank %d in %s in %s, MaxGlobalLabel=%"PRId64"\n", 
					rank, __func__, fcaller, MaxGlobalLabel);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			if (VERTEX_OWNER(h_array[i]) > size) {
				fprintf(stderr, "rank %d in %s in %s,  VERTEX_OWNER(h_array[i]) > size at i=%"PRI64"\n",
					rank, __func__, fcaller, i);
				fprintf(stderr, "rank %d in %s in %s, VERTEX_OWNER(h_array[i])=%d\n", 
					rank, fcaller, __func__, VERTEX_OWNER(h_array[i]));
				fprintf(stderr, "rank %d in %s in %s, size=%d\n", rank, __func__, fcaller,  size);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}

	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	fflush(fout);
	free(h_array); h_array = NULL;
}

void check_unsplit(INT_T *d_array, INT_T nelems, const char *fcaller, FILE *fout)
{
	INT_T *h_array = NULL;
	h_array = (INT_T*)callmalloc(nelems*sizeof(INT_T), "check_unsplit: malloc h_array");
	cudaMemcpy(h_array, d_array, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_unsplit: d_array_u->h_array");
	
	// Check Labels	
	CHECK_GLOBAL_VERTEX(h_array, nelems, __func__, fout);
	
	// Print that everything is ok
	if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	fflush(fout);
	free(h_array); h_array = NULL;
}

void check_merge(INT32_T *d_array, INT_T nelems, const char *fcaller, FILE *fout)
{
	INT32_T *h_array = NULL;
	h_array = (INT32_T*)callmalloc(2*nelems*sizeof(INT32_T), "check_merge: malloc h_array");
	cudaMemcpy(h_array, d_array, 2*nelems*sizeof(INT32_T), cudaMemcpyDeviceToHost);
	checkCUDAError("check_merge: d_array_u->h_array");

	if (nelems < PRINT_MAX_NEDGES) {
		print_edges32(h_array, nelems, fout, "check_back: H_ARRAY");
	}

	// Check Labels
	//CHECK_GLOBAL_VERTEX(h_array, nelems, __func__, fout);

	// Print that everything is ok
	//if (print_checkpassed) fprintf(fout, "rank %d, %s in %s: check passed!\n", rank, __func__, fcaller);

	fflush(fout);
	free(h_array); h_array = NULL;
}

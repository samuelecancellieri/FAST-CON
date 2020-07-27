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

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

#include "header.h"
#include "defines.h"
#include "gputils.h"
#include "cputils.h"
#include "make_struct_gpufunc.h"
#include "cudakernel/mykernels.h"
#include "mythrustlib.h"

extern FILE *fp_struct;
extern int nthreads, nblocks, maxblocks;
extern int rank, size, lgsize;

/* For each edge (u,v) if (u != v) add (v,u) else add (X,X) */
int add_edeges_vu(INT_T *d_edges_new, INT_T nedges)
{
	// Check type
	INT_T X = -1;
	if (X != -1) {
		fprintf(stderr, "add_edges_vu: X is not -1! Quit\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	int nblocks;
	nblocks = MIN((INT_T)maxblocks, ((nedges + (INT_T)nthreads - 1)/
		  ((INT_T)nthreads)));

	k_add_vu<<<nblocks, nthreads>>>(d_edges_new, nedges);
	checkCUDAError("add_edges_vu: kernel launch");
	
	return 0;
}

/*
// Compact the undirect array of edges and return the compact number of edges
// Remove edges (-1,-1)
int compact_elems(INT_T *d_edges, INT_T nelems, INT_T *compact_nelems, int value)
{
	if (nelems <= 0) {
		fprintf(stderr, "%s: nelems = 0! Quit.\n", __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	call_thrust_remove(d_edges, nelems, compact_nelems, value);

	if (compact_nelems[0] > nelems) {
		fprintf(stderr, "ERROR\n");
		fprintf(stderr, "rank %d in %s: compact_nelems > input nelems!\n", 
			rank, __func__);
		fprintf(stderr, "compact_nelems = %"PRI64"\n", compact_nelems);
		fprintf(stderr, "input nelems = %"PRI64"\n", nelems);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if ((compact_nelems[0]%2) != 0) {
		fprintf(stderr, "ERROR\n");
		fprintf(stderr, "rank %d in %s: the number of elems in the compact edge list are uneaven!\n", 
			rank, __func__);
		fprintf(stderr, "compact_nelems = %"PRI64"\n", compact_nelems);
		fprintf(stderr, "input nelems = %"PRI64"\n", nelems);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	return 0;
}
*/


// Split edge list (u,v) in u and v in place. If the number of edges is odd, 
// add a dummy edges (U0,V0) at the end of the edges list. This extra edge
// is a copy of the first edge and does not affect the result.
// nedges_to_split is == nedges if the nedges is even else is nedges+1
// after the split:
// d_edges[nedges_to_split - 1] = U0
// d_edges[2*nedges_to_split - 1] = V0

// Split edge list (u,v) in u0,u1,..,v0,v1,..
// The input edge list is undirect, to split the list the number
// of edges must be even, the number of edges returned, nedges_to_split
// is even.
int split_edges(INT_T* d_edges, INT_T nedges, INT_T* nedges_to_split)
{
	CHECK_SIZE("zero", 0, "nedges", nedges, __func__);

	int nblocks;
	int odd = -1;

	// Check type
	INT_T X = -1;
	if (X != -1) {
		fprintf(stderr, "split_edges: X is not -1! Quit\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	
	if ((nedges%2) == 0) {
		nedges_to_split[0] = nedges;
		odd = 0;
	} else {
		// If the number of edges is odd, I need to add an extra edge
		// at the end of the edge list. I add a copy of the first edge.
		cudaMemcpy(d_edges+2*nedges, d_edges, 2*sizeof(INT_T), cudaMemcpyDeviceToDevice);
		checkCUDAError("split_edges: memcpy dev to dev d_edges->d_edges+4*nedges");
		nedges_to_split[0] = nedges + 1;
		odd = 1;
	}
	//print_device_array(d_edges, (INT_T)2, stderr, "SPLIT_EDGES: FIRST EDGE");

	nblocks = MIN((INT_T)maxblocks, ((nedges_to_split[0]/2 + (INT_T)nthreads - 1)/((INT_T)nthreads)));

	split_kernel<<<nblocks, nthreads>>>(d_edges, nedges_to_split[0], odd);
	checkCUDAError("split_edges: kernel");

	return 0;
}

/* substitute u with owner of u */
int owners_edges(INT_T* d_edges_u, INT_T nedges, INT_T* d_edges_appo)
{
	if (size == 1) {
		fprintf(stderr, "owners_edges: size = 1, nothing to do\n");
		return 0;
	}

	// Compute the number of block needed by the kernel
	int nblocks = MIN((INT_T)maxblocks, ((nedges + (INT_T)nthreads - 1)/ ((INT_T)nthreads)));

	// Launch the kernel
	k_owners<<<nblocks, nthreads>>>(d_edges_u, nedges, d_edges_appo, rank, size, lgsize);
	checkCUDAError("owners_edges: kernel");

	return 0;
}

/* 
int sort_edges_u(INT_T* d_edges_u, INT_T* d_edges_v, INT_T nedges, INT_T *umax)
{
	if (nedges == 0) {
		fprintf(stderr, "sort_edges: nedges = 0! Quit.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

#ifdef GPU_DEBUG_2
	fprintf(fp_struct, "sort_edges: processing d_edges with %"PRI64" "
					"elements\n", nedges);
	fprintf(fp_struct, "sort_edges: nelems = %"PRI64"\n", nedges);
	fflush(fp_struct);
#endif

	// Invoke thrust to sort the array 
	thrust::device_ptr<INT_T> d_thrust_edges_u(d_edges_u);
	thrust::device_ptr<INT_T> d_thrust_edges_v(d_edges_v);
	thrust::sort_by_key(d_thrust_edges_u, d_thrust_edges_u + nedges, 
											d_thrust_edges_v);
	umax[0] = d_thrust_edges_u[nedges-1];
	return 0;
}
*/
 
/* Count edges per proc */
int count_vertices(INT_T* d_edges, INT_T nedges, INT_T* d_count_per_proc, 
		   INT_T* send_count_per_proc, INT_T* host_count_per_proc)
{
	if (size == 1) {
		fprintf(stderr, "count_vertices: size = 1, nothing to do\n");
		return 0;
	}
	int nblocks = MIN((INT_T)maxblocks, ((nedges + (INT_T)nthreads - 1)/ ((INT_T)nthreads)));

	cudaMemset(d_count_per_proc, 0, 2*size*sizeof(INT_T));

	k_count_proc<<<nblocks, nthreads>>>(d_edges, nedges, d_count_per_proc);
	checkCUDAError("count_vertices: kernel");
	
	cudaMemcpy(host_count_per_proc, d_count_per_proc, 2*size*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("count_vertices: d_count_per_proc -> host_count_per_proc");

	// Size is always small, I perform the last step on CPU.
	int i;
	for(i=0; i < size; ++i)
		send_count_per_proc[i] = 2*(host_count_per_proc[2*i+1] - host_count_per_proc[2*i]);
	return 0;
}

int back_vertices(INT_T* d_edges_u, INT_T* d_edges_v, INT_T nedges, 
		  INT_T* d_edges_appo_u, INT_T* d_edges_appo_v)
{
	if (size == 1) {
		fprintf(stderr, "back_vertices: size = 1, nothing to do\n");
		return 0;
	}
	int nblocks;
	nblocks = MIN((INT_T)maxblocks, ((nedges + (INT_T)nthreads - 1)/ ((INT_T)nthreads)));

	back_kernel<<<nblocks, nthreads>>>(d_edges_u, nedges, d_edges_appo_u, d_edges_appo_v);
	checkCUDAError("back_vertices: kernel");

	cudaMemcpy(d_edges_u, d_edges_appo_u, nedges*sizeof(INT_T), cudaMemcpyDeviceToDevice);
	checkCUDAError("back_vertices: copy d_edges_appo_u -> d_edges_u");

	back_kernel<<<nblocks, nthreads>>>(d_edges_v, nedges, d_edges_appo_u, d_edges_appo_v);
	checkCUDAError("back_vertices: kernel");

	cudaMemcpy(d_edges_v, d_edges_appo_u, nedges*sizeof(INT_T), cudaMemcpyDeviceToDevice);
	checkCUDAError("back_vertices: copy d_edges_appo_u -> d_edges_v");

	return 0;
}

int unsplit_edges(INT_T *d_edges_u, INT_T *d_edges_v, INT_T nedges, INT_T *d_edges_appo)
{
	if (size == 1) {
		fprintf(stderr, "unsplit_edges: size = 1, nothing to do\n");
		return 0;
	}

	int nblocks;
	nblocks = MIN((INT_T)maxblocks, ((nedges + 2*(INT_T)nthreads - 1) / (2*(INT_T)nthreads)));

	unsplit_kernel<<<nblocks, nthreads>>>(d_edges_u, d_edges_v, nedges, d_edges_appo);
	checkCUDAError("unsplit_edges: kernel");

	return 0;
}

int make_offset(INT_T *d_edges_u, INT_T nedges, INT_T *d_count_u, 
		INT_T nverts, INT_T *d_degree)
{
	// Check type
	INT_T X = -1;
	if (X != -1) {
		fprintf(stderr, "%s: X is not -1! Quit\n", __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	int nblocks;
	nblocks = MIN((INT_T)maxblocks, ((nedges + (INT_T)nthreads - 1)/ ((INT_T)nthreads)));

	// Init the offset array with 0
	cudaMemset(d_count_u, 0, 2*nverts*sizeof(INT_T));

	// First compute start and end offset
	k_count_vert<<<nblocks, nthreads>>>(d_edges_u, nedges, d_count_u, nverts, rank, size, lgsize);
	checkCUDAError("make_offset: kernel 1 launch");

	// Compute degree
	nblocks = MIN((INT_T)maxblocks, ((nverts + (INT_T)nthreads - 1) / ((INT_T)nthreads)));
	k_degree<<<nblocks, nthreads>>>(d_count_u, nverts, d_degree);
	checkCUDAError("make_offset: kernel 2 launch");

	return 0;
}


int sort_unique_edges(INT_T* d_edges_v, INT_T* d_edges_u,
                      INT_T *compact_nedges, unsigned int *stencil,
                      INT_T *umax, INT_T *vmax)
{
	if (compact_nedges[0] == 0) {
		fprintf(stderr, "sort_unique_edges: compact_nedges[0] = 0! Quit.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

  	int nblocks = MIN((INT_T)maxblocks, ((compact_nedges[0] + (INT_T)nthreads - 1)/
                ((INT_T)nthreads)));

  	// Invoke thrust to sort the array 
	call_thrust_stable_sort_by_key_and_max(d_edges_v, d_edges_u, compact_nedges[0], vmax);
	call_thrust_stable_sort_by_key_and_max(d_edges_u, d_edges_v, compact_nedges[0], umax);
  
	/*
	thrust::device_ptr<INT_T> d_thrust_edges_u(d_edges_u);
  	thrust::device_ptr<INT_T> d_thrust_edges_v(d_edges_v);
  	thrust::stable_sort_by_key(d_thrust_edges_v,
  	                          d_thrust_edges_v + compact_nedges[0],
  	                          d_thrust_edges_u);
  	vmax[0] = d_thrust_edges_v[compact_nedges[0]-1];

  	thrust::stable_sort_by_key(d_thrust_edges_u,
  	                          d_thrust_edges_u + compact_nedges[0],
  	                          d_thrust_edges_v);
  	checkCUDAError("sort_unique_edges: thrust");
  	umax[0] = d_thrust_edges_u[compact_nedges[0]-1];
	*/
  	// Fill in stencil with 0 and 1
  	k_find_duplicates<<<nblocks, nthreads>>>(d_edges_u, d_edges_v, compact_nedges[0], stencil);
  	checkCUDAError("sort_unique_edges: kernel");

	INT_T compact_nelems_u;
	INT_T compact_nelems_v;
	call_thrust_remove_by_stencil(d_edges_u, compact_nedges[0], &compact_nelems_u, stencil);
	call_thrust_remove_by_stencil(d_edges_v, compact_nedges[0], &compact_nelems_v, stencil);
/*
  thrust::device_ptr<unsigned int> d_thrust_stencil(stencil);
  thrust::device_ptr<INT_T> new_end_u;
  thrust::device_ptr<INT_T> new_end_v;
  new_end_u = thrust::remove_if(d_thrust_edges_u,
                                d_thrust_edges_u + compact_nedges[0],
                                d_thrust_stencil, is_uno());
  new_end_v = thrust::remove_if(d_thrust_edges_v,
                                d_thrust_edges_v + compact_nedges[0],
                                d_thrust_stencil, is_uno());

  INT_T compact_nelems_u = new_end_u - d_thrust_edges_u;
  INT_T compact_nelems_v = new_end_v - d_thrust_edges_v;
*/

  assert(compact_nelems_v == compact_nelems_u);
  compact_nedges[0] = compact_nelems_u;

  return 0;
}

int copy_bitmask_on_device(mask *h_bitmask, mask *d_bitmask)
{
	INT32_T pnelems=h_bitmask->p_nelems;
	INT32_T mnelems=h_bitmask->m_nelems;
	// Check memory
	checkFreeMemory((pnelems/2 + mnelems/2), stderr, __func__);
	
	// Allocate d_pointer e d_bitmask
	INT32_T *d_pointer_bitmask = NULL;
	INT32_T *d_mask = NULL;
	cudaMalloc((void**)&d_pointer_bitmask, pnelems*sizeof(INT32_T));	
	cudaMalloc((void**)&d_mask, mnelems*sizeof(INT32_T));	

	// Copy bitmask to device
	cudaMemcpy(d_pointer_bitmask, h_bitmask->pedges, pnelems*sizeof(INT32_T), cudaMemcpyHostToDevice);
	
	cudaMemset(d_mask, 0, mnelems*sizeof(INT32_T));

	d_bitmask->pedges = d_pointer_bitmask;
 	d_bitmask->mask = d_mask;

	/* DEBUG
	cudaMemcpy(h_bitmask->pedges, d_bitmask->pedges, pnelems*sizeof(INT32_T),
		   cudaMemcpyDeviceToHost);
	h_bitmask->mask = (INT32_T*)callmalloc(mnelems*sizeof(INT32_T), 
			  "copy_bitmask_on_device: malloc h_bitmask->mask");
	cudaMemcpy(h_bitmask->mask, d_bitmask->mask, mnelems*sizeof(INT32_T),
		   cudaMemcpyDeviceToHost);

	print_array_32t(h_bitmask->pedges, pnelems, fp_struct, "copy: D_BITMASK_POINTER");
	print_array_32t(h_bitmask->mask, mnelems, fp_struct, "copy: D_BITMASK");
	*/

	return 0;
}


int edge_owners(INT_T* d_edges, INT_T nedges, INT_T* d_mask_1, INT_T* d_mask_2)
{
	CHECK_INPUT(d_edges, nedges, __func__);
	CHECK_INPUT(d_mask_1, nedges, __func__);
	CHECK_INPUT(d_mask_2, nedges, __func__);

	int nblocks = MIN((INT_T)maxblocks, (nedges + (INT_T)nthreads - 1)/(INT_T)nthreads);

	k_edge_owner<<<nblocks, nthreads>>>(d_edges, nedges, d_mask_1, d_mask_2, rank, size, lgsize);
	checkCUDAError("k_edge_owner: Kernel");

	return 0;
}


int count_edges (INT_T* d_array, INT_T nelems, INT_T* d_edges_per_proc)
{

	int nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));

	INT_T *d_count_edges;
	cudaMalloc((void**)&d_count_edges, 2*size*sizeof(INT_T));
	checkCUDAError("count_edges: cudaMalloc d_count_edges");

	if (nelems > 1) {
		cudaMemset(d_count_edges, 0, 2*size*sizeof(INT_T));
		k_bfs_count_proc<<<nblocks, nthreads>>>(d_array, nelems, d_count_edges);
		checkCUDAError("count_edges: kernel1");
		nblocks = MIN((INT_T)maxblocks, ((size + (INT_T)nthreads-1) / ((INT_T)nthreads)));
		k_count_proc<<<nblocks, nthreads>>>(d_edges_per_proc, d_count_edges, rank, size, lgsize);
		checkCUDAError("count_edges: kernel1");
		cudaFree(d_count_edges);

	} else {
		int p0 = d_array[0];
		cudaMemcpy(d_edges_per_proc+p0, &nelems, 1*sizeof(INT_T), cudaMemcpyHostToDevice);
	}

	return 0;
}

int reorder_edges (INT_T* d_array, INT_T nelems, INT_T* d_support, INT_T* d_idx)
{
	CHECK_INPUT(d_array, nelems, __func__);
	CHECK_INPUT(d_support, nelems, __func__);
	CHECK_INPUT(d_idx, nelems, __func__);

	int nblocks;
	nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1) / ((INT_T)nthreads)));
	bfs_back_kernel<<<nblocks, nthreads>>>(d_array, nelems, d_support, d_idx);
	checkCUDAError("reorder_edges: kernel 1");
	cudaMemcpy(d_array, d_support, nelems*sizeof(INT_T), cudaMemcpyDeviceToDevice);
	checkCUDAError("reorder_edges: copy d_support->d_array");
	return 0;
}


int build_bitmask_on_device(mask *h_bitmask, mask *d_bitmask, adjlist *d_graph)
{
	INT_T pnelems=h_bitmask->p_nelems;
	INT_T mnelems=h_bitmask->m_nelems;
	INT_T nverts = d_graph->nverts;
	// Check memory
	checkFreeMemory((pnelems/2 + mnelems/2 + 3*mnelems + nverts/2), stderr, __func__);

	// Allocate d_pointer e d_bitmask
	INT32_T *d_bitmask_pedges = NULL;
	INT32_T *d_bitmask_pverts = NULL;
	INT32_T *d_mask           = NULL;
	INT_T   *d_unique_edges   = NULL;
	INT_T   *d_mask_1, *d_mask_2;
	INT_T   *d_edges  = d_graph->edges;
    INT_T    dummy;
	INT_T *d_edges_offset_per_proc;
	INT_T *h_edges_offset_per_proc;

	cudaMalloc((void**)&d_edges_offset_per_proc, (size+1)*sizeof(INT_T));
	checkCUDAError("build_bitmask_on_device: cudaMalloc d_edges_offset_per_proc");

	h_edges_offset_per_proc = (INT_T*) callmalloc((size+1)*sizeof(INT_T),"");

	cudaMalloc((void**)&d_bitmask_pedges, pnelems*sizeof(INT32_T));
	checkCUDAError("build_bitmask_on_device: cudaMalloc d_bitmask_pedges");
	cudaMalloc((void**)&d_bitmask_pverts, nverts*sizeof(INT32_T));
	checkCUDAError("build_bitmask_on_device: cudaMalloc d_bitmask_pedges");
	cudaMalloc((void**)&d_mask, (mnelems+1)*sizeof(INT32_T));
	checkCUDAError("build_bitmask_on_device: cudaMalloc d_mask");

	cudaMalloc((void**)&d_unique_edges, mnelems*sizeof(INT_T)); //Unique edges
	checkCUDAError("build_bitmask_on_device: cudaMalloc d_unique_edges");
	cudaMalloc((void**)&d_mask_1, mnelems*sizeof(INT_T)); //Unique edges
	checkCUDAError("build_bitmask_on_device: cudaMalloc d_mask_1");
	cudaMalloc((void**)&d_mask_2, mnelems*sizeof(INT_T)); //Unique edges
	checkCUDAError("build_bitmask_on_device: cudaMalloc d_mask_2");

	// Copy unique_edges to device
	cudaMemcpy(d_unique_edges, h_bitmask->unique_edges, mnelems*sizeof(INT_T), cudaMemcpyHostToDevice);
	checkCUDAError("build_bitmask_on_device: cudaMemcpy d_unique_edges");

	//Sort unique edges by owner
	edge_owners(d_unique_edges, mnelems, d_mask_1, d_mask_2);
	call_thrust_sort_by_key(d_mask_1, d_mask_2, mnelems);
	checkCUDAError("build_bitmask_on_device: call_thrust_sort_by_key");
	//Count how may vertex for each processors
	count_edges(d_mask_1, mnelems, d_edges_offset_per_proc);
	call_thrust_exclusive_scan(d_edges_offset_per_proc, &dummy, (INT_T)size+1, d_edges_offset_per_proc);
	//Reorder Unique edges according to owners
	reorder_edges(d_unique_edges, mnelems, d_mask_1, d_mask_2);

	cudaFree(d_mask_1);
	cudaFree(d_mask_2);

	cudaMemcpy(h_edges_offset_per_proc, d_edges_offset_per_proc, (size+1)*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("build_bitmask_on_device: cudaMemcpy h_edges_offset_per_proc");

	//Fill pointer to bitmask with ids of the unique edges
	// Compute the number of block needed by the kernel
	int nblocks = MIN((INT_T)maxblocks, ((pnelems + (INT_T)nthreads - 1)/ ((INT_T)nthreads)));

	// Launch the kernel
	k_bitmask_edges<<<nblocks, nthreads>>>(d_edges, pnelems, d_unique_edges, d_edges_offset_per_proc, d_bitmask_pedges,
			                               rank, size, lgsize);
	checkCUDAError("build_bitmask_on_device: kernel k_bitmask_edges");

	cudaMemset(d_bitmask_pverts, NO_CONNECTIONS, nverts*sizeof(INT32_T));
	int nelems = h_edges_offset_per_proc[rank+1]-h_edges_offset_per_proc[rank];
	INT_T offset = h_edges_offset_per_proc[rank];

	// Compute the number of block needed by the kernel
	nblocks = MIN((INT_T)maxblocks, ((nelems + (INT_T)nthreads - 1)/ ((INT_T)nthreads)));
	// Launch the kernel
	k_bitmask_verts<<<nblocks, nthreads>>>(d_unique_edges, offset, nelems, d_bitmask_pverts,
			                               rank, size, lgsize);
	checkCUDAError("build_bitmask_on_device: kernel k_bitmask_verts");

	//In this way pverts with no local connections will point to a dummy element in d_mask
	call_thrust_replace(d_bitmask_pverts, nverts, NO_CONNECTIONS, mnelems);

	d_bitmask->pedges = d_bitmask_pedges;
	d_bitmask->pverts = d_bitmask_pverts;
 	d_bitmask->mask = d_mask;
 	d_bitmask->unique_edges = d_unique_edges;
 	d_bitmask->proc_offset = d_edges_offset_per_proc;
 	d_bitmask->p_nelems = pnelems;
 	d_bitmask->m_nelems = mnelems;

 	free(h_edges_offset_per_proc);

	return 0;
}


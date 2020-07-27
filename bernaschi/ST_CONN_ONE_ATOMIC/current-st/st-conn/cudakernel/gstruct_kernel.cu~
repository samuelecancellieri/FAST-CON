/* This is to make PRId64 working with c++ compiler */
#ifdef __cplusplus
#define __STDC_FORMAT_MACROS
#endif
/* header of int64_t and PRId64 */
#include <inttypes.h>
#include <stdio.h>
#include <cuda.h>
#include "defines.h"
#include "header.h"

// Count edges per processor with atomic add in gobal nmeory
__global__ void kernel1(INT_T *edges, INT_T nedges, 
			INT_T *ne_per_proc, const int size)
{
	unsigned int tid 	= threadIdx.x;
	unsigned int grid_sz	= gridDim.x*2*blockDim.x;
	unsigned int gid	= blockIdx.x*2*blockDim.x + 2*tid;
	INT_T 	pk, pj; 

	while ( (gid+1) < 2*nedges ) {
		pk = VERTEX_OWNER(edges[gid]);
		atomicAdd((ATOMIC_T*)&ne_per_proc[pk], 1);
		if ( edges[gid] != edges[gid+1] ){
			pj = VERTEX_OWNER(edges[gid+1]);	
			atomicAdd((ATOMIC_T*)&ne_per_proc[pj], 1);
		}
		gid += grid_sz;
	}
}

// Make send array with atomic add in global memory
__global__ void kernel2(INT_T *edges, INT_T nedges, INT_T *send_data, 
			INT_T *Offset, INT_T *ne_to_send, const int size)
{
	unsigned int tid 	= threadIdx.x;
	unsigned int grid_sz	= gridDim.x*2*blockDim.x;
	unsigned int gid	= blockIdx.x*2*blockDim.x + 2*tid;
	INT_T pk;
	INT_T pj; 
	INT_T index;
	INT_T ntosendk, ntosendj;

	while ( (gid+1) < 2*nedges ) 
	{
		pk = VERTEX_OWNER(edges[gid]);
		ntosendk = atomicAdd((ATOMIC_T*)&ne_to_send[pk], 2);
		index = Offset[pk] + ntosendk;
		send_data[index] = edges[gid];
		send_data[index+1] = edges[gid+1];
		
		/* avoid self loop */
		if ( edges[gid] != edges[gid+1] ){	
			pj = VERTEX_OWNER(edges[gid+1]);	
			ntosendj = atomicAdd((ATOMIC_T*)&ne_to_send[pj], 2);
			index = Offset[pj] + ntosendj;
			send_data[index] = edges[gid+1];
			send_data[index+1] = edges[gid];
		}
		gid += grid_sz;
	}
}

// The number of input edges, nedges, must be even, the extra variable "odd"
// tells the function if there is the need to add an extra edge at the end
// of the edge list.
__global__ void split_kernel(INT_T *in, INT_T nedges, int odd)
{
	unsigned int tid = threadIdx.x;
	unsigned int gid = blockDim.x*blockIdx.x + tid;
	unsigned int grid_size = gridDim.x*blockDim.x;

	// edges are of the form (u0,v0)(u1,v1)...
	INT_T v0; // if k < nedges read vk with k = 2*i + 1
	INT_T uK; // K = 2*i + nedges

	// Each thread read two elems and swap them
	while ((2*gid+1) < nedges) {
		v0 = in[2*gid + 1];
		uK = in[2*gid + nedges];

		in[2*gid + 1] = uK;
		in[2*gid + nedges] = v0;
	
		gid += grid_size;
	}
}

__global__ void k_count_proc(INT_T* in, INT_T nelems, INT_T* count_per_proc)
{
	unsigned int  tid = threadIdx.x;
	unsigned int  grid_sz = gridDim.x*blockDim.x;
	unsigned int  gid = blockIdx.x*blockDim.x + tid;

	INT_T p0, p1;

	if (gid == 0) {
		p0 = in[nelems-1];
		count_per_proc[2*p0+1] = nelems;
		count_per_proc[0] = 0;
	}

	while((gid+1) < nelems) {
		p0 = in[gid];
		p1 = in[gid+1];
		if (p0 != p1) {
			count_per_proc[2*p0+1] = (gid+1);
			count_per_proc[2*p1] = (gid+1);
		}
		gid += grid_sz;
	}
}

__global__ void k_count_vert(INT_T* in, INT_T nelems, INT_T* off_per_vert, INT_T nverts,
			     int rank, int size, int lgsize)
{
	unsigned int  tid = threadIdx.x;
	unsigned int  grid_sz = gridDim.x*blockDim.x;
	unsigned int  gid = blockIdx.x*blockDim.x + tid;

	INT_T UMAX = -1;
	INT_T U0, U1;
	INT_T nelems_in_offset = -1; // The actual number of elems in the offset array

	if (gid == 0) {
		UMAX = VERTEX_LOCAL(in[nelems-1]);
		nelems_in_offset = 2*UMAX + 1;
		if (2*nverts > nelems_in_offset) {
			off_per_vert[nelems_in_offset+1] = -1;
			off_per_vert[nelems_in_offset+2] = -1;
		}
		off_per_vert[nelems_in_offset] = nelems;
		off_per_vert[0] = 0;
	}

	while((gid+1) < nelems) {
		U0 = VERTEX_LOCAL(in[gid]);
		U1 = VERTEX_LOCAL(in[gid+1]);
		if (U0 != U1) {
			off_per_vert[2*U0+1] = (gid+1);
			off_per_vert[2*U1] = (gid+1);
		}
		gid += grid_sz;
	}
}

__global__ void back_kernel(INT_T *d_edges, INT_T nedges, 
			    INT_T *d_appo_edges_u, 
			    INT_T* d_appo_edges_v)
{
	unsigned int tid = threadIdx.x;
	unsigned int gid = blockDim.x*blockIdx.x + tid;
	unsigned int grid_size = gridDim.x*blockDim.x;
	INT_T idx;

	while (gid < nedges) {
		idx = d_appo_edges_v[gid];
		d_appo_edges_u[gid] = d_edges[idx];
		gid += grid_size;
	}
}

__global__ void unsplit_kernel(INT_T *array_u, INT_T* array_v, 
			       INT_T nelems, INT_T *array_uv)
{
	unsigned int tid 	= threadIdx.x;
	unsigned int gid 	= blockDim.x*blockIdx.x + tid;
	unsigned int grid_size 	= gridDim.x*blockDim.x;

	// array u and v are of the form u0,u1,u2...,v0,v1,v2...
	INT_T v0;
	INT_T u0;
	
	// tid=0 reads in[0]=u0 and in[N]=v0 and put them next
	while (gid < nelems) {
		u0 = array_u[gid];
		v0 = array_v[gid];
		array_uv[2*gid] = u0;
		array_uv[2*gid+1] = v0;
		gid += grid_size;
	}
}

// For each edge (u,v) add the edge (v,u) at the end of the 
// input edge list. If u == v add the special edge (-1, -1).
// (-1, -1) will be removed in the subsequent kernel
__global__ void k_add_vu(INT_T *d_edges, INT_T nedges)
{
	unsigned int tid 	= threadIdx.x;
	unsigned int gid 	= blockDim.x*blockIdx.x + tid;
	unsigned int grid_size 	= gridDim.x*blockDim.x;

	INT_T u,v;
	INT_T X = -1;
	
	while (gid < nedges) {
		u = d_edges[2*gid];
		v = d_edges[2*gid+1];
		if (u == v) {u=X; v=X;}
		d_edges[2*nedges + 2*gid] = v;
		d_edges[2*nedges + 2*gid + 1] = u;
		gid += grid_size;
	}
}

__global__ void k_degree(INT_T *d_offset, INT_T nverts, INT_T *d_degree)
{
	unsigned int tid = threadIdx.x;
	unsigned int gid = blockDim.x*blockIdx.x + tid;
	unsigned int grid_size = gridDim.x*blockDim.x;

	while (gid < nverts) {
		d_degree[gid] = d_offset[2*gid + 1] - d_offset[2*gid];
		gid += grid_size;
	}
}

__global__ void k_find_duplicates(INT_T *d_edges_u, INT_T *d_edges_v, 
				  INT_T nedges, unsigned int *stencil)
{
	unsigned int tid = threadIdx.x;
	unsigned int gid = blockDim.x*blockIdx.x + tid;
	unsigned int grid_size = gridDim.x*blockDim.x;
	INT_T U1,U2,V1,V2;
	
	while ((gid+1) < nedges) {
		U1 = d_edges_u[gid];
		U2 = d_edges_u[gid+1];
		V1 = d_edges_v[gid];
		V2 = d_edges_v[gid+1];

		stencil[gid+1] = ((V1 == V2) && (U1 == U2)) ?  (unsigned int)1 : 0;
		gid += grid_size;
	}
}

__global__ void k_owners(INT_T* d_edges_u, INT_T nedges, INT_T* d_edges_appo,
			 int rank, int size, int lgsize)
{
	unsigned int tid = threadIdx.x;
	unsigned int gid = blockDim.x*blockIdx.x + tid;
	unsigned int grid_size = gridDim.x*blockDim.x;
	INT_T U;

	while (gid < nedges) {
		U = d_edges_u[gid];
		d_edges_appo[gid] = VERTEX_OWNER(U);
		d_edges_appo[gid + nedges] = gid;
		gid += grid_size;
	}
}

__device__ INT32_T binsearch(INT_T *cells_i4, INT_T ci4, INT32_T ncells)
{
  INT_T min = 0;
  INT_T max = ncells-1;
  INT_T mid =(ncells-1) >> 1;

  while(min <= max)
  {
    if (cells_i4[mid] == ci4) return mid;
    if (cells_i4[mid] < ci4) min = mid+1;
    else max = mid-1;

    mid = (max + min) >> 1;
  }
  return mid;
}


__global__ void k_bitmask_edges(INT_T* d_edges, INT_T nedges, INT_T* d_unique_edges, INT_T *proc_offset, INT32_T* d_bitmask_pedges,
                                int rank, int size, int lgsize)
{
	unsigned int tid = threadIdx.x;
	unsigned int gid = blockDim.x*blockIdx.x + tid;
	unsigned int grid_size = gridDim.x*blockDim.x;
    INT32_T i;
    INT_T V, offset, range;
    int owner;

	while (gid < nedges) {
		V = d_edges[gid];
		owner = VERTEX_OWNER(V);
		offset = proc_offset[owner];
		range = proc_offset[owner+1]-proc_offset[owner];
		// Binsearch works for each specific processor
		i = binsearch(d_unique_edges+offset, V, range);
		d_bitmask_pedges[gid] = i + offset;

		gid += grid_size;
	}
}

__global__ void k_bitmask_verts(INT_T* d_unique_edges, INT_T offset, INT_T nelems, INT32_T* d_bitmask_pverts,
		                        int rank, int size, int lgsize)
{
	unsigned int tid = threadIdx.x;
	unsigned int gid = blockDim.x*blockIdx.x + tid;
	unsigned int grid_size = gridDim.x*blockDim.x;
    INT32_T VL;

	while (gid < nelems) {
		VL = VERTEX_2_LOCAL(d_unique_edges[gid+offset]);
		d_bitmask_pverts[VL] = (INT32_T)(gid+offset);
		gid += grid_size;
	}
}

__global__ void k_edge_owner(INT_T* d_edges, INT_T M, INT_T* d_mask_1, INT_T* d_mask_2,
			                 int rank, int size, int lgsize)
{
	unsigned int tid 	 = threadIdx.x;
	unsigned int grid_sz = gridDim.x*blockDim.x;
	unsigned int gid 	 = blockIdx.x*blockDim.x + tid;
	int owner;

	while (gid < M) {
		owner = VERTEX_OWNER(d_edges[gid]);
		d_mask_1[gid] = owner;
		d_mask_2[gid] = gid;
		gid += grid_sz;
	}
}

__global__ void k_count_proc(INT_T* d_edges_per_proc, INT_T* d_count_edges,
			                 int rank, int size, int lgsize)
{
	unsigned int tid 	 = threadIdx.x;
	unsigned int grid_sz = gridDim.x*blockDim.x;
	unsigned int gid 	 = blockIdx.x*blockDim.x + tid;

	while (gid < size) {
		d_edges_per_proc[gid] = (d_count_edges[2*gid+1] - d_count_edges[2*gid]);
		gid += grid_sz;
	}
}



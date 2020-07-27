/* This is to make PRId64 working with c++ compiler */
#ifdef __cplusplus
#define __STDC_FORMAT_MACROS
#endif
/* header of int64_t and PRId64 */
#include <inttypes.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include "header.h"
#include "defines.h"

//ATOMIC_T is defined in make_bfs_func.c before the inclusion of this file
__global__ void	k_make_queue_deg(INT_T *d_queue, INT_T queue_count, 
				 INT_T *d_queue_off, INT_T *d_queue_deg, 
				 INT_T *dg_off, INT_T nverts, 
				 int rank, int size, int lgsize)
{
	unsigned int  tid = threadIdx.x;
	unsigned int  grid_sz = gridDim.x*blockDim.x;
	unsigned int  gid = blockIdx.x*blockDim.x + tid;

	INT_T U;

	while (gid < queue_count)
	{
		//Vertices in d_queue are already LOCAL
		U = 2 * d_queue[gid];
		d_queue_off[gid] = dg_off[U];
		d_queue_deg[gid] = dg_off[U+1] - dg_off[U];

		gid += grid_sz; 
	}
}

__device__ INT_T binsearch(INT_T *cells_i4, INT_T ci4, INT_T ncells)
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


__device__ INT_T inter(INT_T * a, INT_T item, INT_T len){
      int e,s;
      s = 0;
      e = len - 1;
      int mid = 0;
      while(s < e){
        if (a[s] > item) return s-1;
        if (a[e] < item) return e;
        mid = s + ((item - a[s])*(e-s))/(a[e]-a[s]);
        if (a[mid] < item) s = mid+1;
        else if (a[mid] > item) e = mid-1;
        else   return mid;
      }
      if (s == e){
        if      (mid > s) return s;
        else if (mid < s) return e-1;
      }
      else{
           if          (mid == s) return e;
           else if     (mid == e) return s;
      }
      return -1;

}


__device__ INT_T binsearch_duplicates(INT_T *cells_i4, INT_T ci4, INT_T ncells)
{
  INT_T min = 0;
  INT_T max = ncells-1;
  INT_T mid = (max + min) >> 1;

  while(min <= max)
  {
    if (cells_i4[mid] == ci4) break; //return mid;
    if (cells_i4[mid] < ci4) min = mid+1;
    else max = mid-1;

    mid = (max + min) >> 1;
  }
  //If there are duplicates get the highest index
  while (mid < max && cells_i4[mid+1]==cells_i4[mid]){
	  mid++;
  }

      return mid;
}

__global__ void k_binary_mask_unique_large(INT_T next_level_vertices, INT_T *dg_edges,
			      INT_T *d_queue, INT_T *d_next_off, 
			      INT_T *d_queue_off, INT_T queue_count, 
			      INT_T *d_send_buff,
			      INT32_T *d_bitmask_pedges, INT32_T *d_mask,
			      int rank, int size, int lgsize)
{
	unsigned int  tid = threadIdx.x;
	unsigned int  grid_sz = gridDim.x*blockDim.x;
	unsigned int  gid = blockIdx.x*blockDim.x + tid;

	/* Even index*/
	INT_T i, newidx;
	INT_T V, off_V;
	INT32_T V_label_pointer;
    
        /* Odd index */
	INT_T i_i, newidx_i;
	INT_T V_i, off_V_i;
	INT32_T V_label_pointer_i;
	if (gid == 0 && ((next_level_vertices % 2) == 1)) {
		i = queue_count - 1;
		newidx = (next_level_vertices - 1) - d_next_off[i];
		off_V = d_queue_off[i] + newidx;
		V_label_pointer = d_bitmask_pedges[off_V];
		if (NO_PREDECESSOR == d_mask[V_label_pointer]) {
			d_mask[V_label_pointer] = d_queue[i]; // Update the mask with the predecessor
			V = dg_edges[off_V]; // This vertex is GLOBAL
			d_send_buff[V_label_pointer] = V;
		}
	}
        while (gid < (next_level_vertices/2)) {
                if (queue_count == 1) {
                        i = 0;
                        i_i = 0;
                }
                else{
                        if (gid * 2 < queue_count) { 
                                i = binsearch(d_next_off, (gid * 2), (gid * 2) + 1);
                        } else
                                i = binsearch(d_next_off, (gid * 2), queue_count);
                        if ((gid * 2) + 1 > d_next_off[queue_count - 1]) {
                                i_i = queue_count - 1;
                        } else if (((gid * 2) + 1) >= d_next_off[i + 1]) {
                                i_i = i + 1;
                        } else
                                i_i = i;
                }
                newidx = (gid * 2) - d_next_off[i];
                newidx_i = ((gid * 2) + 1) - d_next_off[i_i];
                off_V = d_queue_off[i] + newidx;
                off_V_i = d_queue_off[i_i] + newidx_i;
                // Check the bitmask
                V_label_pointer = d_bitmask_pedges[off_V];
                V_label_pointer_i = d_bitmask_pedges[off_V_i];
                if (NO_PREDECESSOR == d_mask[V_label_pointer]) {
                        d_mask[V_label_pointer] = d_queue[i]; // Update the mask with the predecessor
                        V = dg_edges[off_V]; // This vertex is GLOBAL
                        d_send_buff[V_label_pointer] = V;
                }
                if (NO_PREDECESSOR == d_mask[V_label_pointer_i]) {
                        d_mask[V_label_pointer_i] = d_queue[i_i]; // Update the mask with the predecessor
                        V_i = dg_edges[off_V_i]; // This vertex is GLOBAL
                        d_send_buff[V_label_pointer_i] = V_i;
                }
                gid += grid_sz;
        }	
}

__global__ void k_binary_mask_unique(INT_T next_level_vertices, INT_T *dg_edges,
			       INT_T *d_queue, INT_T *d_next_off, 
			       INT_T *d_queue_off, INT_T queue_count, 
			       INT_T *d_send_buff,
			       INT32_T *d_mask_pointer, INT32_T *d_mask,
			       int rank, int size, int lgsize)
{
	unsigned int  tid = threadIdx.x;
	unsigned int  grid_sz = gridDim.x*blockDim.x;
	unsigned int  gid = blockIdx.x*blockDim.x + tid;

	INT_T i, newidx;
	INT_T V, off_V;

	INT32_T V_label_pointer;

	while (gid < next_level_vertices) {
		if (queue_count == 1) {
			i = 0;
		} else {
#ifdef INT
			i = inter(d_next_off, gid, queue_count);
#else                  
			if (gid < queue_count)
				i = binsearch(d_next_off, gid, gid + 1);
			else
				i = binsearch(d_next_off, gid, queue_count);
#endif    
		}

		newidx = gid - d_next_off[i];
		off_V = d_queue_off[i] + newidx;

		// Check the bitmask
		V_label_pointer = d_mask_pointer[off_V];

		if (NO_PREDECESSOR == d_mask[V_label_pointer]) {
			d_mask[V_label_pointer] = d_queue[i]; // Update the mask with the predecessor
			V = dg_edges[off_V]; // This vertex is GLOBAL
			d_send_buff[gid] = V; //This is the next level vertex GLOBAL
		} else {
			d_send_buff[gid] = VALUE_TO_REMOVE_BY_MASK;
		}

		gid += grid_sz;
	}
}

__global__ void k_bfs_owner(INT_T* d_sendbuff, INT_T M, 
			    INT_T* d_mask_1, INT_T* d_mask_2, 
			    int rank, int size, int lgsize)
{
	unsigned int tid 	 = threadIdx.x;
	unsigned int grid_sz = gridDim.x*blockDim.x;
	unsigned int gid 	 = blockIdx.x*blockDim.x + tid;
	int owner;

	while (gid < M) {
		owner = VERTEX_OWNER(d_sendbuff[gid]);
		d_mask_1[gid] = owner;
		d_mask_2[gid] = gid;
		gid += grid_sz;
	}
}

__global__ void k_bfs_count_proc(INT_T* in, INT_T nelems, INT_T* count_per_proc)
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

__global__ void bfs_back_kernel(INT_T *d_array, INT_T nelems, INT_T *d_support, INT_T* d_idx)
{
	unsigned int tid 	= threadIdx.x;
	unsigned int gid 	= blockDim.x*blockIdx.x + tid;
	unsigned int grid_size 	= gridDim.x*blockDim.x;
	INT_T idx;

	while (gid < nelems) {
		idx = d_idx[gid];
		d_support[gid] = d_array[idx];
		gid += grid_size;
	}
}

__global__ void bfs_back_kernel32(INT_T *d_array, INT_T nelems, INT_T* d_idx, INT32_T *d_support,
		                          int rank, int size, int lgsize)
{
	unsigned int tid 	= threadIdx.x;
	unsigned int gid 	= blockDim.x*blockIdx.x + tid;
	unsigned int grid_size 	= gridDim.x*blockDim.x;
	INT_T idx;
	INT32_T u0;

	while (gid < nelems) {
		idx = d_idx[gid];
		u0 = VERTEX_2_LOCAL(d_array[idx]);
		d_support[gid] = u0;
		gid += grid_size;
	}
}

__global__ void bfs_back_kernel32_pad(INT_T *d_array, INT_T nelems,
		                              INT_T *d_offset, INT_T *d_padded_offset,
		                              INT_T* d_idx, INT32_T *d_support,
		                              int rank, int size, int lgsize)
{
	unsigned int tid 	= threadIdx.x;
	unsigned int gid 	= blockDim.x*blockIdx.x + tid;
	unsigned int grid_size 	= gridDim.x*blockDim.x;
	INT_T idx, lidx, g1, off;
	INT32_T u0;
	int node_rank;

	while (gid < nelems) {
		node_rank = binsearch_duplicates(d_padded_offset, gid, size);

		g1 = gid - d_padded_offset[node_rank];
		off = d_offset[node_rank];
		lidx = ((g1 < (d_offset[node_rank+1]-off)) ? g1 :  0);

		idx = d_idx[off+lidx];
		u0 = VERTEX_2_LOCAL(d_array[idx]);
		d_support[gid] = u0;
		gid += grid_size;
	}
}


__global__ void k_dequeue_step_8_local(INT32_T* d_sendbuff, INT_T nlocal_verts, INT_T* d_newq, int64_t* d_pred, ATOMIC_T* global_count,
                                       int rank, int size, int lgsize)
{
  unsigned int  tid = threadIdx.x;
  unsigned int  grid_sz = gridDim.x*blockDim.x;
  unsigned int  gid = blockIdx.x*blockDim.x + tid;

  ATOMIC_T old_count = 0;
  INT32_T VL;

  while (gid < nlocal_verts)
  {
    //Vertex is already LOCAL
    VL = d_sendbuff[gid]; // new vertex is already LOCAL

    if (d_pred[VL] == NO_PREDECESSOR)
    {
      d_pred[VL] = rank;  // Store the rank of the processor that found the new vertex
      old_count = atomicAdd((ATOMIC_T*)&global_count[0], 1);
      //New queue contains LOCAL vertex.
      d_newq[old_count] = VL;
    }
    gid += grid_sz;
  }
}

//Parameter d_recv_offset_per_proc is used to determine rank of each parent edge and calculate GLOBAL vertex
__global__ void k_dequeue_step_9_recv_1(INT32_T* d_recvbuff, INT_T recv_count, INT_T* d_recv_offset_per_proc,
					                    INT_T* d_newq, int64_t* d_pred,
					                    int rank, int size, int lgsize)
{
	unsigned int  tid = threadIdx.x;
	unsigned int  grid_sz = gridDim.x*blockDim.x;
	unsigned int  gid = blockIdx.x*blockDim.x + tid;

    unsigned int  node_rank;
	INT32_T V;

	//Usually U is the vertex and V is the parent, here they were swapped. Fixed it.
	while (gid  < recv_count)
	{
		//Vertex in d_recvbuff is LOCAL
		V = d_recvbuff[gid]; //This is the new vertex already LOCAL

		if (d_pred[V] == NO_PREDECESSOR)
		{
			d_pred[V] = TMP_PREDECESSOR;
			//Calculate the node_rank of the received vertex using binary search
			node_rank = binsearch_duplicates(d_recv_offset_per_proc, gid, size);
			d_newq[V] = node_rank;  // Store the node rank
		}

		gid += grid_sz;
	}
}

__global__ void k_dequeue_step_9_recv_2(INT_T* d_newq, INT_T* d_oldq, int64_t* d_pred, INT_T g_nverts,
		                                INT32_T* d_pverts, INT32_T* d_mask, ATOMIC_T* global_count,
					                    int rank, int size, int lgsize)
{
   unsigned int  tid = threadIdx.x;
   unsigned int  grid_sz = gridDim.x*blockDim.x;
   unsigned int  gid = blockIdx.x*blockDim.x + tid;
   ATOMIC_T old_count = 0;

   INT32_T label;

   while ( gid < g_nverts )
   {
      if (d_pred[gid] == TMP_PREDECESSOR)
      {
         d_pred[gid] = d_newq[gid];  // This is the node rank
         old_count = atomicAdd((ATOMIC_T*)&global_count[0], 1);
         //Vertex is LOCAL in the new queue
         d_oldq[old_count] = gid; // This is the new queue
         label = d_pverts[gid];   // Get the label (label will be -1 if vertices does NOT have connections within the same node
         d_mask[label] = OTHER_RANK;
      }

      gid += grid_sz;
   }
}

__global__ void k_unique_local(INT32_T* d_sendbuff, INT_T nlocal_verts, INT_T* d_newq, int64_t* d_pred, 
			                   int rank, int size, int lgsize)
{
  unsigned int  tid = threadIdx.x;
  unsigned int  grid_sz = gridDim.x*blockDim.x;
  unsigned int  gid = blockIdx.x*blockDim.x + tid;

  INT32_T VL;

  while (gid < nlocal_verts)
  {
    //Vertex is already LOCAL
    VL = d_sendbuff[gid]; // new vertex is already LOCAL

    if (d_pred[VL] == NO_PREDECESSOR)
    {
      d_pred[VL] = TMP_PREDECESSOR;
    }
    gid += grid_sz;
  }
}

__global__ void k_atomic_enqueue_local(INT_T* d_newq, INT_T* d_oldq,
                                       int64_t* d_pred, INT_T g_nverts,
                                       ATOMIC_T* global_count, int rank,
                                       int size, int lgsize)
{                                       
  unsigned int  tid = threadIdx.x;
  unsigned int  grid_sz = gridDim.x*blockDim.x;
  unsigned int  gid = blockIdx.x*blockDim.x + tid;

  ATOMIC_T old_count = 0;
  
  while ( gid < g_nverts )
  {
    if ( d_pred[gid] == TMP_PREDECESSOR)
    {
      d_pred[gid] = rank;  // This is the predecessor
      old_count = atomicAdd((ATOMIC_T*)&global_count[0], 1);
      //Vertex is LOCAL in the new queue
      d_oldq[old_count] = gid; // This is the new queue
    } 
    
    gid += grid_sz;
  } 
} 

__global__ void k_remove_pred(int64_t* d_pred, INT_T* d_mask_1, INT_T* d_mask_2, INT_T nelems,
		                        int rank, int size, int lgsize)
{
	unsigned int tid 	 = threadIdx.x;
	unsigned int grid_sz = gridDim.x*blockDim.x;
	unsigned int gid 	 = blockIdx.x*blockDim.x + tid;

	while (gid < nelems) {
		if (d_pred[gid] < 0) {
			d_mask_1[gid] = NO_PREDECESSOR;
			d_mask_2[gid] = NO_PREDECESSOR;
		}
		else {
			d_mask_1[gid] = d_pred[gid];
			d_mask_2[gid] = gid;
		}
		gid += grid_sz;
	}
}

__global__ void k_bfs_copy32(INT_T* d_array, INT_T nelems, INT32_T* d_buffer32)
{
	unsigned int tid 	 = threadIdx.x;
	unsigned int grid_sz = gridDim.x*blockDim.x;
	unsigned int gid 	 = blockIdx.x*blockDim.x + tid;

	while (gid < nelems) {
		d_buffer32[gid] = (INT32_T)d_array[gid];
		gid += grid_sz;
	}
}


__global__ void k_bfs_pred_local(INT32_T *d_buffer32, INT_T nelems, INT32_T *d_pverts, INT32_T* d_mask, int64_t* d_pred,
		                         int rank, int size, int lgsize)
{
	unsigned int tid 	 = threadIdx.x;
	unsigned int grid_sz = gridDim.x*blockDim.x;
	unsigned int gid 	 = blockIdx.x*blockDim.x + tid;

	INT32_T V, label, predVL;

	while (gid < nelems) {
		V = d_buffer32[gid];   // LOCAL
		label = d_pverts[V];
		predVL = d_mask[label]; // Predecessor (local)
		d_pred[V] = VERTEX_TO_GLOBAL(predVL);
		gid += grid_sz;
	}
}

__global__ void k_bfs_pred_recv(INT32_T *d_buffer32, INT_T nelems, INT_T* d_recv_offset_per_proc,
		                        INT_T *d_unique_edges, INT_T *proc_offset, INT32_T *d_pedges, INT32_T* d_mask,
		                        int rank, int size, int lgsize)
{
	unsigned int tid 	 = threadIdx.x;
	unsigned int grid_sz = gridDim.x*blockDim.x;
	unsigned int gid 	 = blockIdx.x*blockDim.x + tid;

	INT32_T V, label;
	INT_T VG, offset, range;
	int node_rank;

	while (gid < nelems) {
		V = d_buffer32[gid]; // This is LOCAL
		// Find Node Rank
		node_rank = binsearch_duplicates(d_recv_offset_per_proc, gid, size);
		VG = VERTEX_2_GLOBAL(V, node_rank); // Convert to Global

		offset = proc_offset[node_rank];
		range = proc_offset[node_rank+1]-proc_offset[node_rank];

		// Find the Global vertex label within the Unique Edges
		label = binsearch(d_unique_edges + offset, VG, range) + offset;

		// Use the label to get the predecessor from the d_mask
		d_buffer32[gid] = d_mask[label]; // Predecessor (local)
		gid += grid_sz;
	}
}

__global__ void k_bfs_pred_remote(INT32_T *d_buffer32, INT_T nelems, INT_T *d_mask_1, INT_T* d_mask_2, int64_t* d_pred,
		                         int rank, int size, int lgsize)
{
	unsigned int tid 	 = threadIdx.x;
	unsigned int grid_sz = gridDim.x*blockDim.x;
	unsigned int gid 	 = blockIdx.x*blockDim.x + tid;

	INT32_T V, predVL;
	int node_rank;

	while (gid < nelems) {
		node_rank = d_mask_1[gid];
		if (node_rank != rank) {
			predVL = d_buffer32[gid];
			V = d_mask_2[gid];
			d_pred[V] = VERTEX_2_GLOBAL(predVL, node_rank);
		}

		gid += grid_sz;
	}
}


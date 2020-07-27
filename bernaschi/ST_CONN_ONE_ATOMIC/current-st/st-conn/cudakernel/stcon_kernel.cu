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
#include "mykernels.h"

/*
ST-CON DEVICE FUNCTION
 */

__device__ INT32_T vertex_2_local(INT_T x, int lgsize){
    INT32_T color[2] = {0, COLOR_MASK};
    INT32_T y = (INT32_T)(DIV_SIZE(x));
    return y | color[((x & COLOR_MASK_64) == COLOR_MASK_64)];
}
/*
__device__ int set_blue_mask32(int x){
        return (x | COLOR_MASK);
}

__device__ int set_red_mask32(int x){
  return (x & (~COLOR_MASK));
}


__device__ int isblue(int x){
    return (x & COLOR_MASK) == COLOR_MASK;
}

__device__ int isblue_64(INT_T x){

  return (x & COLOR_MASK_64) == COLOR_MASK_64;
}
*/
__device__ INT_T st_binsearch(INT_T *cells_i4, INT_T ci4, INT_T ncells)
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

__device__ INT_T st_binsearch_duplicates(INT_T *cells_i4, INT_T ci4, INT_T ncells)
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

//ATOMIC_T is defined in make_bfs_func.c before the inclusion of this file
__global__ void	k_stcon_make_queue_deg(INT_T *d_queue, INT_T queue_count,
				 INT_T *d_queue_off, INT_T *d_queue_deg, 
				 INT_T *dg_off, INT_T nverts, 
				 int rank, int size, int lgsize)
{
	unsigned int  tid = threadIdx.x;
	unsigned int  grid_sz = gridDim.x*blockDim.x;
	unsigned int  gid = blockIdx.x*blockDim.x + tid;

	INT_T U, dq;

	while (gid < queue_count)
	{
		dq = d_queue[gid] & (~COLOR_MASK_64);
		//Vertices in d_queue are already LOCAL
		U = 2 * dq;
                d_queue_off[gid] = dg_off[U];
		d_queue_deg[gid] = dg_off[U+1] - dg_off[U];

		gid += grid_sz; 
	}
}

__global__ void k_stcon_owner(INT_T* d_sendbuff, INT_T M,
			    INT_T* d_mask_1, INT_T* d_mask_2,
			    int rank, int size, int lgsize)
{
	unsigned int tid 	 = threadIdx.x;
	unsigned int grid_sz = gridDim.x*blockDim.x;
	unsigned int gid 	 = blockIdx.x*blockDim.x + tid;
	int owner;
	INT_T temp;

	while (gid < M) {
		temp = d_sendbuff[gid]  & (~COLOR_MASK_64); // Remove the color from the vertex
		owner = VERTEX_OWNER(temp);
		d_mask_1[gid] = owner;
		d_mask_2[gid] = gid;
		gid += grid_sz;
	}
}


//devo anche passare d_pred 
//node_rank_1 =  rank (aka rank of d_queue[i])
//node_rank_2 = d_pred[unique_edges[V_label_pointer]]
__global__ void k_stcon_binary_mask_unique_large(INT_T next_level_vertices, INT_T *dg_edges,
                                  INT_T *d_queue, INT_T *d_next_off,
                                  INT_T *d_queue_off, INT_T queue_count,
                                  INT_T *d_send_buff,
                                  INT32_T *d_bitmask_pedges, INT32_T *d_mask,
                                  int rank, int size, int lgsize,
                                  INT_T *d_st_rank, INT_T *d_pred, INT_T *d_unique_edges, INT32_T* d_mn_found)
{
    


     unsigned int  tid = threadIdx.x;
     unsigned int  grid_sz = gridDim.x*blockDim.x;
     unsigned int  gid = blockIdx.x*blockDim.x + tid;
     INT_T i, newidx;
     INT_T V, off_V, V_colored;
     INT32_T V_label_pointer;
     INT_T parent_vertex, parent_vertex_color; //needed to store (in case) two partent with different color (exit condiction)
     INT32_T  V_parent_32;
     INT32_T COLOR_32[2] = {0, COLOR_MASK};
     INT_T COLOR_64[2] = {0, COLOR_MASK_64};
     INT32_T OLD;
     while (gid < next_level_vertices) {
		 if (queue_count == 1) i = 0;
		 else{
			 i = st_binsearch(d_next_off, gid, queue_count);
		 }
		 newidx  = gid - d_next_off[i];
		 off_V   = d_queue_off[i] + newidx;
		 V_label_pointer = d_bitmask_pedges[off_V];
		 V = dg_edges[off_V];
		 // Check the bitmask
		 parent_vertex = d_queue[i] & (~COLOR_MASK_64); //controllato ok
		 parent_vertex_color = d_queue[i] & (COLOR_MASK_64); //controllato ok
                 short color_idx = (parent_vertex_color == COLOR_MASK_64);

                 V_parent_32  = ((INT32_T)parent_vertex) | COLOR_32[color_idx ];
                 V_colored = V | COLOR_64[color_idx];
                //SET PREDECESSOR (D_QUEUE IS THE PARTENT) IF THE NODE IS UNVISITED
                if (d_mask[V_label_pointer] ==  NO_PREDECESSOR){
                            OLD = atomicCAS( &d_mask[V_label_pointer], NO_PREDECESSOR, V_parent_32) ;
                }else{
                            OLD = d_mask[V_label_pointer];
                }
                if (OLD == NO_PREDECESSOR){
                    d_send_buff[V_label_pointer] = V_colored;
                    //printf ("OLD %d\n",OLD);
                }
                else if ((OLD & COLOR_MASK) != (V_parent_32 & (COLOR_MASK))){ 
                         if (atomicCAS(d_mn_found, 0, 1) == 0) {
					 d_st_rank[0] = rank; //rank of d_queue[i] red
					 if (NO_PREDECESSOR ==  d_pred[VERTEX_LOCAL(d_unique_edges[V_label_pointer])] ) d_st_rank[1] = rank; //d_pred is not yet updated becouse mn is found in the same bfs lev.
					 else d_st_rank[1] = d_pred[VERTEX_LOCAL(d_unique_edges[V_label_pointer])] & (~COLOR_MASK_64); //rank of pred blue
					 d_st_rank[2] = V; //VERTEX_TO_GLOBAL(V); //dg_edges[off_V];
					 d_st_rank[3] = V;//VERTEX_TO_GLOBAL(V); //dg_edges[off_V];
                                         if (color_idx){
                                             d_st_rank[5] = VERTEX_TO_GLOBAL(d_queue[i] & (~COLOR_MASK_64));
                                             d_st_rank[4] = d_mask[V_label_pointer] & (~COLOR_MASK); // | COLOR_MASK_64; //va convertito a 64 bit il colore anche sotto
                                         } //printf("!!!!!!!!!! blue %ld - rosso %d --- %ld\n", d_queue[i]&(~COLOR_MASK_64), d_mask[V_label_pointer], V);
                                         else{
                                             d_st_rank[4] = VERTEX_TO_GLOBAL(d_queue[i] & (~COLOR_MASK_64));
                                             d_st_rank[5] = d_mask[V_label_pointer] & (~COLOR_MASK);
                                         }
			}                }
                //printf("num di cas %d\n", i);
		 gid += grid_sz;
   } // End While gid

}

//bfs_back_kernel32
__global__ void k_stcon_back_kernel32(INT_T *d_array, INT_T nelems, INT_T* d_idx, INT32_T *d_support,
		                          int rank, int size, int lgsize)
{
	unsigned int tid 	= threadIdx.x;
	unsigned int gid 	= blockDim.x*blockIdx.x + tid;
	unsigned int grid_size 	= gridDim.x*blockDim.x;
	INT_T idx;
	INT32_T u0;

	while (gid < nelems) {
		idx = d_idx[gid];
//		u0 = VERTEX_2_LOCAL(d_array[idx]); //la macro passa da 64 a 32
                u0 = vertex_2_local(d_array[idx], lgsize) ;
		d_support[gid] = u0;
		gid += grid_size;
	}
}


/*
MODIFICA NO PETERSON 
   */
__global__ void k_stcon_dequeue_step_8_local(INT32_T* d_sendbuff, INT_T nlocal_verts, INT_T* d_newq, int64_t* d_pred, ATOMIC_T* global_count,
                                       int rank, int size, int lgsize)
{
  unsigned int  tid = threadIdx.x;
  unsigned int  grid_sz = gridDim.x*blockDim.x;
  unsigned int  gid = blockIdx.x*blockDim.x + tid;

  ATOMIC_T old_count = 0;
  INT32_T VL;
  INT32_T vertex;
  INT_T vertex_color, new_vertex;
  INT_T color[2] = {0, COLOR_MASK_64};
  while (gid < nlocal_verts)
  {
    //Vertex is already LOCAL
    VL = d_sendbuff[gid]; // new vertex is already LOCAL
    
    vertex = VL & (~COLOR_MASK);
    //printf( "--------------> k_dequeue_step_8_local  ------ VL %d    vertex %d    d_pred[vertex] %ld\n", VL, vertex, d_pred[vertex]);

    // estraggo il colore a 32bit e setto a 64bit 
    vertex_color = color[((VL & COLOR_MASK) == COLOR_MASK)]; //uso il confronto a 32bit
    if (d_pred[vertex] == NO_PREDECESSOR)
    {
      d_pred[vertex] = rank | vertex_color;/// <-  settare il colore vecchio qui pero' devi usare color_mask_64   // Store the rank of the processor that found the new vertex
      old_count = atomicAdd((ATOMIC_T*)&global_count[0], 1);
      //New queue contains LOCAL vertex.
      //d_newq[old_count] = VL; // No change
      new_vertex = (INT_T)vertex;
      d_newq[old_count] = new_vertex | vertex_color;
    }
    gid += grid_sz;
  }
}
__device__ INT32_T mn_recv = -1;
//Parameter d_recv_offset_per_proc is used to determine rank of each parent edge and calculate GLOBAL vertex
__global__ void k_stcon_dequeue_step_9_recv_1(INT32_T* d_recvbuff, INT_T recv_count, INT_T* d_recv_offset_per_proc,
					                    INT_T* d_newq, int64_t* d_pred,
					                    int rank, int size, int lgsize,
					                    INT_T *d_st_rank, INT32_T* d_mn_found)

    
    

{
	unsigned int  tid = threadIdx.x;
	unsigned int  grid_sz = gridDim.x*blockDim.x;
	unsigned int  gid = blockIdx.x*blockDim.x + tid;

    INT_T  node_rank;
    INT_T node_rank_1 = -1; // questi sarranno global
    INT32_T V, V_color;
    //INT_T V_parent, V_parent_color;
    INT_T COLOR_64[2] = {0, COLOR_MASK_64};
    INT_T V_color_64;
    INT_T OLD = -1;
    INT_T temp_color = 0;
	//Usually U is the vertex and V is the parent, here they were swapped. Fixed it.
	while (gid  < recv_count)
	{
		//Vertex in d_recvbuff is LOCAL
		V = d_recvbuff[gid] & (~COLOR_MASK); //This is the new vertex already LOCAL
		V_color = d_recvbuff[gid] & (COLOR_MASK); 
 
                V_color_64 = COLOR_64[(V_color == COLOR_MASK)];
                if (d_pred[V] == NO_PREDECESSOR ){
                        OLD = atomicCAS((unsigned long long*)&d_pred[V], NO_PREDECESSOR, (TMP_PREDECESSOR | V_color_64)); 
                }
                else{
                    OLD = d_pred[V];
                }

                if (OLD == NO_PREDECESSOR){
                    node_rank = st_binsearch_duplicates(d_recv_offset_per_proc, gid, size);
                    d_newq[V] = node_rank | V_color_64;
                }
                else{
                    temp_color = OLD & COLOR_MASK_64;
                    OLD = OLD & (~COLOR_MASK_64);
                    if (OLD != TMP_PREDECESSOR){
                        temp_color = d_pred[V] & ( COLOR_MASK_64);
                        node_rank_1 = d_pred[V] & (~COLOR_MASK_64);
                    }
                    if (temp_color != V_color_64 ){
                           if (atomicCAS(d_mn_found, 0, 1) == 0) { // QUI HO UNA SEZIONE CRITICA
//printf("recv ESCO con %ld with color %ld e %ld OLD = %ld (%ld)\n", VERTEX_TO_GLOBAL(V), temp_color, V_color_64, OLD, (OLD & ~COLOR_MASK_64));
                                            //exit with two parant rank
                                            d_st_rank[1] = node_rank_1;
                                            d_st_rank[0] = st_binsearch_duplicates(d_recv_offset_per_proc, gid, size) ;//node_rank_2;//recv node. Here is local
                                            d_st_rank[2] = VERTEX_TO_GLOBAL(V);
                                            d_st_rank[3] = VERTEX_TO_GLOBAL(V);
                                            d_st_rank[4] = NO_PREDECESSOR;
                                            d_st_rank[5] = NO_PREDECESSOR;
                                            mn_recv = V;
                            }

                    }
                    
                }
		gid += grid_sz;
	}
}

__global__ void k_stcon_dequeue_step_9_recv_2(INT_T* d_newq, INT_T* d_oldq, int64_t* d_pred, INT_T g_nverts,
		                                INT32_T* d_pverts, INT32_T* d_mask, ATOMIC_T* global_count,
					                    int rank, int size, int lgsize, INT_T *d_st_rank)
{
    /*debug */
   unsigned int  tid = threadIdx.x;
   unsigned int  grid_sz = gridDim.x*blockDim.x;
   unsigned int  gid = blockIdx.x*blockDim.x + tid;
   ATOMIC_T old_count = 0;
   INT32_T color[2] = {0, COLOR_MASK};
   INT32_T label;
   INT_T vertex_color;
   
   while ( gid < g_nverts )
   {
      if ( (d_pred[gid] & (~COLOR_MASK_64)) ==  TMP_PREDECESSOR )
      {
         vertex_color = d_pred[gid] & (COLOR_MASK_64);
         d_pred[gid] = d_newq[gid]; // | vertex_color;  // This is the node rank

         old_count = atomicAdd((ATOMIC_T*)&global_count[0], 1);
         //Vertex is LOCAL in the new queue
         d_oldq[old_count] = gid | vertex_color ; // This is the new queue
         label = d_pverts[gid];   // Get the label (label will be -1 if vertices does NOT have connections within the same node
         
         d_mask[label] = OTHER_RANK | color[(vertex_color == COLOR_MASK_64) ];
        // printf( "[d_mask k_dequeue]_step_9_recv_2 %d\n",d_mask[label]);
         if (mn_recv >= 0){
            if (gid == mn_recv){
                if (d_st_rank[1] == -1)
                    d_st_rank[1] = d_newq[gid] & (~COLOR_MASK_64); //d_newq[gid] & (~COLOR_MASK_64);
            }
         }
      }
      if (mn_recv >= 0){
            if (gid == mn_recv){
                if (d_st_rank[1] == -1)
                    d_st_rank[1] = d_newq[gid] & (~COLOR_MASK_64); //d_newq[gid] & (~COLOR_MASK_64);
            }
      }
     // printf("------------- %d\n", mn_recv);
            gid += grid_sz;
   }
}


__global__ void k_stcon_remove_pred(int64_t* d_pred, INT_T* d_mask_1, INT_T* d_mask_2, INT_T nelems,
		                        int rank, int size, int lgsize)
{
	unsigned int tid 	 = threadIdx.x;
	unsigned int grid_sz = gridDim.x*blockDim.x;
	unsigned int gid 	 = blockIdx.x*blockDim.x + tid;
	while (gid < nelems) {

		if (d_pred[gid] == NO_PREDECESSOR) {
			d_mask_1[gid] = NO_PREDECESSOR;
			d_mask_2[gid] = NO_PREDECESSOR;
		}
		else {
			d_mask_1[gid] = d_pred[gid] & (~COLOR_MASK_64);  // Remove color from the rank
			d_mask_2[gid] = gid; //| (d_pred[gid] & COLOR_MASK_64);
		}
		d_pred[gid] = NO_PREDECESSOR;

		gid += grid_sz;
	}
}


__global__ void k_stcon_pred_recv(INT32_T *d_buffer32, INT_T nelems, INT_T* d_recv_offset_per_proc,
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
		node_rank = st_binsearch_duplicates(d_recv_offset_per_proc, gid, size);

        //V senza colore
                V = V & (~COLOR_MASK);
		VG = VERTEX_2_GLOBAL(V, node_rank); // Convert to Global

		offset = proc_offset[node_rank];
		range = proc_offset[node_rank+1]-proc_offset[node_rank];

		// Find the Global vertex label within the Unique Edges   #?
		label = st_binsearch(d_unique_edges + offset, VG, range) + offset;
		//printf( "nk_bfs_pred_recv: d_mask %d\n", d_mask[label]);
		// Use the label to get the predecessor from the d_mask
		d_buffer32[gid] = d_mask[label]; // Predecessor (local) // d_buffer get d_mask[label] with color
		gid += grid_sz;
	}
}


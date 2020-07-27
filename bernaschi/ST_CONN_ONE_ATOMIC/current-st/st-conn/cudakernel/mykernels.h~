#ifndef _H_MY_KERNELS_
#define _H_MY_KERNELS_

__global__ void kernel1(INT_T *edges, INT_T nedges, INT_T *ne_per_proc, const int size);

__global__ void kernel2(INT_T *edges, INT_T nedges, INT_T *send_data, INT_T *Offset, INT_T *ne_to_send, const int size);

__global__ void split_kernel(INT_T *in_edge_list, INT_T nedge, int odd);

__global__ void k_count_proc(INT_T* in, INT_T nelems, INT_T* count_per_proc);

__global__ void k_count_vert(INT_T* in, INT_T nelems, INT_T* off_per_vert, INT_T nverts,
			                 int rank, int size, int lgsize);

__global__ void back_kernel(INT_T *d_edges, INT_T nedges, INT_T *d_appo_edges_u, INT_T* d_appo_edges_v);

__global__ void unsplit_kernel(INT_T *array_u, INT_T* array_v, INT_T nelems, INT_T *array_uv);

__global__ void k_add_vu(INT_T *d_edges, INT_T nedges);

__global__ void k_degree(INT_T *d_offset, INT_T nverts, INT_T *d_degree);

__global__ void k_find_duplicates(INT_T *d_edges_u, INT_T *d_edges_v, INT_T nedges, unsigned int *stencil);

__global__ void	k_make_queue_deg(INT_T *d_queue, INT_T queue_count, 
				                 INT_T *d_queue_off, INT_T *d_queue_deg,
				                 INT_T *dg_off, INT_T nverts,
				                 int rank, int size, int lgsize);

__global__ void k_binary_mask_unique(INT_T next_level_vertices, INT_T *dg_edges,
			                         INT_T *d_queue, INT_T *d_next_off,
			                         INT_T *d_queue_off, INT_T queue_count,
			                         INT_T *d_send_buff,
			                         INT32_T *d_mask_pointer, INT32_T *d_mask,
			                         int rank, int size, int lgsize);

__global__ void k_binary_mask_unique_large(INT_T next_level_vertices, INT_T *dg_edges,
			                               INT_T *d_queue, INT_T *d_next_off,
			                               INT_T *d_queue_off, INT_T queue_count,
			                               INT_T *d_send_buff,
			                               INT32_T *d_mask_pointer, INT32_T *d_mask,
			                               int rank, int size, int lgsize);

__global__ void k_owners(INT_T* d_edges_u, INT_T h_nedges, INT_T* d_edges_appo, 
			             int rank, int size, int lgsize);

__global__ void k_edge_owner(INT_T* d_edges, INT_T M, INT_T* d_mask_1, INT_T* d_mask_2,
		                    int rank, int size, int lgsize);

__global__ void k_bfs_owner(INT_T* d_sendbuff, INT_T M, INT_T* d_mask_1, INT_T* d_mask_2,
		                    int rank, int size, int lgsize);

__global__ void k_bfs_count_proc(INT_T* in, INT_T nelems, INT_T* count_per_proc);

__global__ void bfs_back_kernel(INT_T *d_array, INT_T nelems, INT_T *d_support, INT_T* d_idx);

__global__ void bfs_back_kernel32(INT_T *d_array, INT_T nelems, INT_T* d_idx, INT32_T *d_support,
		                          int rank, int size, int lgsize);

__global__ void bfs_back_kernel32_pad(INT_T *d_array, INT_T nelems,
		                              INT_T *d_offset, INT_T *d_padded_offset,
		                              INT_T* d_idx, INT32_T *d_support,
		                              int rank, int size, int lgsize);

__global__ void k_dequeue_step_8_local(INT32_T* d_sendbuff, INT_T nlocal_verts,
                                      INT_T* d_newq, int64_t* d_pred,
                                      ATOMIC_T* global_count,
                                      int rank, int size, int lgsize);

__global__ void k_dequeue_step_9_recv_1(INT32_T* d_recvbuff, INT_T recv_count, INT_T* d_recv_offset_per_proc,
                                        INT_T* d_newq, int64_t* d_pred,
                                        int rank, int size, int lgsize);

__global__ void k_dequeue_step_9_recv_2(INT_T* d_newq, INT_T* d_oldq, int64_t* d_pred, INT_T g_nverts,
		                                INT32_T* d_bitmask_pverts, INT32_T* d_mask, ATOMIC_T* global_count,
					                    int rank, int size, int lgsize);

__global__ void k_unique_local(INT32_T* d_sendbuff, INT_T nlocal_verts, INT_T* d_newq, int64_t* d_pred, 
			                   int rank, int size, int lgsize);

__global__ void k_atomic_enqueue_local(INT_T* d_newq, INT_T* d_oldq, int64_t* d_pred, INT_T g_nverts,
                                       ATOMIC_T* global_count, int rank,
                                       int size, int lgsize);

__global__ void k_bitmask_edges(INT_T* d_edges, INT_T nedges, INT_T* d_unique_edges, INT_T *proc_offset, INT32_T* d_bitmask_pedges,
                                int rank, int size, int lgsize);

__global__ void k_bitmask_verts(INT_T* d_unique_edges, INT_T offset, INT_T nelems, INT32_T* d_bitmask_pverts,
		                        int rank, int size, int lgsize);

__global__ void k_remove_pred(int64_t* d_pred, INT_T* d_mask_1, INT_T* d_mask_2, INT_T nelems,
		                        int rank, int size, int lgsize);

__global__ void k_bfs_copy32(INT_T* d_array, INT_T nelems,  INT32_T* d_buffer32);

__global__ void k_bfs_pred_local(INT32_T *d_buffer32, INT_T nelems, INT32_T *d_pverts, INT32_T* d_mask, int64_t* d_pred,
		                         int rank, int size, int lgsize);

__global__ void k_bfs_pred_recv(INT32_T *d_buffer32, INT_T nelems, INT_T* d_recv_offset_per_proc,
		                        INT_T *d_unique_edges, INT_T *proc_offset, INT32_T *d_pedges, INT32_T* d_mask,
		                        int rank, int size, int lgsize);

__global__ void k_bfs_pred_remote(INT32_T *d_buffer32, INT_T nelems, INT_T *d_mask_1, INT_T* d_mask_2, int64_t* d_pred,
		                         int rank, int size, int lgsize);

__global__ void k_count_proc(INT_T* d_edges_per_proc, INT_T* d_count_edges, int rank, int size, int lgsize);

// ST-CON KERNELS
__global__ void	k_stcon_make_queue_deg(INT_T *d_queue, INT_T queue_count, INT_T *d_queue_off, INT_T *d_queue_deg, INT_T *dg_off, INT_T nverts,
				 int rank, int size, int lgsize);

__global__ void k_stcon_owner(INT_T* d_sendbuff, INT_T M, INT_T* d_mask_1, INT_T* d_mask_2,
			    int rank, int size, int lgsize);

__global__ void k_stcon_binary_mask_unique_large(INT_T next_level_vertices, INT_T *dg_edges, INT_T *d_queue, INT_T *d_next_off,
                                  INT_T *d_queue_off, INT_T queue_count, INT_T *d_send_buff, INT32_T *d_bitmask_pedges, INT32_T *d_mask,
                                  int rank, int size, int lgsize,
                                  INT_T *d_st_rank, INT_T *d_pred, INT_T *d_unique_edges, INT32_T* d_mn_found);

__global__ void k_stcon_back_kernel32(INT_T *d_array, INT_T nelems, INT_T* d_idx, INT32_T *d_support,
		                          int rank, int size, int lgsize);

__global__ void k_stcon_dequeue_step_8_local(INT32_T* d_sendbuff, INT_T nlocal_verts, INT_T* d_newq, int64_t* d_pred, ATOMIC_T* global_count,
                                       int rank, int size, int lgsize);

__global__ void k_stcon_dequeue_step_9_recv_1(INT32_T* d_recvbuff, INT_T recv_count, INT_T* d_recv_offset_per_proc,
					                    INT_T* d_newq, int64_t* d_pred,
					                    int rank, int size, int lgsize,
					                    INT_T *d_st_rank, INT32_T* d_mn_found);

__global__ void k_stcon_dequeue_step_9_recv_2(INT_T* d_newq, INT_T* d_oldq, int64_t* d_pred, INT_T g_nverts,
		                                INT32_T* d_pverts, INT32_T* d_mask, ATOMIC_T* global_count,
					                    int rank, int size, int lgsize, INT_T *d_st_rank);

__global__ void k_stcon_remove_pred(int64_t* d_pred, INT_T* d_mask_1, INT_T* d_mask_2, INT_T nelems,
		                        int rank, int size, int lgsize);

__global__ void k_stcon_pred_recv(INT32_T *d_buffer32, INT_T nelems, INT_T* d_recv_offset_per_proc,
		                        INT_T *d_unique_edges, INT_T *proc_offset, INT32_T *d_pedges, INT32_T* d_mask,
		                        int rank, int size, int lgsize);

#endif

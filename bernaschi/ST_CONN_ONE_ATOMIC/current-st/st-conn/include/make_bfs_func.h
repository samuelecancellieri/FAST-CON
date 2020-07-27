#ifndef _MK_BFS_FUNC_H_
#define _MK_BFS_FUNC_H_
// Require header.h
#ifdef __cplusplus
extern "C" {
#endif

int binary_expand_queue_mask(INT_T next_level_vertices, INT_T *dg_edges, INT_T *d_queue, INT_T *d_next_off,
			                 INT_T *d_queue_off, INT_T queue_count, INT_T *d_send_buff, mask *d_bitmask);

int binary_expand_queue_mask_large(INT_T next_level_vertices, INT_T *dg_edges, INT_T *d_queue, INT_T *d_next_off,
			                       INT_T *d_queue_off, INT_T queue_count, INT_T *d_send_buff, mask *d_bitmask);

int make_queue_offset(INT_T* d_queue_deg, INT_T queue_count, INT_T *d_next_offset, INT_T *next_level_vertices);

int make_queue_deg(INT_T *d_queue, INT_T queue_count, INT_T *d_queue_off, INT_T *d_queue_deg, INT_T *dg_off, INT_T nverts);

int bfs_owners(INT_T* d_sendbuff, INT_T next_level_vertices, INT_T* d_mask_1, INT_T* d_mask_2);

int sort_owners_bfs(INT_T* d_mask_1, INT_T* d_mask_2, INT_T next_level_vertices);

int bfs_count_vertices(INT_T* d_array, INT_T nelems, INT_T* d_count_per_proc,
                       INT_T* send_count_per_proc, INT_T* host_count_per_proc);

int bfs_back_vertices32(INT_T* d_array_u, INT_T nelems, INT_T* d_idx, INT32_T* d_support);

int bfs_count_verts_to_recv(INT_T *recv_count_all_proc, INT_T *recv_count_per_proc);

int atomic_enqueue_local(INT32_T *d_myedges, INT_T mynverts, INT_T *d_new_queue,
			             int64_t *d_pred, INT_T nverts, INT_T *new_queue_count);

int unique_and_atomic_enqueue_local(INT32_T *d_myedges, INT_T mynverts, INT_T *d_new_queue, INT_T *d_queue,
				                    int64_t *d_pred, INT_T nverts, INT_T *new_queue_count);

int atomic_enqueue_recv(INT32_T *d_recv_buff, INT_T recv_count, INT_T* d_recv_offset_per_proc,
			            INT_T *d_q_1, INT_T *d_q_2, INT_T *new_queue_count,
			            int64_t *d_pred, INT_T nverts,  mask *d_bitmask);

int bfs_remove_pred(int64_t *d_pred, INT_T *d_mask_1, INT_T *d_mask_2, INT_T nelems,
		            INT_T *d_out_1, INT_T *d_out_2, INT_T *new_nelems);

int bfs_copy32(INT_T *d_array, INT_T nelems, INT32_T  *d_buffer32);

int bfs_pred_local(INT32_T *d_buffer32, INT_T nelems, mask *d_bitmask, INT_T *d_pred);

int bfs_pred_recv(INT32_T *d_buffer32, INT_T nelems, mask *d_bitmask, INT_T* d_recv_offset_per_proc);

int bfs_pred_remote(INT32_T *d_buffer32, INT_T nelems, INT_T *d_mask_1, INT_T *d_mask_2, int64_t *d_pred);

#ifdef __cplusplus
}
#endif
#endif


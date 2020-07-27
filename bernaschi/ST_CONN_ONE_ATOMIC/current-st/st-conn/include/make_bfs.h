#ifndef _MAKE_BFS_H_
#define _MAKE_BFS_H_
// needs definition of: 
// adjlist in header.h
// csr_graph in reference_common.h
#ifdef __cplusplus
extern "C" {
#endif
int run_make_bfs_mask(adjlist *dg, adjlist *hg, mask *d_bitmask);
int make_bfs_mask(INT_T root, adjlist *hg, INT_T *nvisited, INT32_T *h_send_buff, INT32_T *h_recv_buff,
		   INT_T h_send_size, INT_T h_recv_size, adjlist *dg, int64_t *d_pred, INT_T *d_queue, INT_T * d_queue_off,
		   INT_T *d_queue_deg, INT_T *d_next_off, INT_T *d_send_buff, INT_T d_send_size, INT_T d_recv_size,
		   INT_T *d_mask_1, INT_T *d_mask_2, INT_T d_mask_size,
		   INT32_T* d_buffer32, INT32_T* d_recv_buffer32, mask *d_bitmask);
#ifdef __cplusplus
}
#endif
#endif

#ifndef _MAKE_STCON_H_
#define _MAKE_STCON_H_
// needs definition of: 
// adjlist in header.h
// csr_graph in reference_common.h
#ifdef __cplusplus
extern "C" {
#endif
int run_make_stcon(adjlist *dg, adjlist *hg, mask *d_bitmask);

int make_stcon(INT_T root_s, INT_T root_t, adjlist *hg, INT_T *nvisited, INT32_T *h_send_buff, INT32_T *h_recv_buff,
		   INT_T h_send_size, INT_T h_recv_size, adjlist *dg, int64_t *d_pred, INT_T *d_queue, INT_T * d_queue_off,
		   INT_T *d_queue_deg, INT_T *d_next_off, INT_T *d_send_buff, INT_T d_send_size, INT_T d_recv_size,
		   INT_T *d_mask_1, INT_T *d_mask_2, INT_T d_mask_size,
		   INT32_T* d_buffer32, INT32_T* d_recv_buffer32, mask *d_bitmask, INT_T *h_pred, INT_T *h_st_rank, int *global_mn_found, int *local_mn_found);

void print_stcon_path (int max_level, INT_T root_s, INT_T root_t, INT_T mn, INT_T red_pred, INT_T blue_pred, INT_T *h_pred);

void print_bfs_path (INT_T root_s, INT_T root_t, INT_T mn, INT_T *h_pred);




int validate_stcon_bfs (int max_level, INT_T mn, INT_T root_s, INT_T root_t, INT_T *h_all_bfs_pred, INT32_T *all_verts_offset);
#ifdef __cplusplus
}
#endif
#endif

#ifndef _GPU_TEST_FUNC_
#define _GPU_TEST_FUNC_
// header.h, reference_commmon.h
#ifdef __cplusplus
extern "C" {
#endif
void check_queue_off(INT_T *d_queue_deg, INT_T *d_next_off, INT_T queue_count);
void check_binary_expand(INT_T *d_send_buff, INT_T *d_recv_buff, INT_T next_level_vertices);
void check_queue_deg(INT_T *d_queue, INT_T *d_queue_deg, INT_T queue_count);
void check_unique(INT_T *d_array, INT_T nelems, const char *fcaller, FILE *fout);
void check_owners(INT_T *d_owners, INT_T *d_idx, INT_T nelems, INT_T *d_vertices, const char *fcaller, FILE *fout);
void check_back(INT_T *d_array, INT_T nelems, const char *fcaller, FILE *fout);
void bfs_check_new_queue(INT_T *d_new_queue, INT_T new_queue_count, int64_t *d_pred, INT_T nverts, const char *fcaller, FILE *fout);
void bfs_check_new_queue_local(INT_T *d_new_queue, INT_T new_queue_count, int64_t *d_pred, INT_T nverts, const char *fcaller, FILE *fout);
/*
// BFS CHECK FUNC
void check_csr_struct(csr_graph *cg);
void check_reduce_verts(INT_T my_nverts, INT_T nglobalverts);
void check_back_bfs(INT_T nelems, INT_T *h_array);
*/
void check_add_vu(INT_T *d_edges, INT_T nedges, const char *fcaller, FILE *fout);
void check_add_undirect_edges(INT_T *d_edges, INT_T nedges, const char *fcaller, FILE *fout);
void check_compact_edge_list(INT_T* d_edges, INT_T nelems, INT_T compact_nelems, const char* fcaller, FILE* fout);
void check_split_edge_list(INT_T* d_edges, INT_T nedges, INT_T nedges_to_split, const char* fcaller, FILE* fout);
void check_sort(INT_T *d_array, INT_T nelems,  const char *fcaller, FILE *fout);
void check_offset(INT_T *d_array, INT_T nelems,  const char *fcaller, FILE *fout);
void check_degree(INT_T *d_array, INT_T nelems,  const char *fcaller, FILE *fout);
void check_count(INT_T *d_array, INT_T nelems, INT_T *send_count_per_proc, const char *fcaller, FILE *fout);
//void check_back(INT_T *d_array, INT_T nelems, const char *fcaller, FILE *fout);
void check_unsplit(INT_T *d_array, INT_T nelems, const char *fcaller, FILE *fout);
void check_merge(INT32_T *d_array, INT_T nelems, const char *fcaller, FILE *fout);
#ifdef __cplusplus
}
#endif
#endif

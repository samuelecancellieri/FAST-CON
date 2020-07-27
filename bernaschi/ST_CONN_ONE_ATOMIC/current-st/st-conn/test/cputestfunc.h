#ifndef _CPU_TEST_FUNC_
#define _CPU_TEST_FUNC_
// header.h, reference_commmon.h
#ifdef __cplusplus
extern "C" {
#endif
// STRUCT CHECK FUNC
void check_undirect_struct(INT_T nedges, INT_T *test_edges);
void check_split_struct(INT_T *edges, INT_T nedges, INT_T *test_edges);
void check_owners_struct(INT_T nedges, INT_T *test_edges);
void check_count_struct(INT_T *test_edges, INT_T nedges, INT_T *send_count_per_proc,
		 const char *fcaller, FILE *fout);
void check_back_struct(INT_T nedges, INT_T *test_edges, INT_T *send_count_per_proc, 
		INT_T *send_offset_per_proc);
void check_unsplit_struct(INT_T *edges, INT_T nedges, INT_T *test_edges, 
		   INT_T *send_count_per_proc, INT_T *send_offset_per_proc);
void check_offset_struct(INT_T nverts, INT_T *test_edges);
// BFS CHECK FUNC
void check_csr_struct(csr_graph *cg);
void check_reduce_verts(INT_T my_nverts, INT_T nglobalverts);
void check_back_bfs(INT_T nelems, INT_T *h_array);
// IN BOTH
#ifdef __cplusplus
}
#endif
#endif

#ifndef _MK_STRUCT_H_
#define _MK_STRUCT_H_

#ifdef __cplusplus
extern "C" {
#endif
int run_make_struct(INT_T *h_edges, INT_T h_nedges, adjlist *hg);
//int make_g_undirect(INT_T *d_edges_new, INT_T nedges, INT_T *undirect_nedges);
int make_g_undirect(INT_T *d_edges, INT_T *d_edges_appo, INT_T nedges, INT_T *undirect_nedges); //REMOVECOPY
int make_struct(INT_T nedges, INT_T *d_edges, INT_T *d_edges_appo, INT_T *d_count_per_proc, 
		INT_T *sendbuff, INT_T *recvbuff, INT_T send_size, adjlist *hg);
int make_send_buffer(INT_T *d_edges, INT_T *d_edges_appo, INT_T nedges, INT_T *nedges_to_send, 
		     INT_T *send_count_per_proc, INT_T *send_offset_per_proc, INT_T *d_count_per_proc);
int make_own_csr(INT_T *d_own_edges, INT_T my_own_verts, adjlist *dg);
int make_own_csr_nomulti(INT_T *d_own_edges, INT_T my_own_verts, adjlist *dg);
int make_bitmask (adjlist *h_graph, mask *h_bitmask);
#ifdef __cplusplus
}
#endif

#endif


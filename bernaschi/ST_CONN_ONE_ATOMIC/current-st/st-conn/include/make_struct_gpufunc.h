#ifndef _MK_STRUCT_FUNC_H_
#define _MK_STRUCT_FUNC_H_

#ifdef __cplusplus
extern "C" {
#endif
int add_edeges_vu(INT_T *d_edges, INT_T nedges);

//int compact_edges(INT_T *d_edges_new, INT_T new_nedges, INT_T *compact_nedges);
//int compact_elems(INT_T *d_edges, INT_T nelems, INT_T *compact_nelems, int value);

int split_edges(INT_T* d_edges, INT_T nedges, INT_T* nedges_to_split);
int unsplit_edges(INT_T *d_edges_u, INT_T *d_edegs_v, INT_T nedges, INT_T *d_edges_appo);
int owners_edges(INT_T* d_edges_u, INT_T h_nedges, INT_T* d_edges_appo);
int sort_edges_u(INT_T* d_edges_u, INT_T* d_edges_v, INT_T h_nedges, INT_T *umax);
int sort_unique_edges(INT_T* d_edges_v, INT_T* d_edges_u, INT_T *compact_nedges, unsigned int *stencil, INT_T *umax, INT_T *vmax);
int count_vertices(INT_T* d_edges, INT_T h_nedges, INT_T* d_count_per_proc, INT_T* send_count_per_proc, INT_T* host_count_per_proc);
int compact_uv(INT_T* d_edges_appo, INT_T* d_edges_u, INT_T* d_edges_v, INT_T* d_edges, INT_T h_nedges);
int back_vertices(INT_T* d_edges_u, INT_T* d_edges_v, INT_T nedges, INT_T* d_edges_appo_u, INT_T* d_edges_appo_v);
int make_offset(INT_T *d_edges_u, INT_T nedges, INT_T *d_count_u, INT_T nverts, INT_T *d_degree);
int copy_bitmask_on_device(mask *h_bitmask, mask *d_bitmask);
int build_bitmask_on_device(mask *h_bitmask, mask *d_bitmask, adjlist *d_graph);
#ifdef __cplusplus
}
#endif
#endif


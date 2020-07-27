#ifndef _ADJ_FUNC_H_
#define _ADJ_FUNC_H_
// needs header.h for def of adjlist

#ifdef __cplusplus
extern "C" {
#endif
int copyAdjDeviceToHost(adjlist *hg, adjlist *dg);
int copyAdjHostToDevice(adjlist *dg, adjlist *hg);
int init_adjlist(adjlist *adj);
int free_adjlist(adjlist *adj);
int print_adjlist(adjlist *adj, FILE *fout, const char *func_name);
int check_adjlist(adjlist *adj, FILE *fout, const char *func_name);
void print_adjlist_stat(adjlist *adj, const char *fcaller, FILE *fout);
int remove_extra_edge(adjlist *hostadj, INT_T U, INT_T V); 
int print_csr_graph(csr_graph* g, FILE *fp, const char *func_name);
int convert_to_csr(adjlist* g, csr_graph* cg, int64_t nglobalverts, FILE *fout);
void free_csr_graph(csr_graph* const g);
int count_visited_edges(csr_graph *cg, double *edges_counts, int64_t *edge_visit_count,
			int64_t bfs_root_idx, int64_t *pred);
int convert_csr_to_adj(csr_graph *cg, adjlist *hg, FILE *fout);

int init_bitmask(mask *bitmask);

#ifdef __cplusplus
}
#endif
#endif

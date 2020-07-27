#ifndef _REFERENCE_COMMON_H_
#define _REFERENCE_COMMON_H_

#define INT64_T_MPI_TYPE MPI_LONG_LONG
#include "header.h"
typedef struct csr_graph {
  size_t nlocalverts;
  size_t nlocaledges;
  int64_t nglobalverts;
  size_t *rowstarts;
  int64_t *column;
} csr_graph;


#ifdef __cplusplus
extern "C" {
#endif
enum {s_minimum, s_firstquartile, s_median, s_thirdquartile, 
			s_maximum, s_mean, s_std, s_LAST};

int validate_bfs_result(const csr_graph* const g, const int64_t root, 
						const int64_t* const pred, const int64_t nvisited);

void find_bfs_roots(int* num_bfs_roots, const csr_graph* const g, const uint64_t seed1, const uint64_t seed2, int64_t* const bfs_roots);

void find_stcon_roots(int* num_bfs_roots, const csr_graph* const g, adjlist* hg, const uint64_t seed1, const uint64_t seed2, int64_t* const bfs_roots);

void get_statistics(const double x[], int n, double r[s_LAST]);

void convert_graph_to_csr(const int64_t nedges, const int64_t* const edges, csr_graph* const g);

#ifdef __cplusplus
}
#endif
#endif

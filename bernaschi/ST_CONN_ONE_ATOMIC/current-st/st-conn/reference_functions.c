/* This is to make PRId64 working with c++ compiler */
#ifdef __cplusplus
#define __STDC_FORMAT_MACROS
#endif
/* header of int64_t and PRId64 */
#include <inttypes.h>

#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "defines.h"
#include "reference_common.h"
#include "../generator/make_graph.h"

/*
extern int rank, size;
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
extern int lgsize;
#endif
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
#define MOD_SIZE(v) ((v) & size_minus_one)
#define DIV_SIZE(v) ((v) >> lgsize)
#else
#define MOD_SIZE(v) ((v) % size)
#define DIV_SIZE(v) ((v) / size)
#endif
#define VERTEX_OWNER(v) ((int)(MOD_SIZE(v)))
#define VERTEX_LOCAL(v) ((size_t)(DIV_SIZE(v)))
#define VERTEX_TO_GLOBAL(i) ((int64_t)((i) * size + rank))
*/

int print_csr_graph_2(const csr_graph* const g, FILE *fp, const char *func_name)
{
  if ((g == NULL) || (g->column == NULL) || (g->rowstarts == NULL)) {
    fprintf(stderr, "%s: rank %d, in print_csr_graph: NULL input\n",
            func_name, rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if ((g->nlocalverts <= 0) || (g->nlocaledges <= 0)) {
    fprintf(stderr, "%s; rank %d, in print_csr_graph:"
            " nverts or nedges <=0\n", func_name, rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  fprintf(fp,"\n*** *** ** %s: CSR_GRAPH ** *** ***\n", func_name);
  fprintf(fp, "Processor %d says nlocaledges=%zu\n",
                  rank, g->nlocaledges);
  fprintf(fp, "Processor %d says nlocalverts=%zu\n",
                  rank, g->nlocalverts);
  fprintf(fp, "Processor %d says nglobaverts=%"PRId64"\n",
                  rank, g->nglobalverts);

  int64_t ii,jj;
  int64_t ei_s, ei_e;
  for (ii = 0; ii < (int64_t)g->nlocalverts; ++ii) {
    ei_s = g->rowstarts[ii];
    ei_e = g->rowstarts[ii+1];

    fprintf(fp, "%"PRId64"-> ", ((int64_t)((ii) * size + rank)));
    for (jj = ei_s; jj < ei_e; ++jj) {
      fprintf(fp, "%"PRId64", ", g->column[jj]);
    }
    fprintf(fp, "\n");
  }
  fprintf(fp,"\n*** *** ** %s: CSR_GRAPH ** *** ***\n", func_name);
  return 0;
}


void* refmalloc(size_t nbytes, const char *fcaller) {
  void* p = malloc(nbytes);
  if (!p) {
    fprintf(stderr, "rank %d in %s malloc() failed for size %zu\n", rank, fcaller, nbytes);
    abort();
  }
  return p;
}

void* refcalloc(size_t n, size_t unit) {
  void* p = calloc(n, unit);
  if (!p) {
    fprintf(stderr, "calloc() failed for size %zu * %zu\n", n, unit);
    abort();
  }
  return p;
}

static int compare_doubles(const void* a, const void* b) {
  double aa = *(const double*)a;
  double bb = *(const double*)b;
  return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}


int validate_bfs_result(const csr_graph* const g, const int64_t root, const int64_t* const pred, const int64_t nvisited) {
  int validation_passed = 1;
  int root_is_mine = (VERTEX_OWNER(root) == rank);

  const size_t nlocalverts = g->nlocalverts;
  const size_t nlocaledges = g->nlocaledges;
  const int64_t nglobalverts = g->nglobalverts;

	//print_csr_graph_2(g, stderr, "VALIDATE BFS");
	

  /* Check that root is its own parent. */
  if (root_is_mine) {
    if (pred[VERTEX_LOCAL(root)] != root) {
      fprintf(stderr, "%d: Validation error: parent of root vertex %" PRId64 " is %" PRId64 ", not the root itself.\n", rank, root, pred[VERTEX_LOCAL(root)]);
      validation_passed = 0;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  /* Check that nothing else is its own parent, and check for in-range
   * values. */
  int any_range_errors = 0;
  size_t i;
  for (i = 0; i < nlocalverts; ++i) {
    int64_t v = VERTEX_TO_GLOBAL(i);
    assert (VERTEX_OWNER(v) == rank);
    assert (VERTEX_LOCAL(v) == i);
    if (v != root && pred[i] == v) {
      fprintf(stderr, "%d: Validation error: parent of non-root vertex %" PRId64 " is itself.\n", rank, v);
      validation_passed = 0;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (pred[i] < -1 || pred[i] >= nglobalverts) {
      fprintf(stderr, "%d: Validation error: parent of vertex %" PRId64 " is out-of-range value %" PRId64 ".\n", rank, v, pred[i]);
      validation_passed = 0;
      any_range_errors = 1;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &any_range_errors, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

  /* Check that nvisited is correct. */
  int64_t nvisited_actual = 0;
  for (i = 0; i < nlocalverts; ++i) {
    if (pred[i] != -1) ++nvisited_actual;
  }
  MPI_Allreduce(MPI_IN_PLACE, &nvisited_actual, 1, INT64_T_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);
  if (nvisited_actual != nvisited) {
    fprintf(stderr, "%d: Validation error: claimed visit count %" PRId64 " is different from actual count %" PRId64 ".\n", rank, nvisited, nvisited_actual);
    validation_passed = 0;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (!any_range_errors) { /* Other parts of validation assume in-range values */

    /* Check that there is an edge from each vertex to its claimed
     * predecessor. */
    size_t i;
    for (i = 0; i < nlocalverts; ++i) {
      int64_t v = VERTEX_TO_GLOBAL(i);
      int64_t p = pred[i];
      if (p == -1) continue;
      int found_pred_edge = 0;
      if (v == p) found_pred_edge = 1; /* Root vertex */
      size_t ei, ei_end = g->rowstarts[i + 1];
      for (ei = g->rowstarts[i]; ei < ei_end; ++ei) {
        int64_t w = g->column[ei];
        if (w == p) {
          found_pred_edge = 1;
          break;
        }
      }
      if (!found_pred_edge) {
        fprintf(stderr, "%d: Validation error: no graph edge from vertex %" PRId64 " to its parent %" PRId64 ".\n", rank, v, p);
        validation_passed = 0;
	MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }

    /* Create a vertex depth map to use for later validation. */
    int64_t* depth = (int64_t*)refmalloc(nlocalverts * sizeof(int64_t), __func__);
    { /* Scope some code that has a lot of temporary variables. */
      int64_t* pred_depth = (int64_t*)refmalloc(nlocalverts * sizeof(int64_t), __func__); /* Depth of predecessor vertex for each local vertex */
      size_t i;
      for (i = 0; i < nlocalverts; ++i) depth[i] = INT64_MAX;
      if (root_is_mine) depth[VERTEX_LOCAL(root)] = 0;
      /* Send each vertex that appears in the local part of the predecessor map
       * to its owner; record the original locations so we can put the answers
       * into pred_depth. */
      /* Do a histogram sort by owner (this same kind of sort is used other
       * places as well).  First, count the number of vertices going to each
       * destination. */
      int* num_preds_per_owner = (int*)refcalloc(size, sizeof(int)); /* Uses zero-init */
      for (i = 0; i < nlocalverts; ++i) {
        ++num_preds_per_owner[pred[i] == -1 ? size - 1 : VERTEX_OWNER(pred[i])];
      }
      int64_t* preds_per_owner = (int64_t*)refmalloc(nlocalverts * sizeof(int64_t), __func__); /* Predecessors sorted by owner */
      int64_t* preds_per_owner_results_offsets = (int64_t*)refmalloc(nlocalverts * sizeof(int64_t), __func__); /* Indices into pred_depth to write */
      /* Second, do a prefix sum to get the displacements of the different
       * owners in the outgoing array. */
      int* pred_owner_displs = (int*)refmalloc((size + 1) * sizeof(int), __func__);
      pred_owner_displs[0] = 0;
      int r;
      for (r = 0; r < size; ++r) {
        pred_owner_displs[r + 1] = pred_owner_displs[r] + num_preds_per_owner[r];
      }
      /* Last, put the vertices into the correct positions in the array, based
       * on their owners and the counts and displacements computed earlier. */
      int* pred_owner_offsets = (int*)refmalloc((size + 1) * sizeof(int), __func__);
      memcpy(pred_owner_offsets, pred_owner_displs, (size + 1) * sizeof(int));
      for (i = 0; i < nlocalverts; ++i) {
        int* offset_ptr = &pred_owner_offsets[pred[i] == -1 ? size - 1 : VERTEX_OWNER(pred[i])];
        preds_per_owner[*offset_ptr] = pred[i];
        preds_per_owner_results_offsets[*offset_ptr] = i;
        ++*offset_ptr;
      }
      for (r = 0; r < size; ++r) {
        assert (pred_owner_offsets[r] == pred_owner_displs[r + 1]);
      }
      free(pred_owner_offsets);

      /* Send around the number of vertices that will be sent to each destination. */
      int* num_my_preds_per_sender = (int*)refmalloc(size * sizeof(int), __func__);
      MPI_Alltoall(num_preds_per_owner, 1, MPI_INT,
                   num_my_preds_per_sender, 1, MPI_INT,
                   MPI_COMM_WORLD);
      int* my_preds_per_sender_displs = (int*)refmalloc((size + 1) * sizeof(int), __func__);
      my_preds_per_sender_displs[0] = 0;
      for (r = 0; r < size; ++r) {
        my_preds_per_sender_displs[r + 1] = my_preds_per_sender_displs[r] + num_my_preds_per_sender[r];
      }
      /* Send around the actual vertex data (list of depth requests that will
       * be responded to at each BFS iteration). */
      int64_t* my_depth_requests = (int64_t*)refmalloc(my_preds_per_sender_displs[size] * sizeof(int64_t), __func__);
      int64_t* my_depth_replies = (int64_t*)refmalloc(my_preds_per_sender_displs[size] * sizeof(int64_t), __func__);
      MPI_Alltoallv(preds_per_owner, num_preds_per_owner, pred_owner_displs, INT64_T_MPI_TYPE,
                    my_depth_requests, num_my_preds_per_sender, my_preds_per_sender_displs, INT64_T_MPI_TYPE,
                    MPI_COMM_WORLD);

      int64_t* pred_depth_raw = (int64_t*)refmalloc(nlocalverts * sizeof(int64_t), __func__); /* Depth of predecessor vertex for each local vertex, ordered by source proc */

      /* Do a mini-BFS (naively) over just the predecessor graph (hopefully a
       * tree) produced by the real BFS; fill in the depth map. */
      while (1) {
        int any_changed = 0;
        int i;
        /* Create and send the depth values requested by other nodes.  The list
         * of requests is sent once, and are stored on the receiver so the
         * replies can be sent (possibly with updated depth values) at every
         * iteration. */
        for (i = 0; i < my_preds_per_sender_displs[size]; ++i) {
          my_depth_replies[i] = (my_depth_requests[i] == -1 ? INT64_MAX : depth[VERTEX_LOCAL(my_depth_requests[i])]);
        }
        MPI_Alltoallv(my_depth_replies, num_my_preds_per_sender, my_preds_per_sender_displs, INT64_T_MPI_TYPE,
                      pred_depth_raw, num_preds_per_owner, pred_owner_displs, INT64_T_MPI_TYPE,
                      MPI_COMM_WORLD);
        {
          size_t i;
          /* Put the received depths into the local array. */
          for (i = 0; i < nlocalverts; ++i) {
            pred_depth[preds_per_owner_results_offsets[i]] = pred_depth_raw[i];
          }
          /* Check those values to determine if they violate any correctness
           * conditions. */
          for (i = 0; i < nlocalverts; ++i) {
            int64_t v = VERTEX_TO_GLOBAL(i);
            if (v == root) {
              /* The depth and predecessor for this were checked earlier. */
            } else if (depth[i] == INT64_MAX && pred_depth[i] == INT64_MAX) {
              /* OK -- depth should be filled in later. */
            } else if (depth[i] == INT64_MAX && pred_depth[i] != INT64_MAX) {
              depth[i] = pred_depth[i] + 1;
              any_changed = 1;
            } else if (depth[i] != pred_depth[i] + 1) {
              fprintf(stderr, "%d: Validation error: BFS predecessors do not form a tree; see vertices %" PRId64 " (depth %" PRId64 ") and %" PRId64 " (depth %" PRId64 ").\n", rank, v, depth[i], pred[i], pred_depth[i]);
              validation_passed = 0;
              MPI_Abort(MPI_COMM_WORLD, 1);
            } else {
              /* Vertex already has its correct depth value. */
            }
          }
        }
        MPI_Allreduce(MPI_IN_PLACE, &any_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (!any_changed) break;
      }

      free(num_preds_per_owner);
      free(num_my_preds_per_sender);
      free(preds_per_owner);
      free(preds_per_owner_results_offsets);
      free(my_preds_per_sender_displs);
      free(my_depth_requests);
      free(my_depth_replies);
      free(pred_owner_displs);
      free(pred_depth);
      free(pred_depth_raw);
    }

    /* Check that all edges connect vertices whose depths differ by at most
     * one. */
    {
      int64_t maxlocaledges = 0;
      MPI_Allreduce((void*)&nlocaledges, &maxlocaledges, 1, INT64_T_MPI_TYPE, MPI_MAX, MPI_COMM_WORLD);
      /* We break the total list of overall edges into chunks to reduce the
       * amount of data to be sent at a time (since we are using MPI_Alltoallv
       * to send data collectively). */
      const int edge_chunk_size = (1 << 23); /* Reduce memory usage */
      int num_edge_groups = (maxlocaledges + edge_chunk_size - 1) / edge_chunk_size;
      int eg;
      for (eg = 0; eg < num_edge_groups; ++eg) {
        size_t first_edge_index = (size_t)(eg * edge_chunk_size);
        if (first_edge_index > nlocaledges) first_edge_index = nlocaledges;
        size_t last_edge_index = (size_t)((eg + 1) * edge_chunk_size);
        if (last_edge_index > nlocaledges) last_edge_index = nlocaledges;
        /* Sort the edge targets in this chunk by their owners (histogram
         * sort); see the BFS code above for details of the steps of the
         * algorithm. */
        int* num_edge_targets_by_owner = (int*)refcalloc(size, sizeof(int)); /* Uses zero-init */
        size_t ei;
        for (ei = first_edge_index; ei < last_edge_index; ++ei) {
          ++num_edge_targets_by_owner[VERTEX_OWNER(g->column[ei])];
        }
        int* edge_targets_by_owner_displs = (int*)refmalloc((size + 1) * sizeof(int), __func__);
        edge_targets_by_owner_displs[0] = 0;
        int i;
        for (i = 0; i < size; ++i) {
          edge_targets_by_owner_displs[i + 1] = edge_targets_by_owner_displs[i] + num_edge_targets_by_owner[i];
        }
        int64_t* edge_targets_by_owner = (int64_t*)refmalloc(edge_targets_by_owner_displs[size] * sizeof(int64_t), __func__);
        int64_t* edge_targets_by_owner_indices = (int64_t*)refmalloc(edge_targets_by_owner_displs[size] * sizeof(int64_t), __func__); /* Source indices for where to write the targets */
        int* edge_targets_by_owner_offsets = (int*)refmalloc((size + 1) * sizeof(int), __func__);
        memcpy(edge_targets_by_owner_offsets, edge_targets_by_owner_displs, (size + 1) * sizeof(int));
        for (ei = first_edge_index; ei < last_edge_index; ++ei) {
          edge_targets_by_owner[edge_targets_by_owner_offsets[VERTEX_OWNER(g->column[ei])]] = g->column[ei];
          edge_targets_by_owner_indices[edge_targets_by_owner_offsets[VERTEX_OWNER(g->column[ei])]] = ei;
          ++edge_targets_by_owner_offsets[VERTEX_OWNER(g->column[ei])];
        }
        for (i = 0; i < size; ++i) {
          assert (edge_targets_by_owner_offsets[i] == edge_targets_by_owner_displs[i + 1]);
        }
        free(edge_targets_by_owner_offsets);

        /* Send around the number of data elements that will be sent later. */
        int* num_incoming_targets_by_src = (int*)refmalloc(size * sizeof(int), __func__);
        MPI_Alltoall(num_edge_targets_by_owner, 1, MPI_INT,
                     num_incoming_targets_by_src, 1, MPI_INT,
                     MPI_COMM_WORLD);
        int* incoming_targets_by_src_displs = (int*)refmalloc((size + 1) * sizeof(int), __func__);
        incoming_targets_by_src_displs[0] = 0;
        for (i = 0; i < size; ++i) {
          incoming_targets_by_src_displs[i + 1] = incoming_targets_by_src_displs[i] + num_incoming_targets_by_src[i];
        }

        int64_t* target_depth_requests = (int64_t*)refmalloc(incoming_targets_by_src_displs[size] * sizeof(int64_t), __func__);
        int64_t* target_depth_replies = (int64_t*)refmalloc(incoming_targets_by_src_displs[size] * sizeof(int64_t), __func__);

        /* Send the actual requests for the depths of edge targets. */
        MPI_Alltoallv(edge_targets_by_owner, num_edge_targets_by_owner, edge_targets_by_owner_displs, INT64_T_MPI_TYPE,
                      target_depth_requests, num_incoming_targets_by_src, incoming_targets_by_src_displs, INT64_T_MPI_TYPE,
                      MPI_COMM_WORLD);

        free(edge_targets_by_owner);

        /* Fill in the replies for the requests sent to me. */
        for (i = 0; i < incoming_targets_by_src_displs[size]; ++i) {
          assert (VERTEX_OWNER(target_depth_requests[i]) == rank);
          target_depth_replies[i] = depth[VERTEX_LOCAL(target_depth_requests[i])];
        }

        free(target_depth_requests);

        int64_t* target_depth_raw = (int64_t*)refmalloc((last_edge_index - first_edge_index) * sizeof(int64_t), __func__);

        /* Send back the replies. */
        MPI_Alltoallv(target_depth_replies, num_incoming_targets_by_src, incoming_targets_by_src_displs, INT64_T_MPI_TYPE,
                      target_depth_raw, num_edge_targets_by_owner, edge_targets_by_owner_displs, INT64_T_MPI_TYPE,
                      MPI_COMM_WORLD);

        free(target_depth_replies);
        free(num_incoming_targets_by_src);
        free(num_edge_targets_by_owner);
        free(incoming_targets_by_src_displs);
        free(edge_targets_by_owner_displs);

        int64_t* target_depth = (int64_t*)refmalloc((last_edge_index - first_edge_index) * sizeof(int64_t), __func__);

        /* Put the replies into the proper order (original order of the edges).
         * */
        for (ei = 0; ei < last_edge_index - first_edge_index; ++ei) {
          target_depth[edge_targets_by_owner_indices[ei] - first_edge_index] = target_depth_raw[ei];
        }

        free(target_depth_raw);
        free(edge_targets_by_owner_indices);

        /* Check the depth relationship of the endpoints of each edge in the
         * current chunk. */
        size_t src_i = 0;
        for (ei = first_edge_index; ei < last_edge_index; ++ei) {
          while (ei >= g->rowstarts[src_i + 1]) {
            ++src_i;
          }
          int64_t src = VERTEX_TO_GLOBAL(src_i);
          int64_t src_depth = depth[src_i];
          int64_t tgt = g->column[ei];
          int64_t tgt_depth = target_depth[ei - first_edge_index];
          if (src_depth != INT64_MAX && tgt_depth == INT64_MAX) {
            fprintf(stderr, "%d: Validation error: edge connects vertex %" PRId64 " in the BFS tree (depth %" PRId64 ") to vertex %" PRId64 " outside the tree.\n", rank, src, src_depth, tgt);
            validation_passed = 0;
            MPI_Abort(MPI_COMM_WORLD, 1);
          } else if (src_depth == INT64_MAX && tgt_depth != INT64_MAX) {
            /* Skip this for now; this problem will be caught when scanning
             * reversed copy of this edge.  Set the failure flag, though,
             * just in case. */
            validation_passed = 0;
	    //MPI_Abort(MPI_COMM_WORLD, 1);
          } else if (src_depth - tgt_depth < -1 ||
                     src_depth - tgt_depth > 1) {
            fprintf(stderr, "%d: Validation error: depths of edge endpoints %" PRId64 " (depth %" PRId64 ") and %" PRId64 " (depth %" PRId64 ") are too far apart (abs. val. > 1).\n", rank, src, src_depth, tgt, tgt_depth);
            validation_passed = 0;
	    MPI_Abort(MPI_COMM_WORLD, 1);
          }
        }
        free(target_depth);
      }
    }

    free(depth);

  } /* End of part skipped by range errors */
  
  /* Collect the global validation result. */
  MPI_Allreduce(MPI_IN_PLACE, &validation_passed, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  return validation_passed;
}

/* Find num_bfs_roots random vertices, each of which has degree >= 1, with the
 * same answer produced on all nodes. */
void find_bfs_roots(int* num_bfs_roots, const csr_graph* const g, const uint64_t seed1, const uint64_t seed2, int64_t* const bfs_roots) {
  
	/* This implementation is slow, but there aren't enough roots being
   * generated for that to be a big issue. */
  uint64_t counter = 0;
  int bfs_root_idx;
  for (bfs_root_idx = 0; bfs_root_idx < *num_bfs_roots; ++bfs_root_idx) {
    int64_t root;
    while (1) {
      double d[2];
      make_random_numbers(2, seed1, seed2, counter, d);
      root = (int64_t)((d[0] + d[1]) * g->nglobalverts) % g->nglobalverts;
      counter += 2;
      if (counter > 2*g->nglobalverts) break;
      int is_duplicate = 0;
      int i;
      for (i = 0; i < bfs_root_idx; ++i) {
        if (root == bfs_roots[i]) {
          is_duplicate = 1;
          break;
        }
      }
      if (is_duplicate) continue; /* Everyone takes the same path here */
      int root_ok;
      if (VERTEX_OWNER(root) == rank) {
        root_ok = 0;
        size_t ei, ei_end = g->rowstarts[VERTEX_LOCAL(root) + 1];
		for (ei = g->rowstarts[VERTEX_LOCAL(root)]; ei < ei_end; ++ei) {
		  if (g->column[ei] != root) {
			root_ok = 1;
			break;
		  }
		}
      }
      MPI_Bcast(&root_ok, 1, MPI_INT, VERTEX_OWNER(root), MPI_COMM_WORLD);
      if (root_ok) break;
    }
    bfs_roots[bfs_root_idx] = root;
  }
  *num_bfs_roots = bfs_root_idx;
}


/* Find num_stcon_roots random vertices, each of which has degree >= 1 and are different
 * same answer produced on all nodes. */
void find_stcon_roots(int* num_bfs_roots, const csr_graph* const g, adjlist *hg, const uint64_t seed1, const uint64_t seed2, int64_t* const bfs_roots)
{

	/* This implementation is slow, but there aren't enough roots being
   * generated for that to be a big issue. */
  uint64_t counter = 0;
  int bfs_root_idx;
  int root_ok;
  for (bfs_root_idx = 0; bfs_root_idx < *num_bfs_roots; ++bfs_root_idx) {
    int64_t root;
    while (1) {
      double d[2];
      make_random_numbers(2, seed1, seed2, counter, d);
      root = (int64_t)((d[0] + d[1]) * g->nglobalverts) % g->nglobalverts;
      counter += 2;
      if (counter > 2*g->nglobalverts) break;
      int is_duplicate = 0;
      int i;
      for (i = 0; i < bfs_root_idx; ++i) {
        if (root == bfs_roots[i]) {
          is_duplicate = 1;
          break;
        }
      }
      if (is_duplicate) continue; /* Everyone takes the same path here */
      root_ok = 0;
      if (VERTEX_OWNER(root) == rank) {
    	 
         size_t ei, ei_end = g->rowstarts[VERTEX_LOCAL(root) + 1];
		 for (ei = g->rowstarts[VERTEX_LOCAL(root)]; ei < ei_end; ++ei) {
		    if (g->column[ei] != root) {
			  if (hg->degree[VERTEX_LOCAL(root)] > 0) {
				 root_ok = 1;
				 break;
			  }
		   }
		 }
      } // IF VERTEX
      MPI_Bcast(&root_ok, 1, MPI_INT, VERTEX_OWNER(root), MPI_COMM_WORLD);
      if (root_ok) break;
    }
    bfs_roots[bfs_root_idx] = root;
  }
  *num_bfs_roots = bfs_root_idx;
}



void get_statistics(const double x[], int n, double r[s_LAST]) 
{
	double temp;
	int i;
	/* Compute mean. */
	temp = 0;
	for (i = 0; i < n; ++i) temp += x[i];
	temp /= n;
	r[s_mean] = temp;
	/* Compute std. dev. */
	temp = 0;
	for (i = 0; i < n; ++i) temp += (x[i] - r[s_mean]) * (x[i] - r[s_mean]);
	temp /= n - 1;
	r[s_std] = sqrt(temp);
	/* Sort x. */
	double* xx = (double*)refmalloc(n * sizeof(double), __func__);
	memcpy(xx, x, n * sizeof(double));
	qsort(xx, n, sizeof(double), compare_doubles);
	/* Get order statistics. */
	r[s_minimum] = xx[0];
	r[s_firstquartile] = (xx[(n - 1) / 4] + xx[n / 4]) * .5;
	r[s_median] = (xx[(n - 1) / 2] + xx[n / 2]) * .5;
	r[s_thirdquartile] = (xx[n - 1 - (n - 1) / 4] + xx[n - 1 - n / 4]) * .5;
	r[s_maximum] = xx[n - 1];
	/* Clean up. */
	free(xx);
}

void* xrealloc(void* p, size_t nbytes) {
  p = realloc(p, nbytes);
  if (!p && nbytes != 0) {
    fprintf(stderr, "realloc() failed for size %zu\n", nbytes);
    abort();
  }
  return p;
}

/* STINGER-like group of edges, forming a linked list of pages. */
typedef struct edge_page {
  int64_t targets[16]; /* Unused elements filled with -1 */
  struct edge_page* next; /* NULL if no more */
} edge_page;

static inline edge_page* new_edge_page(void) {
  edge_page* ep = (edge_page*)refmalloc(sizeof(edge_page), __func__);
  int i;
  ep->next = NULL;
  for (i = 0; i < 16; ++i) ep->targets[i] = -1;
  return ep;
}

static inline void delete_edge_page(edge_page* ep) {
  if (!ep) return;
  delete_edge_page(ep->next);
  free(ep);
}

typedef struct adjacency_list {
  size_t nvertices; /* User-visible number of vertices */
  size_t nvertices_allocated; /* Actual allocated size of data */
  edge_page** data; /* Indexed by vertex */
} adjacency_list;

static void grow_adj_list(adjacency_list* al, size_t min_nvertices) {
  if (min_nvertices <= al->nvertices) return;
  while (min_nvertices > al->nvertices_allocated) {
    al->nvertices_allocated = (al->nvertices_allocated == 0) ? 16 : (al->nvertices_allocated * 2);
    al->data = (edge_page**)xrealloc(al->data, al->nvertices_allocated * sizeof(edge_page*));
  }
  size_t i;
  for (i = al->nvertices; i < min_nvertices; ++i) {
    al->data[i] = NULL;
  }
  al->nvertices = min_nvertices;
}

static void add_adj_list_edge(adjacency_list* al, size_t src, int64_t tgt) {
  grow_adj_list(al, src + 1);
  edge_page** p = al->data + src;
  /* Each page is filled before we allocate another one, so we only need to
   * check the last one in the chain. */
  while (*p && (*p)->next) {p = &((*p)->next);}
  if (*p) {
    assert (!(*p)->next);
    int i;
    for (i = 0; i < 16; ++i) {
      if ((*p)->targets[i] == -1) {
        (*p)->targets[i] = tgt;
        return;
      }
    }
    p = &((*p)->next);
    assert (!*p);
  }
  assert (!*p);
  *p = new_edge_page();
  (*p)->targets[0] = tgt;
}

static void clear_adj_list(adjacency_list* al) {
  size_t i;
  for (i = 0; i < al->nvertices; ++i) delete_edge_page(al->data[i]);
  free(al->data);
  al->data = NULL;
  al->nvertices = al->nvertices_allocated = 0;
}

void convert_graph_to_csr(const int64_t nedges, const int64_t* const edges, csr_graph* const g) {
  adjacency_list adj_list = {0, 0, NULL}; /* Adjacency list being built up with
                                           * received data */
  {
    /* Redistribute each input undirected edge (a, b) to create two directed
     * copies: (a -> b) on VERTEX_OWNER(a) and (b -> a) on VERTEX_OWNER(b)
     * [except for self-loops, of which only one copy is kept]. */
    const size_t edge_buffer_size = (1 << 27) / (2 * sizeof(int64_t)) / size; /* 128 MiB */
    /* Note that these buffers are edge pairs (src, tgt), where both elements
     * are global vertex indexes. */
    int64_t* recvbuf = (int64_t*)refmalloc(edge_buffer_size * 2 * sizeof(int64_t), __func__);
    MPI_Request recvreq;
    int recvreq_active = 0;
    int64_t* coalescing_buf = (int64_t*)refmalloc(size * edge_buffer_size * 2 * sizeof(int64_t), __func__);
    size_t* coalescing_counts = (size_t*)refcalloc(size, sizeof(size_t)); /* Uses zero-init */
    MPI_Request* sendreqs = (MPI_Request*)refmalloc(size * sizeof(MPI_Request), __func__);
    int* sendreqs_active = (int*)refcalloc(size, sizeof(int)); /* Uses zero-init */
    int num_sendreqs_active = 0;
    int num_senders_done = 1; /* Number of ranks that have said that they will
                               * not send to me again; I will never send to
                               * myself at all (see test in SEND). */

    /* The heavy use of macros here is to create the equivalent of nested
     * functions with proper access to variables from the enclosing scope. */

#define PROCESS_EDGE(src, tgt) \
    /* Do the handling for one received edge. */ \
    do { \
      assert (VERTEX_OWNER((src)) == rank); \
      add_adj_list_edge(&adj_list, VERTEX_LOCAL((src)), (tgt)); \
    } while (0)

#define START_IRECV \
    /* Start/restart the receive operation to wait for blocks of edges, if
     * needed. */ \
    do { \
      if (num_senders_done < size) { \
        MPI_Irecv(recvbuf, edge_buffer_size * 2, INT64_T_MPI_TYPE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recvreq); \
        recvreq_active = 1; \
      } \
    } while (0)

#define PROCESS_REQS \
    /* Handle all outstanding MPI requests and progress the MPI implementation.
     * */ \
    do { \
      int flag; \
      /* Test receive request. */ \
      while (recvreq_active) { \
        MPI_Status st; \
        MPI_Test(&recvreq, &flag, &st); \
        if (!flag) break; \
        /* A message arrived. */ \
        recvreq_active = 0; \
        int count; \
        MPI_Get_count(&st, INT64_T_MPI_TYPE, &count); \
        count /= 2; \
        if (count == 0 /* This count is used as a flag when each sender is done */ ) { \
          ++num_senders_done; \
        } else { \
          /* Process the edges in the received message. */ \
          int c; \
          for (c = 0; c < count; ++c) { \
            PROCESS_EDGE(recvbuf[c * 2], recvbuf[c * 2 + 1]); \
          } \
        } \
        START_IRECV; \
      } \
      /* Test send requests to determine when their buffers are available for
       * reuse. */ \
      int c; \
      for (c = 0; c < size; ++c) { \
        if (sendreqs_active[c]) { \
          MPI_Test(&sendreqs[c], &flag, MPI_STATUS_IGNORE); \
          if (flag) {sendreqs_active[c] = 0; --num_sendreqs_active;} \
        } \
      } \
    } while (0)

#define SEND(src, tgt) \
    do { \
      int dest = VERTEX_OWNER((src)); \
      if (dest == rank) { \
        /* Process self-sends locally. */ \
        PROCESS_EDGE((src), (tgt)); \
      } else { \
        while (sendreqs_active[dest]) PROCESS_REQS; /* Wait for send buffer to be available */ \
        /* Push onto coalescing buffer. */ \
        size_t c = coalescing_counts[dest]; \
        coalescing_buf[dest * edge_buffer_size * 2 + c * 2] = (src); \
        coalescing_buf[dest * edge_buffer_size * 2 + c * 2 + 1] = (tgt); \
        ++coalescing_counts[dest]; \
        /* Send if the buffer is full. */ \
        if (coalescing_counts[dest] == edge_buffer_size) { \
          FLUSH_COALESCING_BUFFER(dest); \
        } \
      } \
    } while (0)

#define FLUSH_COALESCING_BUFFER(dest) \
    do { \
      while (sendreqs_active[(dest)]) PROCESS_REQS; /* Wait for previous sends to finish */ \
      /* Ssend plus only having one request to a given destination active at a time should act as flow control. */ \
      MPI_Issend(coalescing_buf + (dest) * edge_buffer_size * 2, coalescing_counts[(dest)] * 2, INT64_T_MPI_TYPE, (dest), 0, MPI_COMM_WORLD, &sendreqs[(dest)]); \
      sendreqs_active[(dest)] = 1; \
      ++num_sendreqs_active; \
      /* Clear the buffer for the next user. */ \
      coalescing_counts[(dest)] = 0; \
    } while (0)

    START_IRECV;
    size_t i;
    for (i = 0; i < (size_t)nedges; ++i) {
      if ((i % (1 << 16)) == 0) PROCESS_REQS;
      if (edges[i * 2 + 0] == -1 || edges[i * 2 + 1] == -1) {
        continue;
      }
      SEND(edges[i * 2 + 0], edges[i * 2 + 1]);
      if (edges[i * 2 + 0] != edges[i * 2 + 1]) {
        /* Only send reverse for non-self-loops. */
        SEND(edges[i * 2 + 1], edges[i * 2 + 0]);
      }
    }
    int offset;
    for (offset = 1; offset < size; ++offset) {
      int dest = MOD_SIZE(rank + offset);
      if (coalescing_counts[dest] != 0) {
        /* Send actual data, if any. */
        FLUSH_COALESCING_BUFFER(dest);
      }
      /* Send empty message to indicate that we won't send anything else to
       * this rank (takes advantage of MPI non-overtaking rules). */
      FLUSH_COALESCING_BUFFER(dest);
    }
    while (num_senders_done < size || num_sendreqs_active > 0) PROCESS_REQS;
    free(recvbuf);
    free(coalescing_buf);
    free(coalescing_counts);
    free(sendreqs);
    free(sendreqs_active);

#undef PROCESS_REQS
#undef PROCESS_EDGE
#undef FLUSH_COALESCING_BUFFER
#undef SEND
#undef START_IRECV

  }

  /* Compute global number of vertices and count the degrees of the local
   * vertices. */
  int64_t nverts_known = 0; /* We only count vertices touched by at least one
                             * edge, and because of edge doubling each vertex
                             * incident to an edge must be the target of some
                             * copy of that edge. */
  size_t nlocalverts_orig = adj_list.nvertices;
  size_t* degrees = (size_t*)refcalloc(nlocalverts_orig, sizeof(size_t)); /* Uses zero-init */
  size_t i, j;
  for (i = 0; i < nlocalverts_orig; ++i) {
    size_t deg = 0;
    edge_page* p;
    for (p = adj_list.data[i]; p; p = p->next) {
      for (j = 0; j < 16; ++j) {
        if (p->targets[j] != -1) {
          ++deg;
          if (p->targets[j] >= nverts_known) nverts_known = p->targets[j] + 1;
        }
      }
    }
    degrees[i] = deg;
  }
  int64_t nglobalverts = 0;
  MPI_Allreduce(&nverts_known, &nglobalverts, 1, INT64_T_MPI_TYPE, MPI_MAX, MPI_COMM_WORLD);
  g->nglobalverts = nglobalverts;
  /* Compute the final number of local vertices based on the global maximum
   * vertex number. */
  size_t nlocalverts = VERTEX_LOCAL(nglobalverts + size - 1 - rank);
  g->nlocalverts = nlocalverts;
  grow_adj_list(&adj_list, nlocalverts);

  /* Build CSR data structure. */
  size_t *rowstarts = (size_t*)refmalloc((nlocalverts + 1) * sizeof(size_t), __func__);
  g->rowstarts = rowstarts;
  /* Compute offset to start of each row. */
  rowstarts[0] = 0;
  for (i = 0; i < nlocalverts; ++i) {
    rowstarts[i + 1] = rowstarts[i] + (i >= nlocalverts_orig ? 0 : degrees[i]);
  }
  size_t nlocaledges = rowstarts[nlocalverts];
  g->nlocaledges = nlocaledges;
  int64_t* column = (int64_t*)refmalloc(nlocaledges * sizeof(int64_t), __func__);
  g->column = column;
  /* Append outgoing edges for each vertex to the column array, in order. */
  for (i = 0; i < nlocalverts; ++i) {
    edge_page* p;
    int offset = 0;
    for (p = adj_list.data[i]; p; p = p->next, offset += 16) {
      size_t deg = (i >= nlocalverts_orig ? 0 : degrees[i]);
      size_t nelts = (deg - offset > 16) ? 16 : deg - offset;
      memcpy(column + rowstarts[i] + offset,
             p->targets,
             nelts * sizeof(int64_t));
    }
  }
  free(degrees); degrees = NULL;
  clear_adj_list(&adj_list);
}


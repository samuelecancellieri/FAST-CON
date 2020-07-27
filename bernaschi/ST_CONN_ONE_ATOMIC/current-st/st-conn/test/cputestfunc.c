#ifndef COMMON_H
/* This is to make PRId64 working with c++ compiler */
#ifdef __cplusplus
#define __STDC_FORMAT_MACROS
#endif
/* header of int64_t and PRId64 */
#include <inttypes.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include "defines.h"
#include "header.h"
#include "reference_common.h"
#include "cputils.h"

extern FILE *fp_struct;
extern FILE *fp_bfs;
extern int rank, size;
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
extern int lgsize;
#endif
extern int64_t MaxLabel;
extern int64_t MaxGlobalLabel;
/*
void CHECK_LOCAL_VERTEX(INT_T *pointer, INT_T nelems, 
			const char *fcaller, FILE *fout) 
{
	if (pointer == NULL) {
		fprintf(fout, "%s: in %s input array is NULL\n",
			fcaller, __func__);
		fflush(fout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nelems < 0) {
		fprintf(fout, "%s: in %s nelems < 0\n",
			fcaller, __func__);
		fflush(fout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nelems == 0) {
		fprintf(fout, "\n*** WARNING ***\n");
		fprintf(fout, "%s: in %s nelems = 0\n",
			fcaller, __func__);
		fprintf(fout, "*** WARNING ***\n\n");
		fflush(fout);
	}
	INT_T i;
	for (i=0; i < nelems; ++i) {
		if (pointer[i] < 0) {
			fprintf(fout, "%s: in %s,"
				" local vertex = %"PRI64" <= 0\n",
				fcaller, __func__, pointer[i]);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (VERTEX_LOCAL(pointer[i]) > MaxLabel) {
			fprintf(fout, "%s: in %s, pointer[i] >= MaxLabel\n",
				fcaller, __func__);
			fprintf(fout, "%s: in %s, pointer=%"PRI64"\n", 
				fcaller, __func__, pointer[i]);
			fprintf(fout, "%s: in %s, MaxLabel=%"PRId64"\n", 
				fcaller, __func__, MaxLabel);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (rank != VERTEX_OWNER(pointer[i])) {
			fprintf(fout, "%s: in %s, rank != VERTEX_OWNER(pointer[i])\n",
				fcaller, __func__);
			fprintf(fout, "%s: in %s, VERTEX_OWNER(pointer)=%d ", 
				fcaller, __func__, VERTEX_OWNER(pointer[i]));
			fprintf(fout, "%s: in %s, rank=%d\n", 
				fcaller, __func__, rank);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

}
void CHECK_GLOBAL_VERTEX(INT_T *pointer, INT_T nelems, 
			 const char *fcaller, FILE *fout) 
{
	if (pointer == NULL) {
		fprintf(fout, "%s: in %s input array is NULL\n",
			fcaller, __func__);
		fflush(fout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nelems <= 0) {
		fprintf(fout, "%s: in %s nelems = 0\n",
			fcaller, __func__);
		fflush(fout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	INT_T i;
	for (i=0; i < nelems; ++i) {
		if (pointer[i] < 0) {
			fprintf(fout, "%s: in %s,"
				" local vertex = %"PRI64" <= 0\n",
				fcaller, __func__, pointer[i]);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (pointer[i] > MaxGlobalLabel) {
			fprintf(fout, "%s: in %s, pointer[i] >= MaxGlobalLabel\n",
				fcaller, __func__);
			fprintf(fout, "%s: in %s, pointer=%"PRI64"\n", 
				fcaller, __func__, pointer[i]);
			fprintf(fout, "%s: in %s, MaxGlobalLabel=%"PRId64"\n", 
				fcaller, __func__, MaxGlobalLabel);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (VERTEX_OWNER(pointer[i]) > size) {
			fprintf(fout, "%s: in %s,  VERTEX_OWNER(pointer[i]) > size\n",
				fcaller, __func__);
			fprintf(fout, "%s: in %s, VERTEX_OWNER(pointer)=%d ", 
				fcaller, __func__, VERTEX_OWNER(pointer[i]));
			fprintf(fout, "%s: in %s, size=%d\n", 
				fcaller, __func__, size);
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

}
*/
void check_undirect_struct(INT_T nedges, INT_T *test_edges)
{
	INT_T j,u,v;	
	if (nedges < PRINT_MAX_NEDGES){
		print_edges(test_edges, nedges, fp_struct, 
		"check_undirect_struct, TEST EDGES");
	}
	
	for (j=0; j < nedges; ++j) {
		u = test_edges[2*j];
		v = test_edges[2*j + 1];
		if (u != v) {
			assert(u == test_edges[2*nedges + 2*j + 1]);
			assert(v == test_edges[2*nedges + 2*j]);
		}
	}
}

void check_split_struct(INT_T *edges, INT_T nedges, INT_T *test_edges)
{
	INT_T j;
	INT_T half = 0;
	int odd;

	if ((nedges%2) == 0) { 
		odd = 0;
	} else {          
		odd = 1;        
	}                 

	if (nedges < PRINT_MAX_NEDGES){
		print_edges(edges, nedges, fp_struct, "check_split_struct, EDGES");
		print_array(test_edges, nedges, fp_struct, "check_split_struct, " 
			   "TEST EDGES U");
		print_array(test_edges + nedges + odd, nedges, fp_struct, 
			   "check_split_struct, TEST EDGES V");
	}

	half = nedges+odd;
	INT_T *t_edges_u = &test_edges[0];
	INT_T *t_edges_v = &test_edges[nedges+odd];

	for(j=0; 2*j < (nedges+odd); ++j){
		t_edges_u[2*j] = edges[2*j];
		t_edges_v[2*j] = edges[2*j+1];
	}
	for(j=0; 2*j < (nedges+odd); ++j){
		t_edges_u[2*j+1] = edges[2*j+half];
		t_edges_v[2*j+1] = edges[2*j+1+half];
	}
}

void check_owners_struct(INT_T nedges, INT_T *test_edges)
{
	INT_T j;
	if (nedges < PRINT_MAX_NEDGES){
		print_array(test_edges, nedges, fp_struct, 
			   "check_owners_struct, TEST EDGES U");
		print_array(test_edges+nedges, nedges, fp_struct, 
			   "check_owners_struct, TEST EDGES V");
	}
	for(j=0; j < nedges; ++j){
		assert(test_edges[j] == VERTEX_OWNER(test_edges[j]));
	}
}

void check_count_struct(INT_T *h_edges, INT_T nedges, 
		 INT_T *send_count_per_proc,
		 const char *fcaller, FILE *fout)
{
	INT_T j;
	if (nedges < PRINT_MAX_NEDGES){
		print_array(h_edges, nedges, fout, 
			   "check_count_struct, H_EDGES");
		print_array(send_count_per_proc, size, fout, 
			   "check_count_struct, SEND_COUNT_PER_PROC");
	}
	INT_T count_per_proc[size];
	for (j=0; j < size; ++j) {
		count_per_proc[j] = 0;
	}
	INT_T proc;
	for(j=0; j < nedges; ++j) {
		proc = h_edges[j];
		if (proc >= size) {
			fprintf(fout, "h_edges[j] > size\n");
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		count_per_proc[proc] += 2;
	}
	for (j=0; j < size; ++j) {
		if(count_per_proc[j] != send_count_per_proc[j]){
			fprintf(fout, "count_per_proc[j] != send_count_per_proc[j]\n");
			print_array(count_per_proc, size, fout, "COUNT_PER_PROC");
			print_array(send_count_per_proc, size, fout, "SEND_COUNT_PER_PROC");
			fflush(fout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}	
	fflush(fout);
}

void check_back_struct(INT_T nedges, INT_T *test_edges, INT_T *count_per_proc, 
		INT_T *offset_per_proc)
{
	INT_T j;
	INT_T cc = 0;
	INT_T mycount = count_per_proc[rank]/2;	
	INT_T mystart = offset_per_proc[rank]/2;

	if (nedges < PRINT_MAX_NEDGES){
		print_array(test_edges, nedges, fp_struct, 
			    "check_back_struct, TEST EDGES U");
		print_array(test_edges+nedges, nedges, fp_struct, 
			    "check_back_struct, TEST EDGES V");
	}

	for (j=0; j < nedges; ++j) {
		if (rank == VERTEX_OWNER(test_edges[j])) {
			cc += 1;
		}
	}
	assert(cc == mycount);

	for (j=mystart; j < (mystart+cc); ++j) {
		if (nedges < 1) {
			fprintf(fp_struct, "check_back_struct: rank %d "
				"test_edges[%"PRI64"]=%"PRI64"\n", 
				rank, j, test_edges[j]);
		}
		assert(rank == VERTEX_OWNER(test_edges[j]));
	} 
}

void check_unsplit_struct(INT_T *edges, INT_T nedges, INT_T *test_edges, 
		   INT_T *count_per_proc, INT_T *offset_per_proc)
{
	INT_T j;
	INT_T cc = 0;
	INT_T mycount = count_per_proc[rank];	
	INT_T mystart = offset_per_proc[rank];

	if (nedges < PRINT_MAX_NEDGES){
		print_edges(edges, nedges, fp_struct, 
			    "check_unsplit_struct, EDGES");
		print_edges(test_edges, nedges, fp_struct, 
			    "check_unsplit_struct, TEST EDGES U");
	}

	if (size == 1) {
		for (j=0; j < 2*nedges; ++j) {
			assert(edges[j] == test_edges[j]);
		}
		return;
	}

	for (j=0; j < nedges; ++j) {
		if (rank == VERTEX_OWNER(test_edges[2*j])) {
			cc += 2;
		}
	}
	assert(cc == mycount);

	for (j=mystart; 2*j < (mystart+cc); ++j) {
		if (nedges < 1) {
			fprintf(fp_struct, "check_unsplit_struct: rank %d " 
				"test_edges[%"PRI64"]=%"PRI64"\n", 
				rank, 2*j, test_edges[2*j]);
		}
		assert(rank == VERTEX_OWNER(test_edges[2*j]));
	} 
}

void check_offset_struct(INT_T nverts, INT_T *test_edges)
{
	if (nverts < PRINT_MAX_NVERTS){
		print_edges(test_edges, nverts, fp_struct, 
			    "check_offset_struct, TEST EDGES OFFSET");
	}

	assert(test_edges[0] == 0);

	INT_T X = 0;
	INT_T j;
	for (j=1; j < (nverts-1); j++) {
			if((test_edges[j] > test_edges[j+1]) && (test_edges[j+1] != X)) {
				fprintf(stderr,"test_edges[%"PRI64"]=%"PRI64"\n", 
					j, test_edges[j]);
				fprintf(stderr,"test_edges[%"PRI64"]=%"PRI64"\n", 
					(j+1), test_edges[j+1]);
				fprintf(stderr,"offset are not ordered! Quit.\n");
			}
			if (test_edges[j+1] != X) {
				assert((test_edges[j] <= test_edges[j+1]));
			}
	}
}

// my_nverts are the vertices owned by the processor
// nglobalverts is the result of MPI_Allreduce(&my_nverts, nglobalverts, ...)
void check_reduce_verts(INT_T my_nverts, INT_T nglobalverts)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if (nglobalverts < 0) {
		fprintf(stderr, "rank %d, %s: nglobalverts < 0\n",
						rank, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nglobalverts > (MaxGlobalLabel + 1)) {
		fprintf(stderr, "rank %d, %s: nglobalverts > (MaxGlobalLabel + 1)\n",
						rank, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
#endif
}

// csr_graph is the data structure used in the grap500 reference code
// this function check that the data structure is consistent
void check_csr_struct(csr_graph *cg)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if ((cg == NULL) || (cg->column == NULL) || (cg->rowstarts == NULL)) {
		fprintf(stderr, "rank %d, %s: one of the input array is NULL\n",
		rank, __func__);
	}
	size_t i;
	for (i = 0; i < cg->nlocaledges; ++i) {
		if (cg->nlocalverts > (MaxLabel+1) ) {
			fprintf(stderr, "rank %d, %s: cg->nlocalverts > MaxLabel\n",
				rank, __func__);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (cg->nglobalverts > (MaxGlobalLabel+1) ) {
			fprintf(stderr, "rank %d, %s: cg->nglobalverts > MaxGlobalLabel\n",
				rank, __func__);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (cg->column[i] < 0) {
			fprintf(stderr, "rank %d, %s: cg->column[%zu]=%"PRId64" < 0\n",
				rank, __func__, i, cg->column[i]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
#endif
}


// Check the array to send
void check_back_bfs(INT_T nelems, INT_T *h_array)
{
	if (nelems < PRINT_MAX_NEDGES) {
		print_array(h_array, nelems, fp_bfs,
			    "check_back_bfs: H_ARRAY");
	}
	INT_T i;
	for (i=0; i < nelems; ++i) {
		assert(h_array[i] >= 0);
		if (rank == VERTEX_OWNER(h_array[i])) {
			if(VERTEX_LOCAL(h_array[i]) > MaxLabel) {
				fprintf(fp_bfs, "h_array[i] >= MaxLabel\n");
				fprintf(fp_bfs, "h_array=%"PRI64" ", h_array[i]);
				fprintf(fp_bfs, "MaxLabel=%"PRId64"\n", MaxLabel);
				fflush(fp_bfs);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		} else {
			assert(h_array[i] <= MaxGlobalLabel);
		}
	}
	fflush(fp_bfs);
}

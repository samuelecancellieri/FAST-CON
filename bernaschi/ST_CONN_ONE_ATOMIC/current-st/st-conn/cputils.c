/* This is to make PRId64 working with c++ compiler */
#ifdef __cplusplus
#define __STDC_FORMAT_MACROS
#endif
/* header of int64_t and PRId64 */
#include <inttypes.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h> 
#include <mpi.h>
#include <cuda_runtime.h>

#include "defines.h"
#include "header.h"
#include "cputils.h"

extern int64_t MaxLabel;
extern int64_t MaxGlobalLabel;

extern FILE *fp_bfs;
extern FILE *fp_time;
extern int dbg_lvl;

extern int rank, size;
extern int lgsize;

void setup_globals_MPI() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
  //size_minus_one = size - 1;
  if (/* Check for power of 2 */ (size & (size - 1)) != 0) {
    fprintf(stderr, "Number of processes %d is not a power of two, yet SIZE_MUST_BE_A_POWER_OF_TWO is defined in main.cpp.\n", size);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  for (lgsize = 0; lgsize < size; ++lgsize) {
    if ((1 << lgsize) == size) break;
  }
  assert (lgsize < size);
#endif
}

void* callmalloc(size_t nbytes, const char* funcname) {
	void* p = malloc(nbytes);
	if (!p) {
		fprintf(stderr, "in function: %s, malloc() failed for size %zu\n", 
						funcname, nbytes);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
  return p;
}

void* callrealloc(void* p, size_t nbytes, const char* funcname) {
	p = realloc(p, nbytes);
	if (!p) {
		fprintf(stderr, "in function: %s, realloc() failed for size %zu\n",
						funcname, nbytes);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
  return p;
}

void freemem(void *p) {
	if (p != NULL) {
		free(p);
		p = NULL;
	}
}

int exclusive_scan(INT_T* count_array, INT_T* offset_array, INT_T count_nelems, INT_T offset_nelems)
{
	if ((count_array == NULL) || (offset_array == NULL)) {
		fprintf(stderr, " Null input array, quit.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	assert(offset_nelems == (count_nelems+1));
	INT_T i;
	INT_T tmpcount = 0;

	double tstart, tstop, t;
	START_TIMER(dbg_lvl, tstart);

	offset_array[0] = 0;
	for (i = 0; i < count_nelems; ++i) {
		tmpcount += count_array[i];
		offset_array[i+1] = tmpcount;
	}
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
 	return 0;
}

int exclusive_scan_pad(INT_T* count_array,  INT_T* offset_array, INT_T* padded_offset_array,
		                INT_T count_nelems, INT_T offset_nelems)
{
	if ((count_array == NULL) || (offset_array == NULL)) {
		fprintf(stderr, " Null input array, quit.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	assert(offset_nelems == (count_nelems+1));

	INT_T i;
	INT_T tmpcount = 0;

	offset_array[0] = 0;
	padded_offset_array[0] = 0;
	for (i = 0; i < count_nelems; ++i) {
		tmpcount = ( (ROUNDING -  count_array[i] % ROUNDING) % ROUNDING);
		offset_array[i+1] = offset_array[i] + count_array[i];
		count_array[i] += tmpcount;
		padded_offset_array[i+1] = padded_offset_array[i] + count_array[i];
	}

 	return 0;
}

int exclusive_scan_INT32(INT32_T* count_array, INT32_T* offset_array, INT32_T count_nelems, INT32_T offset_nelems)
{
	if ((count_array == NULL) || (offset_array == NULL)) {
		fprintf(stderr, " Null input array, quit.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	assert(offset_nelems == (count_nelems+1));
	INT32_T i;
	INT32_T tmpcount = 0;

	double tstart, tstop, t;
	START_TIMER(dbg_lvl, tstart);

	offset_array[0] = 0;
	for (i = 0; i < count_nelems; ++i) {
		tmpcount += count_array[i];
		offset_array[i+1] = tmpcount;
	}
	STOP_TIMER(dbg_lvl, tstart, tstop, t, fp_bfs, __func__);
 	return 0;
}
void print_edges(INT_T* edges, INT_T nedges, FILE *fout, const char *func_name)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if (fout == NULL){
		fprintf(stderr, "%s: in print_edges fout is NULL\n", func_name);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (edges == NULL){
		fprintf(stderr, "%s: in print_edges edges is NULL\n", func_name);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nedges <= 0){
		fprintf(stderr, "%s in print_edges nedges <= 0\n", func_name);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	INT_T i;
	if(nedges <= PRINT_MAX_NEDGES){
		fprintf(fout, "\n*** *** EDGES %s *** ***\n", func_name);
		fprintf(fout, "The number of edges in %s is %"PRI64"\n", func_name, nedges);
		for (i=0; i < nedges; ++i){
			fprintf(fout, "(%"PRI64",%"PRI64") ", edges[2*i], edges[2*i + 1]);
		}
		fprintf(fout, "\n*** *** EDGES END %s *** ***\n", func_name);
	} else {
		fprintf(fout, "The number of edges in %s is %"PRI64"\n", func_name, nedges);
		return;
	}
	fflush(fout);
#endif
}

void print_array_32t(INT32_T *in, INT_T nelems, FILE* fout, const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if(in == NULL){
		fprintf(stderr, "rank %d, %s: in %s input file is NULL\n",
						rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if(fout == NULL){
	  fprintf(stderr, "rank %d, %s: in %s fout is NULL\n",
						rank, fcaller, __func__);
	}

	if (nelems > PRINT_MAX_NEDGES){
		fprintf(fout, "Array in %s has %"PRId64" elements\n", fcaller, nelems);
		return;
	}

	int j;
	fprintf(fout, "\n*** *** %s *** ***\n", fcaller);
	fprintf(fout, "Array in %s has %"PRId64" elements\n[", fcaller, nelems);
	for (j=0; j < nelems; ++j){
		fprintf(fout, "%d ", in[j]);
	}
	fprintf(fout, "]\n*** *** END %s *** ***\n", fcaller);
	fflush(fout);
#endif
}


void print_edges32(INT32_T* edges, INT_T nedges, FILE *fout, const char *func_name)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if (fout == NULL){
		fprintf(stderr, "%s: in print_edges32 fout is NULL\n", func_name);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (edges == NULL){
		fprintf(stderr, "%s: in print_edges32 edges is NULL\n", func_name);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nedges <= 0){
		fprintf(stderr, "%s in print_edges32 nedges <= 0\n", func_name);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	INT_T i;
	if(nedges <= PRINT_MAX_NEDGES){
		fprintf(fout, "\n*** *** EDGES 32 %s *** ***\n", func_name);
		fprintf(fout, "The number of edges in %s is %"PRI64"\n", func_name, nedges);
		for (i=0; i < nedges; ++i){
			fprintf(fout, "(%i,%i) ", edges[2*i], edges[2*i + 1]);
		}
		fprintf(fout, "\n*** *** EDGES END %s *** ***\n", func_name);
	} else {
		fprintf(fout, "The number of edges in %s is %"PRI64"\n", func_name, nedges);
		return;
	}
	fflush(fout);
#endif
}

int checkGlobalEdgeList(INT_T* edges, INT_T nedges, FILE *fout, const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if (edges == NULL){
		fprintf(stderr, "rank %d, %s: null edge list,"
						" quit.\n", rank, fcaller);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nedges == 0){
		fprintf(stderr, "rank %d, %s: nedges = 0,"
						" quit\n", rank, fcaller);
	}

	INT_T j;
	for(j=0; j < 2*nedges; ++j){
		if (edges[j] > MaxGlobalLabel){
			fprintf(stderr, "rank %d, %s: edges[%"PRI64"]=%"PRI64" > "
							"MaxGlobalLabel=%"PRId64"\n",
							rank, fcaller, j, edges[j], MaxGlobalLabel);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (edges[j] < 0) {
			fprintf(stderr, "rank %d, %s: edges[%"PRI64"]=%"PRI64" < 0 \n",
							rank, fcaller, j, edges[j]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
#endif
	return 0;
}

int checkLocalEdgeList(INT_T* edges, INT_T nedges, FILE *fout,
											const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if (edges == NULL){
		fprintf(stderr, "rank %d, %s: Null edge list, quit.\n", rank, fcaller);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nedges == 0){
		fprintf(stderr, "rank %d, %s: nedges = 0, quit\n", rank, fcaller);
	}

	INT_T j;
	for(j=0; j < nedges; ++j){
		if (rank != VERTEX_OWNER(edges[2*j])) {
			fprintf(stderr, "rank %d, %s: I'm not the owner of"
							" edges[%"PRI64"]=%"PRI64"\n", rank, fcaller, 
							2*j, edges[2*j]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (VERTEX_LOCAL(edges[2*j]) > MaxLabel){
			fprintf(stderr, "rank %d, %s: edges[%"PRI64"]=%"PRI64" > "
							"MaxLabel=%"PRId64"\n", rank, fcaller, 
							2*j, edges[2*j], MaxLabel);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (edges[2*j+1] > MaxGlobalLabel){
			fprintf(stderr, "rank %d, %s: edges[%"PRI64"]=%"PRI64" > "
							"MaxGlobalLabel=%"PRId64"\n", rank, fcaller, 
							(2*j+1), edges[2*j+1], MaxGlobalLabel);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (edges[2*j] < 0) {
			fprintf(stderr, "rank %d, %s: edges[%"PRI64"]=%"PRI64" < 0 \n",
							rank, fcaller, 2*j, edges[2*j]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (edges[2*j+1] < 0) {
			fprintf(stderr, "rank %d, %s: edges[%"PRI64"]=%"PRI64" < 0 \n",
							rank, fcaller, 2*j+1, edges[2*j+1]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
#endif
	return 0;
}

void print_array(INT_T *in, INT_T nelems, FILE* fout, const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if(in == NULL){
		fprintf(stderr, "rank %d, %s: in %s input file is NULL\n",
			rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if(fout == NULL){
	  fprintf(stderr, "rank %d, %s: in %s fout is NULL\n",
		  rank, fcaller, __func__);
	}

	if (nelems > PRINT_MAX_NEDGES){
		fprintf(fout, "Array in %s has %"PRI64" elements\n", fcaller, nelems);
		return;
	}

	int j;
	fprintf(fout, "\n*** *** %s *** ***\n", fcaller);
	fprintf(fout, "Array %s has %"PRI64" elements\n[", fcaller, nelems);
	for (j=0; j < nelems; ++j){
		fprintf(fout, "%"PRI64" ", in[j]);
	}
	fprintf(fout, "]\n*** *** END %s *** ***\n", fcaller);
	fflush(fout);
#endif
}

void print_array_64t(int64_t *in, int64_t nelems, FILE* fout,
										const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if(in == NULL){
		fprintf(stderr, "rank %d, %s: in %s input file is NULL\n",
						rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if(fout == NULL){
	  fprintf(stderr, "rank %d, %s: in %s fout is NULL\n",
						rank, fcaller, __func__);
	}

	if (nelems > PRINT_MAX_NEDGES){
		fprintf(fout, "Array in %s has %"PRId64" elements\n", fcaller, nelems);
		return;
	}

	int j;
	fprintf(fout, "\n*** *** %s *** ***\n", fcaller);
	fprintf(fout, "Array in %s has %"PRId64" elements\n[", fcaller, nelems);
	for (j=0; j < nelems; ++j){
		fprintf(fout, "%"PRId64" ", in[j]);
	}
	fprintf(fout, "]\n*** *** END %s *** ***\n", fcaller);
	fflush(fout);
#endif
}

void print_uncolor_array_64t(int64_t *in, int64_t nelems, FILE* fout, const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if(in == NULL){
		fprintf(stderr, "rank %d, %s: in %s input file is NULL\n",
						rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if(fout == NULL){
	  fprintf(stderr, "rank %d, %s: in %s fout is NULL\n",
						rank, fcaller, __func__);
	}

	if (nelems > PRINT_MAX_NEDGES){
		fprintf(fout, "Array in %s has %"PRId64" elements\n", fcaller, nelems);
		return;
	}

	int j;
	int64_t unc;
	fprintf(fout, "\n*** *** %s *** ***\n", fcaller);
	fprintf(fout, "Array in %s has %"PRId64" elements\n[", fcaller, nelems);
	for (j=0; j < nelems; ++j){
		if (in[j] != -1) unc = in[j] & (~COLOR_MASK_64); else unc = in[j];
		fprintf(fout, "%"PRId64" ", unc);
	}
	fprintf(fout, "]\n*** *** END %s *** ***\n", fcaller);
	fflush(fout);
#endif
}


void print_fullarray_64t(int64_t *in, int64_t nelems, FILE* fout, const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if(in == NULL){
		fprintf(stderr, "rank %d, %s: in %s input file is NULL\n",
						rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if(fout == NULL){
	  fprintf(stderr, "rank %d, %s: in %s fout is NULL\n",
						rank, fcaller, __func__);
	}

	fprintf(fout, "Array in %s has %"PRId64" elements\n", fcaller, nelems);

	int j;
	fprintf(fout, "\n*** *** %s *** ***\n", fcaller);
	fprintf(fout, "Array in %s has %"PRId64" elements\n[", fcaller, nelems);
	for (j=0; j < nelems; ++j){
		fprintf(fout, "%"PRId64" ", in[j]);
	}
	fprintf(fout, "]\n*** *** END %s *** ***\n", fcaller);
	fflush(fout);
#endif
}

void print_array_uint(unsigned int *in, unsigned int nelems, FILE* fout,
											const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if(in == NULL){
		fprintf(stderr, "rank %d, %s: in %s input file is NULL\n",
						rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if(fout == NULL){
	  fprintf(stderr, "rank %d, %s: in %s fout is NULL\n",
						rank, fcaller, __func__);
	}

	if (nelems > PRINT_MAX_NEDGES){
		fprintf(fout, "Array in %s has %u elements\n", fcaller, nelems);
		return;
	}

	int j;
	fprintf(fout, "\n*** *** %s *** ***\n", fcaller);
	fprintf(fout, "Array in %s has %u elements\n[", fcaller, nelems);
	for (j=0; j < (int)nelems; ++j){
		fprintf(fout, "%d ", in[j]);
	}
	fprintf(fout, "]\n*** *** END %s *** ***\n", fcaller);
	fflush(fout);
#endif
}

// If A > B Abort!
void CHECK_SIZE(const char *A_name, INT_T A_value, 
		const char *B_name, INT_T B_value, 
		const char *fcaller)
{
//#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if (A_value > B_value) {
		fprintf(stderr, "%s=%"PRI64"\n", A_name, A_value);
		fprintf(stderr, "%s=%"PRI64"\n", B_name, B_value);
		fprintf(stderr, "%s > %s! Abort\n", A_name, B_name);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
//#endif
}
void CHECK_INPUT(INT_T *pointer, INT_T nelems, const char *fcaller) 
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if (pointer == NULL) {
		fprintf(stderr, "rank %d, %s: in %s input array is NULL\n",
			rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nelems <= 0) {
		fprintf(stderr, "rank %d, %s: in %s nelems <= 0\n",
			rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
#endif
}

void CHECK_INPUT32(INT32_T *pointer, INT_T nelems, const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if (pointer == NULL) {
		fprintf(stderr, "rank %d, %s: in %s input array is NULL\n",
			rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (nelems <= 0) {
		fprintf(stderr, "rank %d, %s: in %s nelems <= 0\n",
			rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
#endif
}
/*
void PRINT_TIME(const char *fcaller, FILE *fout, double time)
{
#ifdef DBG_TIME
	if(fout == NULL) {
		fprintf(stderr, "%s in %s: error opening time file\n",
			fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	fprintf(fout, "TIME: %s: %f\n", fcaller, time);	
	fflush(fout);
#endif
}
void PRINT_SPACE(const char *fcaller, FILE *fout, 
		 const char *myname, double space)
{
#ifdef DBG_TIME
	if(fout == NULL) {
		fprintf(stderr, "%s in %s: error opening time file\n",
			fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if(myname == NULL) {
		fprintf(stderr, "%s in %s: myname == NULL\n",
			fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	fprintf(fout, "SPACE: %s: %s %.1f\n", fcaller, myname, space);	
	fflush(fout);
#endif
}
*/

void PRINT_TIME(const char *fcaller, FILE *fout, double time)
{
#if defined (DBG_LEVEL) && (DBG_TIME)
	if(fout == NULL) {
		fprintf(stderr, "%s in %s: error opening time file\n",
			fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	fprintf(fout, "TIME:%s:void:%f:\n", fcaller, time);
	fflush(fout);
#endif
}
void PRINT_SPACE(const char *fcaller, FILE *fout,
		const char *myname, double space)
{
#if defined (DBG_LEVEL) && (DBG_TIME)
	if(fout == NULL) {
		fprintf(stderr, "%s in %s: error opening file\n",
			fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if(myname == NULL) {
		fprintf(stderr, "%s in %s: myname == NULL\n",
			fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	fprintf(fout, "SPACE:%s:%s:%.2f:\n", fcaller, myname, space);
	fflush(fout);
#endif
}

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
#include <cuda_runtime.h>
#include <mpi.h>
//#include <omp.h>

#include "reference_common.h"
#include "header.h"
#include "gputils.h"
#include "cputils.h"
#include "adj_func.h"

#include "cputestfunc.h"

extern int rank, size;
extern int64_t MaxLabel;
extern int64_t MaxGlobalLabel;
//#define PRINT_MAX_NVERTS 100000000

int init_adjlist(adjlist *adj)
{
	if (adj != NULL){
		adj->edges = NULL;
		adj->offset = NULL;
		adj->degree = NULL;
		adj->nverts = 0;
		adj->nedges = 0;
	} else if (adj == NULL) {
		fprintf(stderr,"init_adjlist: trying to initialize a NULL "
						"adjacency list!\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return 0;
}

int init_bitmask(mask *bitmask)
{
	if (bitmask != NULL){
		bitmask->unique_edges = NULL;
		bitmask->pedges = NULL;
		bitmask->pverts = NULL;
		bitmask->mask = NULL;
		bitmask->proc_offset = NULL;
		bitmask->p_nelems = 0;
		bitmask->m_nelems = 0;
	} else if (bitmask == NULL) {
		fprintf(stderr,"init_bitmask: trying to initialize a NULL bitmask!\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return 0;
}

int free_adjlist(adjlist *adj)
{
	if (adj != NULL){
		if (adj->edges != NULL) free(adj->edges);
		if (adj->offset != NULL) free(adj->offset);
		if (adj->degree != NULL) free(adj->degree);
		adj = NULL;
	} else {
		fprintf(stderr,"free_adjlist: trying to free a NULL pointer!\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return 0;
}

int copyAdjDeviceToHost(adjlist* hg, adjlist *dg)
{ 
	if (dg == NULL){
		fprintf(stderr,"rank %d, in copyAdjDeviceToHost dg is NULL\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	} else {
	    if ((dg->edges == NULL)||(dg->offset == NULL)||(dg->degree == NULL)){
			fprintf(stderr, "rank %d, in copyAdjDeviceToHost one of dg-> is NULL\n", rank);
    			MPI_Abort(MPI_COMM_WORLD, 1);	
		}
	}	
	
	if ((dg->nverts <= 0) || (dg->nedges <= 0)){
		fprintf(stderr,"rank %d in copyAdjDeviceToHost nverts or nedges <= 0\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if(dg->nverts > (MaxLabel+1)){
		fprintf(stderr, "rank %d, check adjacency list error: nverts > MaxLabel + 1\n", rank);
    		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	/* copy back to host */
	hg->nedges = dg->nedges;
	hg->nverts = dg->nverts;
	hg->edges = NULL;
	hg->offset = NULL;
	hg->degree = NULL;
	
	hg->edges = (INT_T*)callmalloc((size_t)(2*hg->nedges*sizeof(INT_T)), "copyAdjDeviceToHost: h_edges");
	hg->offset = (INT_T*)callmalloc((size_t)(2*(hg->nverts+1)*sizeof(INT_T)), "copyAdjDeviceToHost: h_offset");
	hg->degree = (INT_T*)callmalloc((size_t)(hg->nverts*sizeof(INT_T)), "copyAdjDeviceToHost: h_offset");

	cudaMemcpy(hg->edges, dg->edges, (hg->nedges*sizeof(INT_T)), cudaMemcpyDeviceToHost);
	checkCUDAError("copyAdjDeviceToHost: dg->edges -> hg->edges");
	
	cudaMemcpy(hg->offset, dg->offset, (2*(hg->nverts+1)*sizeof(INT_T)), cudaMemcpyDeviceToHost);
	checkCUDAError("copyAdjDeviceToHost: dg->offset -> hg->offset");
	
	cudaMemcpy(hg->degree, dg->degree, (hg->nverts*sizeof(INT_T)), cudaMemcpyDeviceToHost);
	checkCUDAError("copyAdjDeviceToHost: dg->degree -> hg->degree");
	
	return 0;
}

int copyAdjHostToDevice(adjlist* dg, adjlist *hg)
{ 
  if (hg == NULL){
    fprintf(stderr,"rank %d, in copyAdjHostToDevice dg is NULL\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else {
    if ((hg->edges == NULL)||(hg->offset == NULL)||(hg->degree == NULL)){
			fprintf(stderr, "rank %d, in copyAdjHostToDevice one of hg-> is NULL\n",
							rank);
    	MPI_Abort(MPI_COMM_WORLD, 1);
		}
  }
	
	if ((hg->nverts <= 0) || (hg->nedges <= 0)){
		fprintf(stderr,"rank %d in copyAdjHostToDevice nverts or nedges <= 0\n", 
						rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if(hg->nverts > (MaxLabel+1)){
		fprintf(stderr, "rank %d, check adjacency list error: " "nverts > MaxLabel + 1\n", rank);
    		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	/* copy to Device */
	dg->nedges = hg->nedges;
	dg->nverts = hg->nverts;
	dg->edges = NULL;
	dg->offset = NULL;
	dg->degree = NULL;
	
	cudaMalloc((void**)&dg->edges, (size_t)(dg->nedges*sizeof(INT_T)));
	cudaMalloc((void**)&dg->offset, (size_t)(2*(dg->nverts+1)*sizeof(INT_T)));
	cudaMalloc((void**)&dg->degree, (size_t)(dg->nverts*sizeof(INT_T)));

	cudaMemcpy(dg->edges, hg->edges, (dg->nedges*sizeof(INT_T)), 
						cudaMemcpyHostToDevice);
	checkCUDAError("copyAdjHostToDevice: hg->edges -> dg->edges");
	
	cudaMemcpy(dg->offset, hg->offset, (2*(dg->nverts+1)*sizeof(INT_T)),
						cudaMemcpyHostToDevice);
	checkCUDAError("copyAdjHostToDevice: hg->offset -> dg->offset");
	
	cudaMemcpy(dg->degree, hg->degree, (dg->nverts*sizeof(INT_T)), 
						cudaMemcpyHostToDevice);
	checkCUDAError("copyAdjHostToDevice: hg->degree -> dg->degree");
	
	return 0;
}

int print_adjlist(adjlist *adj, FILE *fout, const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
  if (adj == NULL){
    fprintf(stderr,"%s: rank %d, in %s adj is NULL\n",
						fcaller, rank, __func__);
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else {
    if ((adj->edges == NULL)||(adj->offset == NULL)||(adj->degree == NULL)){
			fprintf(stderr, "%s: rank %d, in %s adj is NULL\n", 
							fcaller, rank, __func__);
    	MPI_Abort(MPI_COMM_WORLD, 1);
		}
  }
	
	if ((adj->nverts <= 0) || (adj->nedges <= 0)){
		fprintf(stderr,"%s: rank %d, in %s nverts or nedges <= 0",
						fcaller, rank, __func__);
	}

	INT_T ii; //number of vertices
	INT_T jj;
	INT_T off_s, off_e;
	
	fprintf(fout,"\n*** *** ** %s: ADJLIST ** *** ***\n", fcaller);
	fprintf(fout,"Processor %d says nedges=%"PRI64"\n", rank, adj->nedges);
	fprintf(fout,"Processor %d says nverts=%"PRI64"\n", rank, adj->nverts);

	if (adj->nverts > PRINT_MAX_NVERTS){
		fprintf(fout,"*** *** ** %s: ADJLIST END ** *** ***\n", fcaller);
		return 0;
	}	
	for (ii=0; ii<adj->nverts; ii += 1){
		fprintf(fout,"%"PRI64" -> ", VERTEX_TO_GLOBAL(ii));
		off_s = adj->offset[2*ii];
		off_e = adj->offset[2*ii+1];
		for (jj=off_s; jj<off_e; ++jj){
			fprintf(fout,"%"PRI64", ", adj->edges[jj]);
		}
		fprintf(fout,"\n");
	}
	
	fprintf(fout,"*** *** ** %s: ADJLIST END ** *** ***\n", fcaller);
#endif
	return 0;
}

int check_adjlist(adjlist *adj, FILE *fout, const char *fcaller)
{

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if (adj == NULL){
		fprintf(stderr,"rank %d, in %s adj is NULL\n", rank, fcaller);
		MPI_Abort(MPI_COMM_WORLD, 1);
	} else {
		if ((adj->edges == NULL)||(adj->offset == NULL)||(adj->degree == NULL)){
			fprintf(stderr, "rank %d, %s one of adj-> is NULL\n", rank, fcaller);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	if ((adj->nverts <= 0) || (adj->nedges <= 0)){
		fprintf(stderr,"rank %d in %s nverts or nedges <= 0\n", rank, fcaller);
	}

	fprintf(fout, "rank %d, %s:" " nverts=%"PRI64" nedges=%"PRI64"\n",
		rank, fcaller, adj->nverts, adj->nedges);

	if(adj->nverts > (MaxLabel+1)){
		fprintf(stderr, "rank %d, %s list error: " "nverts > MaxLabel + 1\n", 
			rank, fcaller);
    		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	INT_T ii; 
	INT_T jj;
	INT_T off_s, off_e;
	INT_T V;
	
	for (ii=0; ii<adj->nverts; ii += 1){
		assert(ii <= MaxLabel);
		off_s = adj->offset[2*ii];
		off_e = adj->offset[2*ii+1];
		assert(adj->degree[ii] == (off_e - off_s));
		for (jj=off_s; jj<off_e; ++jj){
			V = adj->edges[jj];
			assert(V <= MaxGlobalLabel);
			if (adj->nverts < PRINT_MAX_NVERTS){
				//fprintf(fout, "check_adjlist: jj=%"PRI64" V=%"PRI64" "
				//	"MaxGlobalLabel=%"PRI64"\n", jj, V, MaxGlobalLabel);
			}
		}
	}
#endif
	return 0;
}

static int compare_INT_T(const void* a, const void* b) {
  INT_T aa = *(const INT_T*)a;
  INT_T bb = *(const INT_T*)b;
  return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

void print_adjlist_stat(adjlist *adj, const char *fcaller, FILE *fout)
{
	INT_T nverts = adj->nverts;
	INT_T nedges = adj->nedges;
	INT_T *degree = adj->degree;
			
	fprintf(fout, "rank %d in %s in %s: nverts=%"PRI64"\n", rank, __func__, fcaller, nverts);
	fprintf(fout, "rank %d in %s in %s: nedges=%"PRI64"\n", rank, __func__, fcaller, nedges);

	// Sort degree to find max and min
	qsort(degree, nverts, sizeof(INT_T), compare_INT_T);

	if (nverts < PRINT_MAX_NVERTS) {
		print_array(degree, nverts, fout, "ARRAY OF DEGREE");
	}

	// The first ten elements of sorted degree
	int i, maxi;
	maxi = (nverts > 10) ? 10 : nverts;
	for (i=0; i < maxi; ++i) {
		unsigned long long idx = nverts -1 -i;
		fprintf(fout, "rank %d in %s in %s: degree[%llu]=%"PRI64"\n", rank, __func__, fcaller, idx, degree[idx]);
	}
	// The min greater then 0 and the first 0
	INT_T j=1;
	while (degree[nverts-j] == 0)
		j++;
	fprintf(fout, "rank %d in %s in %s: min degree[%"PRI64"]=%"PRI64"\n", rank, __func__, fcaller, j, degree[j]);
	fprintf(fout, "rank %d in %s in %s: min degree[%"PRI64"]=%"PRI64"\n", rank, __func__, fcaller, j+1, degree[j+1]);

	// Mean
	INT_T mean = 0;
	for (j=0; j < nverts; ++j) {
		mean += degree[j];
	}
	mean = mean/nverts;
	fprintf(fout, "rank %d in %s in %s: mean=%"PRI64"\n", rank, __func__, fcaller, mean);

	// How many degree are greater of mean?
	long long int gtmean, gt10mean, gt100mean;
	gtmean = gt10mean = gt100mean = -2*nverts;
	for (j=0; j < nverts; ++j)
		if (degree[j] > mean)
			break;
	gtmean=j;
	for (j=0; j < nverts; ++j) 
		if (degree[j] > 10*mean)
			break;
	gt10mean=j;
	for (j=0; j < nverts; ++j) 
		if (degree[j] > 100*mean)
			break;
	gt100mean=j;
	fprintf(fout, "rank %d in %s in %s: number of degree lesser equal than mean=%lld\n", rank, __func__, fcaller, gtmean);
	fprintf(fout, "rank %d in %s in %s: number of degree greater than mean=%lld\n", rank, __func__, fcaller, (nverts - gtmean));
	fprintf(fout, "rank %d in %s in %s: number of degree greater than 10mean=%lld\n", rank, __func__, fcaller, (nverts - gt10mean));
	fprintf(fout, "rank %d in %s in %s: number of degree greater than 100mean=%llu\n", rank, __func__, fcaller, (nverts - gt100mean));
	
}

int count_visited_edges(csr_graph *cg, double *edge_counts, 
			int64_t *edge_visit_count, 
			int64_t bfs_root_idx, 
			int64_t *h_pred)
{
	int64_t v_local;
	int64_t j;
	for (v_local = 0; v_local < (int64_t)cg->nlocalverts; ++v_local) {
		if (h_pred[v_local] != -1) {
			INT_T ei = cg->rowstarts[v_local];
			INT_T ei_end = cg->rowstarts[v_local+1];
			for (j = ei; j < ei_end; ++j) {
				if (cg->column[j] <= VERTEX_TO_GLOBAL(v_local))	{
					++edge_visit_count[0];
				}
			}
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, edge_visit_count, 1, INT64_T_MPI_TYPE, 
		      MPI_SUM, MPI_COMM_WORLD);
	edge_counts[bfs_root_idx] = (double)edge_visit_count[0];
	return 0;
}

int convert_to_csr(adjlist *hg, csr_graph *cg, int64_t nglobalverts, FILE *fout)
{
	if ((hg == NULL) || (hg->edges == NULL) || (hg->offset == NULL)) {
		fprintf(stderr, "rank %d, convert_to_csr: NULL input\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if ((hg->nverts <= 0) || (hg->nedges <= 0)) {
		fprintf(stderr, "rank %d, convert_to_csr:"
			" nverts or nedges <=0\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (hg->nverts > (MaxGlobalLabel+1)) {
		fprintf(stderr, "rank %d, convert_to_csr:"
		" nverts > MaxGlobalLabel\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	cg->nlocalverts = (size_t)hg->nverts;
	cg->nlocaledges = (size_t)hg->nedges;
	cg->nglobalverts = nglobalverts;

	cg->rowstarts = (size_t *)callmalloc((cg->nlocalverts+1)*sizeof(size_t), 
					     "convert_to_csr: malloc rowstarts");

	cg->column = (int64_t *)callmalloc((cg->nlocaledges+1)*sizeof(int64_t), 
					   "convert_to_csr: malloc column");

	INT_T* g_offset = NULL;
	g_offset = (INT_T *)callmalloc((hg->nverts+1)*sizeof(INT_T), 
				       "convert_to_csr: malloc g_offset");
	memset(g_offset, 0, (hg->nverts+1)*sizeof(INT_T));
	exclusive_scan(hg->degree, g_offset, hg->nverts, (hg->nverts+1));

	INT_T i;
	for (i = 0; i < (hg->nverts+1); ++i)
		cg->rowstarts[i] = (size_t)g_offset[i];

	for (i = 0; i < hg->nedges; ++i)
		cg->column[i] = (int64_t)hg->edges[i];

	// Check
	// check_csr_struct(cg);
	print_csr_graph(cg, fout, __func__);

	free(g_offset); g_offset = NULL;
	return 0;
}

int convert_csr_to_adj(csr_graph *cg, adjlist *hg, FILE *fout)
{
	if ((cg == NULL) || (cg->rowstarts == NULL) || (cg->column == NULL)) {
			fprintf(stderr, "rank %d, convert_csr_to_adj: NULL input\n", rank);
			MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if ((cg->nlocalverts <= 0) || (cg->nlocaledges <= 0)) {
		fprintf(stderr, "rank %d, convert_csr_to_adj:"
			" nlocalverts or nlocaledges <=0\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	hg->nverts = cg->nlocalverts;
	hg->nedges = cg->nlocaledges;

	if (hg->nverts > (MaxGlobalLabel+1)) {
		fprintf(stderr, "rank %d, convert_csr_to_adj:"
		" nlocalverts > MaxGlobalLabel\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	hg->edges = (INT_T*)callmalloc((hg->nedges)*sizeof(INT_T), "convert_csr_to_adj: malloc edges");
	hg->offset = (INT_T*)callmalloc(2*(hg->nverts+1)*sizeof(INT_T), "convert_csr_to_adj: malloc offset");
	hg->degree = (INT_T*)callmalloc((hg->nverts)*sizeof(INT_T), "convert_csr_to_adj: malloc degree");

	INT_T i;

//#pragma omp parallel for
	for (i = 0; i < (hg->nverts); ++i) {
		hg->degree[i] = (INT_T)(cg->rowstarts[i+1]-cg->rowstarts[i]);
		hg->offset[2*i] = cg->rowstarts[i];
		hg->offset[2*i+1] = cg->rowstarts[i+1];
	}

//#pragma omp parallel for
	for (i = 0; i < hg->nedges; ++i)
	    hg->edges[i] = (INT_T)cg->column[i];

	// Check
	// check_csr_struct(cg);
	print_csr_graph(cg, fout, __func__);

	//free(g_offset); g_offset = NULL;
	return 0;
}


void free_csr_graph(csr_graph* g) {
   if (g->rowstarts != NULL) {
	   free(g->rowstarts);
	   g->rowstarts = NULL;
   }
   if (g->column != NULL) {
	   free(g->column);
	   g->column = NULL;
   }
}

int print_csr_graph(csr_graph* g, FILE *fp, const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if ((g == NULL) || (g->column == NULL) || (g->rowstarts == NULL)) {
		fprintf(stderr, "rank %d, %s: in %s NULL input\n", 
			rank, fcaller, __func__);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if ((g->nlocalverts <= 0) || (g->nlocaledges <= 0)) {
		fprintf(stderr, "%s; rank %d, in print_csr_graph:"
			" nverts or nedges <=0\n", fcaller, rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	fprintf(fp,"\n*** *** ** %s: CSR_GRAPH ** *** ***\n", fcaller);
	fprintf(fp, "Processor %d says nlocaledges=%zu\n", 
		rank, g->nlocaledges);
	fprintf(fp, "Processor %d says nlocalverts=%zu\n", 
		rank, g->nlocalverts);
	fprintf(fp, "Processor %d says nglobaverts=%"PRId64"\n", 
		rank, g->nglobalverts);
	if (g->nlocalverts < PRINT_MAX_NVERTS) {
		int64_t ii,jj;
		int64_t ei_s, ei_e;
		for (ii = 0; ii < (int64_t)g->nlocalverts; ++ii) {
			ei_s = g->rowstarts[ii];
			ei_e = g->rowstarts[ii+1];
	
			fprintf(fp, "%"PRId64"-> ", 
				((int64_t)((ii) * size + rank)));
			for (jj = ei_s; jj < ei_e; ++jj) {
				fprintf(fp, "%"PRId64", ", g->column[jj]);
			}
			fprintf(fp, "\n");
		}
	}
	fprintf(fp,"\n*** *** ** %s: CSR_GRAPH ** *** ***\n", fcaller);
	fflush(fp);
#endif
	return 0;
}

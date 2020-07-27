/* This is to make PRId64 working with c++ compiler */
#ifdef __cplusplus
#define __STDC_FORMAT_MACROS
#endif
/* header of int64_t and PRId64 */
#include <inttypes.h>

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <dlfcn.h>
#include <ctype.h>
#include <math.h>
#include <cuda_runtime.h>
#include "header.h"
#include "gputils.h"
#include "cputils.h"

extern int nthreads, maxblocks;
extern int rank, size;
extern FILE *fp_struct;

int stringCmp(const void *a, const void *b);

/* Error handling function */
//#define CHECK_ERROR_MODE
void checkCUDAError(const char *msg)
{
#if defined (CHECK_ERROR_MODE)
#ifdef CHECK_ERROR_MODE_2
	fprintf(stderr,"rank %d, debug mode, run checkCUDAerror.\n", rank);
#endif
	cudaThreadSynchronize();
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "rank %d, cuda error: %s: %s.\n", rank, msg, cudaGetErrorString(err));
		/* This is needed to avoid some rare crashes that generate deadlocks */
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
#ifdef CHECK_ERROR_MODE_2
	fprintf(stderr,"rank %d, checkCUDAerror: No error found.\n", rank);
#endif
#endif
}

void warm_gpu()
{
	float* warm;
	float ciccio = 5.0;
	cudaMalloc( (void**)&warm, sizeof(float) );
	cudaMemcpy(warm, &ciccio, sizeof(float), cudaMemcpyHostToDevice);
	cudaFree(warm);
}


int stringCmp(const void *a, const void *b)
{
	return strcmp((const char *)a,(const char *)b);
}

//static char host_name[MPI_MAX_PROCESSOR_NAME];

void assignDeviceToProcess()
{
	char	host_name[MPI_MAX_PROCESSOR_NAME];
	char	(*host_names)[MPI_MAX_PROCESSOR_NAME];
	MPI_Comm nodeComm;
	
	int n, namelen, color, rank, nprocs;
	int myrank, gpu_per_node;
	size_t bytes;
	int dev;
	struct cudaDeviceProp deviceProp;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Get_processor_name(host_name,&namelen);
	
	bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
	host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
	
	strcpy(host_names[rank], host_name);
	
	for (n=0; n<nprocs; n++)
	{
		MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD); 
	}
	
	qsort(host_names, nprocs,  sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);
	
	color = 0;
	
	for (n=0; n<nprocs; n++){
		if(n>0&&strcmp(host_names[n-1], host_names[n])) color++;
		if(strcmp(host_name, host_names[n]) == 0) break;
	}
	
	MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
	
	MPI_Comm_rank(nodeComm, &myrank);
	MPI_Comm_size(nodeComm, &gpu_per_node);

	if (getenv("MV2_COMM_WORLD_LOCAL_RANK")) {
		printf("MVAPICH2 detected, skip cudaSetDevice\n");
		return;
	}

	/* Find out how many DP capable GPUs are in the system and their 
	device number */
	int deviceCount,slot=0;
	int *devloc;
	cudaGetDeviceCount(&deviceCount);
	devloc=(int *)malloc(deviceCount*sizeof(int));
	devloc[0]=999;

	//fflush(stdout);
	//fprintf(stdout,"\n*** %s IN %s ***\n", __func__, "");
	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaGetDeviceProperties(&deviceProp, dev);
		if(deviceProp.major==2)	{
			devloc[slot]=dev;
			slot++;
		} else {
			fprintf(stdout, "rank %d, set_device, warning: major = %d\n", 
				rank, deviceProp.major);
			if (deviceProp.major > 10000)
				MPI_Abort(MPI_COMM_WORLD, 1);
			devloc[slot]=dev;
			slot++;
		}
	}
	
	fprintf(stdout, "rank %d Assigning device %d  to process on node %s \n", rank, devloc[myrank], host_name);
	
	/* Assign device to MPI process */
	cudaSetDevice(devloc[myrank]);
	
	 cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); 
	//fprintf(stderr, "Using max cache\n");
	//fflush(stdout);
	//fprintf(stdout,"*** %s EOF ***\n\n", __func__);
}


#define PRINT_DEVICE_PROPERTIES(device)\
  cudaGetDeviceProperties(&deviceProp, device);\
  fflush(stdout);\
  fprintf(stdout,"\nMy rank is %d The Device ID is %d\n",rank, device);\
  fprintf(stdout,"\nThe Properties of the Device with ID %d are\n",device);\
  fprintf(stdout,"\tDevice Name : %s",deviceProp.name);\
  fprintf(stdout,"\n\tDevice Global Memory : %zu",deviceProp.totalGlobalMem); \
  fprintf(stdout,"\n\tDevice Shared Size   : %zu",deviceProp.sharedMemPerBlock);\
  fprintf(stdout,"\n\tDevice registers per Block : %d",deviceProp.regsPerBlock);\
  fprintf(stdout,"\n\tDevice warpSize : %d",deviceProp.warpSize);\
  fprintf(stdout,"\n\tDevice maxThreadsPerBlock : %d",deviceProp.maxThreadsPerBlock);\
  fprintf(stdout,"\n\tDevice multiProcessorCount : %d",deviceProp.multiProcessorCount);\
  fprintf(stdout,"\n\tDevice computeMode : %d",deviceProp.computeMode);\
  fprintf(stdout,"\n\tDevice Major Revision Numbers : %d",deviceProp.major);\
  fprintf(stdout,"\n\tDevice Minor Revision Numbers : %d",deviceProp.minor);\
  fprintf(stdout,"\n\n");\


int checkMaxScale(int SCALE, FILE *fout, int coeff)
{
	double log_Max;
	size_t avail;
	size_t total;
	double used;
	double log_avail;
	double log_total;
	double log_used;
	double scale_max;
	double MaxMemoryUsed;
	double logsize; 
	// coeff is the number of array with nverts allcated
	MaxMemoryUsed = coeff * sizeof(INT_T);

	cudaMemGetInfo(&avail, &total);

	used = total - avail;
	log_used = log2(used);
	log_avail = log2((double)(avail));
	log_total = log2((double)(total));
	log_Max = log2(MaxMemoryUsed);
	scale_max = avail/MaxMemoryUsed;
	scale_max = log2(scale_max);

	fprintf(fout,"\n*** %s IN %s ***\n", __func__, "MAIN");
	fprintf(fout, "rank %d, Device memory total: 		2^%.1f\n", rank, log_total);
	fprintf(fout, "rank %d, Device memory available: 	2^%.1f\n", rank,  log_avail);
	fprintf(fout, "rank %d, Device memory already used: 	2^%.1f\n", rank, log_used);
	fprintf(fout, "rank %d, Maximum size to be used: 	2^(%d + %.1f)\n", rank, SCALE, log_Max);
	fprintf(fout, "rank %d, Maximum SCALE with one proc: 	%.1f\n", rank, scale_max);

	logsize = log2((double)(size));
	scale_max += logsize;
	fprintf(fout, "rank %d, Maximum SCALE with %d procs: 	%.1f\n", rank, size, scale_max);
	fflush(fout);
	fprintf(fout,"*** %s EOF ***\n\n", __func__);

	if (SCALE > (scale_max + 1)) {
		fprintf(stderr, "WARNING: Not enough memory to run this SCALE\n");
		//MPI_Abort(MPI_COMM_WORLD, 1);
	} else if (SCALE > scale_max) {
		fprintf(stderr, "WARNING: SCALE > scale_max\n");
	}
	return 0;
}

int printDeviceFreeMemory(FILE *fout)
{
	size_t avail;
	size_t total;
	double used;
	double log_avail;
	double log_total;
	double log_used;

	cudaMemGetInfo(&avail, &total);

	used = total - avail;
	log_used = log2(used);
	log_avail = log2((double)(avail));
	log_total = log2((double)(total));

	fprintf(fout,"rank %d, ** DEVICE MEMORY INFO **\n", rank);
	fprintf(fout, "rank %d, Device memory total: 		2^%.1f (%.3f GB)\n", rank, log_total, total/GIGABYTE);
	fprintf(fout, "rank %d, Device memory available: 	2^%.1f (%.3f GB)\n", rank, log_avail, avail/GIGABYTE);
	fprintf(fout, "rank %d, Device memory already used: 	2^%.1f (%.3f GB)\n", rank, log_used, used/GIGABYTE);
	fprintf(fout,"rank %d, ** EOF DEVICE MEMORY INFO **\n", rank);
	return 0;
}

// nelems is the number of elements to allocate before to call checkFreeMemory
int checkFreeMemory(INT_T nelems, FILE *fout, const char *fcaller)
{
	size_t avail;
	size_t total;
	double used;
	double log_avail;
	double log_total;
	double log_used;
	double elementSize; 
	double deviceMemSize; 
	double nsize; 

	cudaMemGetInfo(&avail, &total);

	used = total - avail;
	log_used = log2(used);
	log_avail = log2((double)(avail));
	log_total = log2((double)(total));

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	fprintf(fout,"\n*** %s IN %s ***\n", __func__, fcaller);
	fprintf(fout, "rank %d, Device memory total: 		2^%.1f (%.3f GB)\n", rank, log_total, total/GIGABYTE);
	fprintf(fout, "rank %d, Device memory available: 	2^%.1f (%.3f GB)\n", rank, log_avail, avail/GIGABYTE);
	fprintf(fout, "rank %d, Device memory already used: 	2^%.1f (%.3f GB)\n", rank, log_used, used/GIGABYTE);
	fprintf(fout,"*** %s EOF ***\n\n", __func__);
	fflush(fout);
#endif

	deviceMemSize = (double)avail;
	elementSize = (double)sizeof(INT_T);
	nsize = nelems * elementSize;

	if (deviceMemSize < nsize) {
		fprintf(stderr, "Not enough memory to malloc %"PRI64" elems\n", nelems);
		fprintf(stderr,"\n*** %s IN %s ***\n", __func__, fcaller);
		fprintf(stderr, "rank %d, Device memory total: 		2^%.1f (%.3f GB)\n", rank, log_total, total/GIGABYTE);
		fprintf(stderr, "rank %d, Device memory available: 	2^%.1f (%.3f GB)\n", rank, log_avail, avail/GIGABYTE);
		fprintf(stderr, "rank %d, Device memory already used: 	2^%.1f (%.3f GB)\n", rank, log_used, used/GIGABYTE);
		fprintf(stderr,"*** %s EOF ***\n\n", __func__);

		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return 0;
}


void print_device_array(INT_T *d_in, INT_T nelems, FILE* fout, const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if(fout == NULL){
	  fprintf(stderr, "rank %d, %s: in %s fout is NULL\n",
		  rank, fcaller, __func__);
	}

	if (nelems > PRINT_MAX_NEDGES){
		fprintf(fout, "Array in %s has %"PRI64" elements\n", fcaller, nelems);
		return;
	}

	INT_T* in = NULL;
	in = (INT_T*)callmalloc(nelems*sizeof(INT_T), "print_device_array: malloc in");
	cudaMemcpy(in, d_in, nelems*sizeof(INT_T), cudaMemcpyDeviceToHost);
	checkCUDAError("print_device_array: d_in->in");

	int j;
	fprintf(fout, "\n*** *** %s *** ***\n", fcaller);
	fprintf(fout, "Array %s has %"PRI64" elements\n[", fcaller, nelems);
	for (j=0; j < nelems; ++j){
		fprintf(fout, "%"PRI64" ", in[j]);
	}
	fprintf(fout, "]\n*** *** END %s *** ***\n", fcaller);
	fflush(fout);
	free(in);
#endif
}

void print_device_array32(INT32_T *d_in, INT_T nelems, FILE* fout, const char *fcaller)
{
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	if(fout == NULL){
	  fprintf(stderr, "rank %d, %s: in %s fout is NULL\n",
		  rank, fcaller, __func__);
	}

	if (nelems > PRINT_MAX_NEDGES){
		fprintf(fout, "Array in %s has %"PRI64" elements\n", fcaller, nelems);
		return;
	}

	INT32_T* in = NULL;
	in = (INT32_T*)callmalloc(nelems*sizeof(INT32_T), "print_device_array: malloc in");
	cudaMemcpy(in, d_in, nelems*sizeof(INT32_T), cudaMemcpyDeviceToHost);
	checkCUDAError("print_device_array: d_in->in");

	int j;
	fprintf(fout, "\n*** *** %s *** ***\n", fcaller);
	fprintf(fout, "Array %s has %"PRI64" elements\n[", fcaller, nelems);
	for (j=0; j < nelems; ++j){
		fprintf(fout, "%d ", in[j]);
	}
	fprintf(fout, "]\n*** *** END %s *** ***\n", fcaller);
	fflush(fout);
	free(in);
#endif
}

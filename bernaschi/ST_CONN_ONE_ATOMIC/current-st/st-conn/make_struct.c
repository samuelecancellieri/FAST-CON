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

#include "header.h"
#include "gputils.h"
#include "make_struct.h"
#include "make_struct_gpufunc.h"
#include "cputils.h"
#include "reference_common.h" // required by cputestfunc.h
#include "gputestfunc.h"
#include "cputestfunc.h"
#include "adj_func.h"
#include "mythrustlib.h"
#include "defines.h"

extern size_t freed_memory_size; 
extern size_t current_allocated_size;
#define REC_VT_TAG 10000

extern int nthreads, maxblocks;
extern int rank, size;
extern int64_t MaxLabel;
extern int64_t MaxGlobalLabel;
extern int global_scale;
extern FILE* fp_struct;

int run_make_struct(INT_T* h_edges, INT_T nedges, adjlist* dg)
{
	// Device Allocated arrays: 
	INT_T *d_edges 		= NULL;	// undirected edges list on Device
	INT_T *d_edges_appo 	= NULL;	// support array
	INT_T *d_count_per_proc = NULL;	// number of vertices per proc

	double make_struct_time_start = 0;
	double make_struct_time_end = 0;
	double make_struct_time = 0;

	printDeviceFreeMemory(stderr);

	// h_edges has 2*nedges elements, I will make it undirected 
	// so device array must have 2*(2*nedges) elements +2 elems 
	// for the split kernel
    cudaMalloc((void**)&d_edges, TENTRYPE(nedges)*sizeof(INT_T));
	checkCUDAError("run_make_struct: malloc d_edges");

	// Device array for make_send_buffer
    cudaMalloc((void**)&d_edges_appo, TENTRYPE(nedges)*sizeof(INT_T));
	checkCUDAError("run_make_struct: cudaMalloc d_edges_appo");
	cudaMalloc((void**)&d_count_per_proc, 2*size * sizeof(INT_T));
	checkCUDAError("run_make_struct: cudaMalloc d_count_per_proc");

	// Copy edges from host to device
	//cudaMemcpy(d_edges, h_edges, 2*nedges*sizeof(INT_T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges_appo, h_edges, 2*nedges*sizeof(INT_T), cudaMemcpyHostToDevice);  //REMOVECOPY
	checkCUDAError("run_make_struct: h_edges -> d_edges");


	// Host arrays
	INT_T *sendbuff = NULL;
	INT_T *recvbuff = NULL;
	INT_T send_size = -1;
	// When the SCALE is small the difference between edges lists of
	// different processors can be large. The recv_size (== send_size), 
	// must be large enough (And for such SCALE I don't care).
    if (global_scale <= MIN_GLOBAL_SCALE) {
        send_size = MF_RECV_SIZE_SMALL_SCALE *nedges;
	} else {
        send_size = MF_RECV_SIZE*nedges + nedges;       // The graph will be undirect + recv is unknown
	}
	sendbuff = (INT_T*)callmalloc((size_t)(send_size*sizeof(INT_T)), "run_make_struct: sendbuff");
	recvbuff = (INT_T*)callmalloc((size_t)(send_size*sizeof(INT_T)), "run_make_struct: recvbuff");

    current_allocated_size = (TENTRYPE(nedges)*2 + 2*size)*sizeof(INT_T);
    double G_current_allocated_size = current_allocated_size/(double)(1024*1024*1024);

        if (rank == 0) {
                fprintf(stdout,"\n"
                        "#######################################\n"
                        "run_make_struct:			\n"
                        "#######################################\n"
                        "d_edges: 4*nedges + 2\n"
                        "d_edges_appo: 4*nedges + 2\n"
                        "(d_count_per_proc: 2*number_of_tasks)\n"
                        "input nedges=%"PRI64"\n"
			            "current_allocated_size=%.3f GB\n"
                        "#######################################\n"
			            "\n", nedges, G_current_allocated_size);
	}
	fflush(stdout);

	make_struct_time_start = MPI_Wtime();
	make_struct(nedges, d_edges, d_edges_appo, d_count_per_proc, sendbuff, recvbuff, send_size, dg); 
	make_struct_time_end = MPI_Wtime();
	make_struct_time = make_struct_time_end - make_struct_time_start;
	PRINT_TIME("run_make_struct:make_struct_time", fp_struct, make_struct_time);
#ifndef DBG_LEVEL
	//fprintf(stdout, "TIME:MAIN:make_struct:%f\n", make_struct_time);
#endif

	free(sendbuff); sendbuff = NULL;
	free(recvbuff); recvbuff = NULL;
	cudaFree(d_count_per_proc); d_count_per_proc = NULL;

	return 0;
}

int make_struct(INT_T nedges, INT_T *d_edges, INT_T *d_edges_appo, INT_T *d_count_per_proc, INT_T *sendbuff, 
		        INT_T* recvbuff, INT_T send_size, adjlist *dg)
{

	CHECK_INPUT(sendbuff, send_size, __func__);
	CHECK_INPUT(recvbuff, send_size, __func__);

	INT_T send_count_per_proc[size+1];	// number of vertices to send per procs
	INT_T send_offset_per_proc[size+1];	// exclusive scan of send_count_per_proc
	INT_T recv_count_per_proc[size+1]; 	// number of vertices to recv from procs
	INT_T recv_offset_per_proc[size+1];	// exclusive scan of recv_count_per_proc
	INT_T recv_count_all_proc[size*size+1];	// sum of vertices I recv from other 

	INT_T my_own_verts = 0; 	// number of vertices I own (2 times the number of edges I own)
	INT_T non_own_verts = 0;	// number of verts I have to send
	INT_T recv_count = 0;		// number of vertices to receive
	INT_T myoff = 0; 		// start offset of owned edges in sendbuff

	memset(send_count_per_proc, 0, (size+1)*sizeof(INT_T));
	memset(send_offset_per_proc, 0, (size+1)*sizeof(INT_T));
	memset(recv_count_per_proc, 0, (size+1)*sizeof(INT_T));
	memset(recv_offset_per_proc, 0, (size+1)*sizeof(INT_T));
	memset(recv_count_all_proc, 0, (size*size+1)*sizeof(INT_T));

	// Make the graph undirected in place, d_edges is allocated 4*nedegs + 2
	INT_T und_nedges, nedges_to_send;
	// Here thrust::remove allocate extra memory, why???
	make_g_undirect(d_edges, d_edges_appo, nedges, &und_nedges); //REMOVECOPY
	//make_g_undirect(d_edges, nedges, &und_nedges);

	// d_edges_appo is the buffer to send
	if (size > 1) {
		make_send_buffer(d_edges, d_edges_appo, und_nedges, &nedges_to_send, send_count_per_proc, 
				send_offset_per_proc, d_count_per_proc);

		// Copy d_edges_appo to host 
		CHECK_SIZE("nedges_to_send", 2*nedges_to_send, "send_size", send_size, __func__);
		//assert(send_size >= 2*nedges_to_send);
		cudaMemcpy(sendbuff, d_edges_appo, 2*nedges_to_send*sizeof(INT_T), cudaMemcpyDeviceToHost);
		checkCUDAError("make_struct: d_edges_appo -> sendbuff");
		cudaFree(d_edges_appo); d_edges_appo = NULL;
		cudaFree(d_edges); d_edges = NULL;

		// Send vertices to send and keep my own vertices
		// Set useful variables
		myoff = send_offset_per_proc[rank];
		my_own_verts = send_count_per_proc[rank];
		non_own_verts = send_offset_per_proc[size] - my_own_verts; 
		// Set to zero I don't want to send/recv to me 
		send_count_per_proc[rank] = 0;
        // Gather the number of vertices to receive
		MPI_Allgather(send_count_per_proc, size, MPI_INT_T, recv_count_all_proc, size, MPI_INT_T, MPI_COMM_WORLD);
		// Count the total number of vertices to recv 
		int i;
		for (i = 0; i < size; ++i)
		{
			int p = rank + i*size;
			// Recv from p 
			recv_count_per_proc[i] = recv_count_all_proc[p];
			// Sum up the total number of vertices to recv 
			recv_count_per_proc[size] += recv_count_all_proc[p];
		} 

		recv_count = recv_count_per_proc[size];
		assert(recv_count_per_proc[rank] == 0);
		//assert(recv_count <= send_size);
		CHECK_SIZE("recv_count", recv_count, "send_size", send_size, __func__);
		// Compute offset for receiving buffer
		exclusive_scan(recv_count_per_proc, recv_offset_per_proc, size, size+1);
	
		// Post MPI Irecv, fill recvbuff
	    MPI_Request recv_req[size];
	    int senderc=0;
	    for (i = 0; i < size; ++i) {
	      if (recv_count_per_proc[i] > 0) {
	        MPI_Irecv((recvbuff + recv_offset_per_proc[i]),
			  recv_count_per_proc[i], MPI_INT_T,
	                  i, REC_VT_TAG+rank, MPI_COMM_WORLD, &recv_req[senderc]);
	        senderc++;
	      }
	    }

		// MPI Send
	    for (i = 0; i < size; ++i) {
	          if (send_count_per_proc[i] > 0) {
	            /* Send to proc i */
	            MPI_Send((sendbuff + send_offset_per_proc[i]),
	    		 send_count_per_proc[i], MPI_INT_T, i,
	                     REC_VT_TAG+i, MPI_COMM_WORLD);
	    	}
		}

        MPI_Waitall(senderc, recv_req, MPI_STATUSES_IGNORE);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
		checkGlobalEdgeList(recvbuff, recv_count/2, fp_struct, __func__);
		print_edges(recvbuff, recv_count/2, fp_struct, "MAKE_STRUCT: recvbuff");
		fflush(fp_struct);
#endif

		// put local edges at the end of recvbuff
		assert((send_size - recv_count - my_own_verts) > 0);
		memcpy((recvbuff+recv_count), (sendbuff+myoff), (size_t)(my_own_verts*sizeof(INT_T)));
	
		// update the number of local vertices
		my_own_verts += recv_count;

	}	//EOF if(size > 1)
		else if (size == 1) {
		my_own_verts = 2*und_nedges;
		CHECK_SIZE("my_own_verts", my_own_verts, "send_size", send_size, __func__);
		cudaMemcpy(recvbuff, d_edges, my_own_verts*sizeof(INT_T), cudaMemcpyDeviceToHost);
		checkCUDAError("make_struct: size==1, d_edges -> recvbuff");
		cudaFree(d_edges);
		cudaFree(d_edges_appo);
	}

    freed_memory_size =  2*TENTRYPE(nedges) * sizeof(INT_T);
	current_allocated_size = current_allocated_size - freed_memory_size;
	double G_current_allocated_size = current_allocated_size/(double)(1024*1024*1024);
	if (rank == 0) {
		fprintf(stdout,"\n"
                        "###############################################\n"
                        "Free after send in make_struct:		\n"
			"###############################################\n"
			"Free d_edges_appo and d_edges: 2*(4*nedges + 2)\n"
			"current_allocated_size=%.3f GB\n"
                        "#######################################\n"
                        "\n", G_current_allocated_size);
	}
	fflush(stdout);

	// Malloc + 2 for the split kernel
	INT_T *d_own_edges = NULL;
	cudaMalloc((void**)&d_own_edges, (my_own_verts + 4)*sizeof(INT_T));	
	checkCUDAError("MAKE_STRUCT: malloc d_own_edges");
	cudaMemcpy(d_own_edges, recvbuff, my_own_verts*sizeof(INT_T), cudaMemcpyHostToDevice);
	checkCUDAError("MAKE_STRUCT: recvbuff -> d_own_edges");

	current_allocated_size = current_allocated_size + (my_own_verts + 4)*sizeof(INT_T);
	G_current_allocated_size = current_allocated_size/(double)(1024*1024*1024);
	if (rank == 0) {
		fprintf(stdout,"\n"
                        "####################################\n"
                        "make_struct malloc for make_own_csr:\n"
                        "####################################\n"
			"d_own_edges: my_own_verts ~ 4*nedges\n"
			"my_own_verts=%"PRI64"\n"
			"current_allocated_size=%.3f GB\n"
                        "####################################\n"
                        "\n", my_own_verts, G_current_allocated_size);
	}
	fflush(stdout);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	fprintf(fp_struct, "\nSEND-RECV of generated EDGES is FINISHED!\n");
	fprintf(fp_struct, "generated verts = %"PRI64"\n", 2*nedges);
	fprintf(fp_struct, "send_size = %"PRI64"\n", send_size);
	fprintf(fp_struct, "received verts = %"PRI64"\n", recv_count);
	fprintf(fp_struct, "my_own_verts = %"PRI64"\n", my_own_verts - recv_count);
	fprintf(fp_struct, "\n\n");
	fprintf(fp_struct, "own verts before send-recv = %"PRI64"\n", 
					(my_own_verts - recv_count));
	fprintf(fp_struct, "sent verts = %"PRI64"\n", non_own_verts);
	fprintf(fp_struct, "own verts after send-recv = %"PRI64"\n", my_own_verts);
	fprintf(fp_struct, "exceeding verts = %"PRI64"\n", (-2*nedges + my_own_verts));
	fprintf(fp_struct, "percentage of own verts on input verts = %.2f\n", 
		((double)my_own_verts/(double)(2*nedges))*100.0);
	fprintf(fp_struct, "EOF MAKE_SEND_BUFF!\n\n");
	fflush(fp_struct);

	MPI_Barrier(MPI_COMM_WORLD);

#endif

	// Each processor build it's own csr on Device
	make_own_csr_nomulti(d_own_edges, my_own_verts, dg);

	return 0;
}

// Add undirect edges at the end of the edge list.
// For each (u,v) with u!=v add (v,u), if u==v add (-1,-1).
// Use thrust::remove to remove all the -1 values.
//int make_g_undirect(INT_T *d_edges, INT_T nedges, INT_T *undirect_nedges)
int make_g_undirect(INT_T *d_edges, INT_T *d_edges_appo, INT_T nedges, INT_T *undirect_nedges) //REMOVECOPY
{
	CHECK_SIZE("zero", 0, "nedges", nedges, __func__);

	INT_T compact_nelems = 0; // number of elems after the compact operation
	INT_T nelems = 4*nedges; // The total number of elems in the undirect edge list is 4*nedges
	int value = -1;

	// For each edge (u,v) if (u != v) add (v,u) else add (X,X)
	//add_edeges_vu(d_edges, nedges);
	add_edeges_vu(d_edges_appo, nedges); //REMOVECOPY

	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	//check_add_vu(d_edges, nedges, __func__, fp_struct);
	//check_add_undirect_edges(d_edges, nedges, __func__, fp_struct);
	check_add_vu(d_edges_appo, nedges, __func__, fp_struct);                 //REMOVECOPY
	check_add_undirect_edges(d_edges_appo, nedges, __func__, fp_struct);     //REMOVECOPY
#endif
	
	// Remove the edges (-1,-1) introduced by the above function
	// This function allocate extra memory!!!!!!!!
	double G_current_allocated_size = (current_allocated_size + 4*nedges*sizeof(INT_T))
					 /(double)(1024*1024*1024);
	if (rank == 0) {
		fprintf(stdout,"\n"
                        "####################################\n"
                        "make_g_undirect:\n"
                        "####################################\n"
			            "thrust::remove 4*nedges\n"
			            "nedges=%"PRI64"\n"
			            "current_allocated_size=%.3f GB\n"
                        "####################################\n"
			"\n",
			nedges, G_current_allocated_size);
	}
	// if (rank == 0) checkFreeMemory(nelems, stderr, __func__);                     //REMOVECOPY
	call_thrust_remove_copy(d_edges_appo, nelems, d_edges, &compact_nelems, value);  //REMOVECOPY
	//call_thrust_remove(d_edges, nelems, &compact_nelems, value);
	
	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	INT_T nloop = nelems - compact_nelems;
	check_compact_edge_list(d_edges, nelems, compact_nelems, __func__, fp_struct);
	print_device_array(d_edges, (compact_nelems+nloop)/2, fp_struct, "D_EDGES UNDIRECT U,V");
	print_device_array(d_edges+(compact_nelems+nloop)/2, (compact_nelems-nloop)/2, fp_struct, "D_EDGES UNDIRECT V,U");
#endif

	// Set the new number of edges for the undirect graph
	*undirect_nedges = compact_nelems/2;

	return 0;	
}

// The input edge list is undirect and has nedges edges. The number of
// elements is ~< 4*(RMAT-generated nedges) because in the previous functions
// the number of vertices is nearly doubled to make the graph undirect.
int make_send_buffer(INT_T *d_edges, INT_T *d_edges_appo, INT_T nedges, INT_T *nedges_to_send,
		     INT_T *send_count_per_proc, INT_T *send_offset_per_proc, 
		     INT_T *d_count_per_proc)
{	
	// Host array
	INT_T host_count_per_proc[2*size+1];
	memset(host_count_per_proc, 0, (2*size+1)*sizeof(INT_T));

	// Device Useful pointers
	INT_T* d_edges_u = NULL;
	INT_T* d_edges_v = NULL;
	INT_T* d_edges_appo_u = NULL;
	INT_T* d_edges_appo_v = NULL;

	INT_T nedges_to_split = 0;

	split_edges(d_edges, nedges, &nedges_to_split);
	d_edges_u = &d_edges[0];
	d_edges_v = &d_edges[nedges_to_split];

	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_split_edge_list(d_edges, nedges, nedges_to_split, __func__, fp_struct);
	print_device_array(d_edges, nedges_to_split, fp_struct, "MAKE_SEND_BUFFER: SPLITTED EDGE LIST U");
	print_device_array(d_edges+nedges_to_split, nedges_to_split, fp_struct, "MAKE_SEND_BUFFER: SPLITTED EDGE LIST V");
#endif

	// Fill d_edges_appo with owner of u and threads gid
	owners_edges(d_edges_u, nedges_to_split, d_edges_appo);
	d_edges_appo_u = &d_edges_appo[0];
	d_edges_appo_v = &d_edges_appo[nedges_to_split];
	
	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_owners(d_edges_appo, d_edges_appo+nedges_to_split, nedges_to_split, d_edges_u, __func__, fp_struct);
#endif

	current_allocated_size = current_allocated_size + 2*nedges_to_split*sizeof(INT_T);
	double G_current_allocated_size = current_allocated_size/(double)(1024*1024*1024);

        if (rank == 0) {
                fprintf(stdout,"\n"
                        "#######################################\n"
                        "make_send_buffer:			\n"
                        "#######################################\n"
			"thrust: ~2*nedges_to_split"
                        "nedges_to_split=%"PRI64"\n"
			"current_allocated_size=%.3f GB\n"
                        "#######################################\n"
			"\n", nedges, G_current_allocated_size);
	}

	// Sort d_edges_appo_u (owners) with d_edges_appo_v (gid) as payload 
	INT_T Pmax = -1;
	call_thrust_sort_by_key_and_max(d_edges_appo_u, d_edges_appo_v, nedges_to_split, &Pmax);

	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_sort(d_edges_appo_u, nedges_to_split, __func__, fp_struct);
	print_device_array(d_edges_appo_u, nedges_to_split, fp_struct, "MAKE_SEND_BUFFER: SORTED LIST P(U)");
	print_device_array(d_edges_appo_u+nedges_to_split, nedges_to_split, fp_struct, "MAKE_SEND_BUFFER: SORTED LIST GID");
	CHECK_SIZE("number of procs", size, "Max Proc", (Pmax+1), __func__);
	fprintf(fp_struct, "rank %d in %s, Pmax=%"PRI64"\n", rank, __func__, Pmax);
	fflush(fp_struct);
#endif

	// Count edges per proc 
	count_vertices(d_edges_appo, nedges_to_split, d_count_per_proc, send_count_per_proc, host_count_per_proc);
	exclusive_scan(send_count_per_proc, send_offset_per_proc, size, size+1);

	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_count(d_edges_appo, nedges_to_split, send_count_per_proc, __func__, fp_struct);
#endif

	// Put back vertices in d_edges with order from d_edges_appo_v
	back_vertices(d_edges_u, d_edges_v, nedges_to_split, d_edges_appo_u, d_edges_appo_v);

	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_back(d_edges_u, 2*nedges_to_split, __func__, fp_struct);
#endif

	// Unsplit edges in d_edges_appo
	unsplit_edges(d_edges_u, d_edges_v, nedges_to_split, d_edges_appo);

	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_unsplit(d_edges_appo, 2*nedges_to_split, __func__, fp_struct);
	fflush(fp_struct);
#endif

	// return the number of edges to send
	nedges_to_send[0] = nedges_to_split;
	
	return 0;

} // EOF make_send_buffer

// The input is the edge list composed only by local edges i.e. (u,v) u belong to me
int make_own_csr(INT_T *d_own_edges, INT_T my_own_verts, adjlist *dg)
{
	// Device Useful pointers
	INT_T* d_edges_u = NULL;
	INT_T* d_edges_v = NULL;
	
	// Host useful vars
	INT_T umax = 0;
	INT_T nverts = 0;

	INT_T my_nedges = my_own_verts/2;
	INT_T nedges_to_split = 0;
	
	split_edges(d_own_edges, my_nedges, &nedges_to_split);
	d_edges_u = &d_own_edges[0];
	d_edges_v = &d_own_edges[nedges_to_split];

	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_split_edge_list(d_own_edges, my_nedges, nedges_to_split, __func__, fp_struct);
	print_device_array(d_own_edges, nedges_to_split, fp_struct, "MAKE_OWN_CSR: SPLITTED EDGE LIST U");
	print_device_array(d_own_edges+nedges_to_split, nedges_to_split, fp_struct, "MAKE_OWN_CSR: SPLITTED EDGE LIST V");
#endif

	// Sort u and compute the maximum label
	call_thrust_sort_by_key_and_max(d_edges_u, d_edges_v, nedges_to_split, &umax);

	// I want to have an even number of vertices, if they are odd I add one vertex.
	// umax is the maximum label that I have found among my vertices.
	umax = VERTEX_LOCAL(umax);
	nverts = umax + 1; // nverts = maxlabel + 1
	nverts = ((nverts % 2) == 0) ? nverts : nverts + 1;	
	MaxLabel = nverts - 1;


	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_sort(d_own_edges, nedges_to_split, __func__, fp_struct);
	print_device_array(d_own_edges, nedges_to_split, fp_struct, "MAKE_OWN_CSR: SORTED LIST U");
	print_device_array(d_own_edges+nedges_to_split, nedges_to_split, fp_struct, "MAKE_OWN_CSR: SORTED LIST V");
	CHECK_SIZE("number of procs", size, "Max Label", umax, __func__);
	fprintf(fp_struct, "rank %d in %s, umax=%"PRI64"\n", rank, __func__, umax);
	fprintf(fp_struct, "rank %d in %s, nverts=%"PRI64"\n", rank, __func__, nverts);
	fprintf(fp_struct, "rank %d in %s, MaxLabel=%"PRI64"\n", rank, __func__, MaxLabel);
	fprintf(fp_struct, "rank %d in %s, nedges=%"PRI64"\n", rank, __func__, nedges_to_split);
	fflush(fp_struct);
#endif

	// Compute offset and degree of each u
	INT_T *d_count_u = NULL;
	INT_T *d_degree = NULL;

	cudaMalloc((void**)&d_count_u, 2*(nverts+1)*sizeof(INT_T));
	checkCUDAError("make_own_csr: malloc d_count_u");
	cudaMalloc((void**)&d_degree, nverts*sizeof(INT_T));
	checkCUDAError("make_own_csr: malloc d_degree");

	// Compute offset and degree of the csr
	make_offset(d_edges_u, nedges_to_split, d_count_u, nverts, d_degree);

	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_offset(d_count_u, 2*nverts, __func__, fp_struct);
	check_degree(d_degree, nverts, __func__, fp_struct);
	print_device_array(d_count_u, 2*nverts, fp_struct, "D_OFFSET ARRAY");
	print_device_array(d_degree, nverts, fp_struct, "D_DEGREE ARRAY");
#endif

	INT_T *d_adj_edges = NULL;
	cudaMalloc((void**)&d_adj_edges, nedges_to_split*sizeof(INT_T));
	checkCUDAError("make_own_csr: malloc d_adj_edges");
	cudaMemcpy(d_adj_edges, d_edges_v, nedges_to_split*sizeof(INT_T), cudaMemcpyDeviceToDevice);
	checkCUDAError("make_own_csr: d_own_edges -> d_adj_edgej");
	cudaFree(d_own_edges);
				
	dg->offset = d_count_u;
	dg->degree = d_degree;
	dg->edges = d_adj_edges;
	dg->nedges = nedges_to_split;
	dg->nverts = nverts;

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	adjlist hostgraph;
	copyAdjDeviceToHost(&hostgraph, dg);
	check_adjlist(&hostgraph, fp_struct, "make_own_csr");
	print_adjlist(&hostgraph, fp_struct, "make_own_csr");
	//print_adjlist_stat(&hostgraph, __func__, fp_struct);
	free_adjlist(&hostgraph);
#endif
	return 0;
}

int make_own_csr_nomulti(INT_T *d_own_edges, INT_T my_own_verts, adjlist *dg)
{
	// Device Useful pointers
	INT_T* d_edges_u = NULL;
	INT_T* d_edges_v = NULL;
	
	// Host useful vars
	INT_T umax = 0;
	INT_T nverts = 0;
	INT_T my_nedges = my_own_verts/2;
	INT_T nedges_to_split = 0;
	

	// Split edge list (u,v) in u and v 
	split_edges(d_own_edges, my_nedges, &nedges_to_split);
	d_edges_u = &d_own_edges[0];
	d_edges_v = &d_own_edges[nedges_to_split];
	
	// Sort v and make edges unique, keep multiplicity
	INT_T vmax = 0;
	INT_T compact_nedges = nedges_to_split;
	unsigned int *d_stencil = NULL;
	cudaMalloc((void**)&d_stencil, compact_nedges*sizeof(unsigned int));
	checkCUDAError("make_own_csr_nomulti: mallc d_stencil");

	//INT_T number_of_elems_to_remove;
	//call_thrust_count_unsigned(d_stencil, compact_nedges, 1, &number_of_elems_to_remove);
	//fprintf(stdout, "MAKE_OWN_CSR_NOMULTI, NUMBER OF ELEMS TO REMOVE: %"PRI64"\n", number_of_elems_to_remove);

	current_allocated_size = current_allocated_size + 2*compact_nedges*sizeof(INT_T) + 
				compact_nedges*sizeof(unsigned int);
	double G_current_allocated_size = current_allocated_size/(double)(1024*1024*1024);
	if (rank == 0) {
		fprintf(stdout,"\n"
                        "####################################\n"
                        "make_own_csr_nomulti:\n"
                        "####################################\n"
			"d_stencil: compact_nedges (32 bit)\n"
			"thrust memory: ~2*compact_nedges\n"
			"compact_nedges=%"PRI64"\n"
			"current_allocated_size=%.3f GB\n"
                        "####################################\n"
			"\n",
			compact_nedges, 
			G_current_allocated_size);
	}
	fflush(stdout);
	
	sort_unique_edges(d_edges_v, d_edges_u, &compact_nedges, d_stencil, &umax, &vmax);
	
	assert(compact_nedges <= nedges_to_split);
	assert(vmax <= MaxGlobalLabel);

	cudaFree(d_stencil);
	
	freed_memory_size =  compact_nedges*sizeof(unsigned int) + 2*compact_nedges*sizeof(INT_T);
	current_allocated_size = current_allocated_size - freed_memory_size;
	G_current_allocated_size = current_allocated_size/(double)(1024*1024*1024);
	if (rank == 0) {
		fprintf(stdout,"\n"
                        "##################################################\n"
                        "Free after sort_unique in make_own_csr_nomulti:   \n"
			"##################################################\n"
			"Free d_stencil and thrust: 2*compact_nedges + \n"
						   "compact_nedges (32 bit)\n"
			"current_allocated_size=%.3f GB\n"
                        "##################################################\n"
			"\n",
			G_current_allocated_size);
	}
	fflush(stdout);
	
	// I want to have an even number of vertices, if they are odd I add one vertex.
	// umax is the maximum label that I have found among my vertices.
	umax = VERTEX_LOCAL(umax);
	nverts = umax + 1; // nverts = maxlabel + 1
	nverts = ((nverts % 2) == 0) ? nverts : nverts + 1;	
	MaxLabel = nverts - 1;

	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_sort(d_edges_u, compact_nedges, __func__, fp_struct);
	//check_sort(d_edges_v, compact_nedges, __func__, fp_struct);
	print_device_array(d_edges_u, compact_nedges, fp_struct, "MAKE_OWN_CSR_NOMULTI: SORTED LIST U");
	print_device_array(d_edges_v, compact_nedges, fp_struct, "MAKE_OWN_CSR_NOMULTI: SORTED LIST V");
	CHECK_SIZE("number of procs", size, "Max Label", umax, __func__);
	fprintf(fp_struct, "rank %d in %s, umax=%"PRI64"\n", rank, __func__, umax);
	fprintf(fp_struct, "rank %d in %s, nverts=%"PRI64"\n", rank, __func__, nverts);
	fprintf(fp_struct, "rank %d in %s, MaxLabel=%"PRI64"\n", rank, __func__, MaxLabel);
	fprintf(fp_struct, "rank %d in %s, nedges=%"PRI64"\n", rank, __func__, nedges_to_split);
	fflush(fp_struct);
#endif

	// Compute offset and degree of each u
	INT_T *d_count_u = NULL;
	INT_T *d_degree = NULL;

	cudaMalloc((void**)&d_count_u, 2*(nverts+1)*sizeof(INT_T));
	checkCUDAError("make_own_csr: malloc d_count_u");
	cudaMalloc((void**)&d_degree, nverts*sizeof(INT_T));
	checkCUDAError("make_own_csr: malloc d_degree");
	
	current_allocated_size = current_allocated_size + 3*nverts* sizeof(INT_T); 
	G_current_allocated_size = current_allocated_size/(double)(1024*1024*1024);
	if (rank == 0) {
		fprintf(stdout,"\n"
                        "####################################\n"
                        "make_own_csr_nomulti:\n"
                        "####################################\n"
			"d_count_u=2*nverts\n"
			"d_degreee=nverts\n"
			"nverts=%"PRI64"\n"
			"current_allocated_size=%.3f GB\n"
                        "####################################\n"
			"\n",
			nverts,
			G_current_allocated_size);
	}
	fflush(stdout);

	// Compute offset and degree of the csr
	make_offset(d_edges_u, compact_nedges, d_count_u, nverts, d_degree);

	// Debug, check correctness of the above function
#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	check_offset(d_count_u, 2*nverts, __func__, fp_struct);
	check_degree(d_degree, nverts, __func__, fp_struct);
	print_device_array(d_count_u, 2*nverts, fp_struct, "D_OFFSET ARRAY");
	print_device_array(d_degree, nverts, fp_struct, "D_DEGREE ARRAY");
#endif

	INT_T *d_adj_edges = NULL;
	cudaMalloc((void**)&d_adj_edges, compact_nedges*sizeof(INT_T));
	checkCUDAError("make_own_csr: malloc d_adj_edges");
	cudaMemcpy(d_adj_edges, d_edges_v, compact_nedges*sizeof(INT_T), cudaMemcpyDeviceToDevice);
	checkCUDAError("make_own_csr: d_own_edges -> d_adj_edgej");
	current_allocated_size = current_allocated_size + compact_nedges*sizeof(INT_T);
	G_current_allocated_size = current_allocated_size/(double)(1024*1024*1024);
	if (rank == 0) {
		fprintf(stdout,"\n"
                        "####################################\n"
                        "make_own_csr_nomulti:\n"
                        "####################################\n"
			"d_adj_edges: compact_nedges\n"
			"compact_nedges=%"PRI64"\n"
			"current_allocated_size=%.3f GB\n"
                        "####################################\n"
			"\n",
			compact_nedges, 
			G_current_allocated_size);
	}
        cudaFree(d_own_edges);
	freed_memory_size =  my_own_verts*sizeof(INT_T);
	current_allocated_size = current_allocated_size - freed_memory_size;
	G_current_allocated_size = current_allocated_size/(double)(1024*1024*1024);
	if (rank == 0) {
		fprintf(stdout,"\n"
                        "########################################\n"
                        "Free in make_own_csr_nomulti:   	 \n"
			"########################################\n"
			"Free d_own_edges: my_own_verts ~4*nedges\n"
			"my_own_verts=%"PRI64"\n"
			"current_allocated_size=%.3f GB\n"
                        "########################################\n"
			"\n",
			my_own_verts, G_current_allocated_size);
	}
	fflush(stdout);

	dg->offset = d_count_u;
	dg->degree = d_degree;
	dg->edges = d_adj_edges;
	dg->nedges = compact_nedges;
	dg->nverts = nverts;

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	adjlist hostgraph;
	copyAdjDeviceToHost(&hostgraph, dg);
	check_adjlist(&hostgraph, fp_struct, "make_own_csr");
	print_adjlist(&hostgraph, fp_struct, "make_own_csr: GRAPH");
	//print_adjlist_stat(&hostgraph, __func__, fp_struct);
	free_adjlist(&hostgraph);
#endif

	return 0;
}

int make_bitmask (adjlist *h_graph, mask *h_bitmask)
{
	// Dummy variables to simplify call
	INT_T *hedges=h_graph->edges;
	INT_T nedges=h_graph->nedges;

	INT_T NUmax=0;
	INT_T *h_unique_edges;

	// Store original order of edges
	h_unique_edges = (INT_T*)callmalloc(nedges*sizeof(INT_T), "make_bitmask h_unique_edges");
	// Copy edges into h_unique_edges
	memcpy(h_unique_edges, hedges, nedges*sizeof(INT_T));

	//Sort unique h_unique_edges
	call_thrust_sort_unique_host(h_unique_edges, &NUmax, nedges);

#if defined (DBG_LEVEL) && (DBG_LEVEL > 0)
	print_array_64t(hedges, nedges, fp_struct, "make_bitmask: EDGES");
	print_array_64t(h_unique_edges, NUmax, fp_struct, "make_bitmask: Unique EDGES");
#endif

	h_bitmask->p_nelems=(INT32_T)nedges;
	h_bitmask->m_nelems=(INT32_T)NUmax;
	h_bitmask->unique_edges = h_unique_edges;

	return 0;
}

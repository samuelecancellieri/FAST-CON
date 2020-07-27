/* Copyright (C) 2010 Universita' degli Studi di Roma La Sapienza          */
/*                                                                         */
/* This is to make PRId64 working with c++ compiler */
#ifdef __cplusplus
#define __STDC_FORMAT_MACROS
#endif

#include <mpi.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include "reference_common.h"
#include "header.h"
#include "cputils.h"
#include "load_graph.h"

void load_graph(char* fname_graph, int64_t* nedges_ptr, int64_t** result_ptr, int64_t* gnverts)
{
	FILE *fp_graph = NULL;

	char line[80], token[20];
	const char* fromNodeIdStr = "FromNodeId";
	const char* nodesStr = "Nodes:";

	unsigned long nodes, nedges;

	int64_t u,v;

	fp_graph = fopen(fname_graph, "rt");
	if (fp_graph == NULL){
		//fprintf(stderr,"Error opening file %s", fname_graph);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	fseek(fp_graph,0,SEEK_END);
	int64_t fsize = ftell(fp_graph);
	rewind(fp_graph);

	//fprintf(stdout, "[rank %d] File Size = %lu\n", rank, fsize);

	while(fgets(line, 80, fp_graph) != NULL)
	{
		/* get a line, up to 80 chars from fr.  done if NULL */
		sscanf (line, "# %s", token);
		//fprintf(stdout, "[rank %d] %s", rank, line);

		if (strcmp(token, fromNodeIdStr)==0) {
			break;
		} else if (strcmp(token,nodesStr)==0) {
			sscanf (line, "# Nodes: %lu Edges: %lu ", &nodes, &nedges);
			//fprintf(stdout, "[rank %d] NODES=%lu - EDGES=%lu\n", rank, nodes, nedges);
		}
	}

	unsigned long headerSize = ftell(fp_graph);

	//fprintf(stdout, "[rank %d] headerSize=%lu\n", rank, headerSize);

	fsize = fsize - headerSize;

	//fprintf(stdout, "[rank %d] fsize=%lu\n", rank, fsize);
	int factor = 2;
        if (nedges > 200000000) factor = 1;
	int64_t* local_edges = (int64_t*)callmalloc(factor*nedges*sizeof(int64_t), "load_graph");
	int64_t curr_index = 0;

	int64_t part_size = fsize / size;

	//fprintf(stdout, "[rank %d] part_size=%lu\n", rank, part_size);

	long int offset = (part_size * rank) + headerSize;
    long int curr_pos = 0;

	fseek(fp_graph, offset-1, SEEK_SET);

	//fprintf(stdout, "[rank %d] offset=%lu\n", rank, offset);

	int fc = fgetc(fp_graph);

	//fprintf(stdout, "[rank %d] fc=%d\n", rank, fc);

	if (fc != 10) {
		fgets(line, 80, fp_graph);
		curr_pos = strlen(line);
	}

	while ((fgets(line, 80, fp_graph) != NULL) && (curr_pos < part_size)) {
		curr_pos += strlen(line);
		sscanf (line, "%lu %lu ", &u, &v);
		local_edges[curr_index] = u;
		local_edges[curr_index+1] = v;
		curr_index += 2;
	}

	*result_ptr = local_edges;
	*nedges_ptr = curr_index/2;
    *gnverts = nodes;

	//fprintf(stdout, "[rank %d] nedges=%lu; gnverts=%lu\n", rank, *nedges_ptr, *gnverts);

}

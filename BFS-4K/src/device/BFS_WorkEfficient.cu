#pragma once

long long int totalFrontierNodes;
long long int totalFrontierEdges;

void cudaGraph::cudaBFS4K_N(int nof_tests) {
	srand(time(NULL));
	if (DUPLICATE_REMOVE)
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	long long int totalVisitedEdges = 0, totalVisitedNodes = 0, diameter = 0;
	totalFrontierNodes = 0, totalFrontierEdges = 0;

	int* hostDistance = new int[V];
	Timer<DEVICE> TM;
	double totalTime = 0;

	if(ATOMICCAS)
		std::cout<<"atomic comparison enabled"<<std::endl;
	else
		std::cout<<"atomic comparison disabled"<<std::endl;


	for (int i = 0; i < N_OF_TESTS; i++) {

		const int source = N_OF_TESTS == 1 ? 0 : mt::randf_co() * V;
		if (i == 0 || CHECK_TRAVERSED_EDGES || N_OF_TESTS == 1) {
			graph.BfsInit(source, hostDistance);
			graph.bfs();
			if (graph.V >= 10000 && graph.visitedEdges() < 10000)
				{ i--;	continue; }
		}
		totalVisitedNodes += graph.visitedNodes();
		totalVisitedEdges += graph.visitedEdges();
		diameter += graph.getMaxDistance();

		this->Reset(&source);
		TM.start();

		this->cudaBFS4K();

		TM.stop();
		cudaError("BFS Kernel N");
		float time = TM.duration();
		totalTime += time;

		if (nof_tests > 1)
			std::cout 	<< "iter: " << std::left << std::setw(10) << i << "time: " << std::setw(10) << time << "Edges: "
						<< std::setw(10) << graph.visitedEdges() << "source: " << source << std::endl;
		else
			cudaUtil::Compare(hostDistance, devDistance, V, "Distance Check", 1);
	}

	std::cout	<< std::setprecision(1) << std::fixed << std::endl
				<< "\t    Number of TESTS:  " << nof_tests << std::endl
				<< "\t          Avg. Time:  " << totalTime / nof_tests << " ms" << std::endl
				<< "\t         Avg. MTEPS:  " << totalVisitedEdges / (totalTime * 1000) << std::endl
				<< "\t      Avg. Diameter:  " << diameter / nof_tests << std::endl << std::setprecision(2)
				<< "\tAvg. Enqueued Nodes:  " << totalFrontierNodes / nof_tests << "\t\t( +" << 1 + (double) (totalFrontierNodes - totalVisitedNodes) / totalVisitedNodes << "x )" << std::endl << std::endl;

	if (COUNT_DUP && nof_tests == 1) {
		int duplicates;
		cudaMemcpyFromSymbol(&duplicates, duplicateCounter, sizeof(int));
		std::cout	<< "\t     Duplicates:  " << duplicates << std::endl << std::endl;
	}
}


//#define fun(a)		BFS_KernelMainGLOB	<BLOCKDIM, (a), false, DUPLICATE_REMOVE>\
//								<<<DIV(FrontierSize, (BLOCKDIM / (a)) * ITEM_PER_WARP), BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>\
//								(devOutNodes, devOutEdges, devDistance, devF1, devF2, FrontierSize, level);

#define fun(a)		BFS_KernelMainGLOB	<BLOCKDIM, (a), false, DUPLICATE_REMOVE>\
						<<<gridDim, BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>\
						(devOutNodes, devOutEdges, devDistance, devF1, devF2, FrontierSize, level);

#define funB(a)		BFS_KernelMainGLOBB	<BLOCKDIM, (a), false, DUPLICATE_REMOVE>\
						<<<gridDim, BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>\
						(devOutNodes, devOutEdges, devDistance, devF1, devF2, FrontierSize, level);

inline void cudaGraph::cudaBFS4K() {
	if (INTER_BLOCK_SYNC) {
        int num_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks,
                                                      BFSDispath<256, DUPLICATE_REMOVE>,
                                                      256,  SMem_Per_Block(256));

		BFSDispath<256, DUPLICATE_REMOVE>
            <<<num_blocks, 256, SMem_Per_Block(256)>>>
				(devOutNodes, devOutEdges, devDistance, devF1, devF2);
	}
	else {
		int SizeArray[4];
		int level = 1, FrontierSize = 1;
		do {
			FrontierDebug(FrontierSize, level, PRINT_FRONTIER);
			int size = logValueHost<MIN_VW, MAX_VW>(FrontierSize);

			const int DynBlocks = DYNAMIC_PARALLELISM ? RESERVED_BLOCKS : 0;
			const int gridDim = min(MAX_CONCURR_TH/BLOCKDIM - DynBlocks, DIV(FrontierSize, BLOCKDIM));
			//const int gridDim = DIV(FrontierSize, BLOCKDIM * (BLOCKDIM * ITEM_PER_WARP));

			if (size >= 3 && FrontierSize > MAX_CONCURR_TH) {
				def_SWITCHB(size);
			} else {
				def_SWITCH(size);
			}

			cudaMemcpyFromSymbolAsync(SizeArray, devF2Size, sizeof(int) * 4);
			FrontierSize = SizeArray[level & 3];
			if (FrontierSize > this->allocFrontierSize)
				error("BFS Frontier too large. Required more GPU memory. N. of Vertices/Edges in frontier: " << FrontierSize << " >  allocated: " << this->allocFrontierSize);

			std::swap<int*>(devF1, devF2);
			level++;
		} while ( FrontierSize );
	}
}

#undef fun

void cudaGraph::Reset(const int Sources[], int nof_sources) {
	cudaMemcpy(devF1, Sources, nof_sources * sizeof(int), cudaMemcpyHostToDevice);

	cudaUtil::fillKernel<dist_t><<<DIV(V, 128), 128>>>(devDistance, V, INF );
	cudaUtil::scatterKernel<dist_t><<<DIV(nof_sources, 128), 128>>>(devF1, nof_sources, devDistance, 0);

	int SizeArray[4] = {0, 0, 0, 0};
	cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);

	GReset<<<1, 256>>>();
	cudaError("Graph Reset");
}


// ---------------------- AUXILARY FUNCTION ---------------------------------------------

inline void cudaGraph::FrontierDebug(int FrontierSize, int level, bool PRINT_F) {
	totalFrontierNodes += FrontierSize;
	if (PRINT_F == 0)
		return;
	std::stringstream ss;
	ss << "Level: " << level << "\tF2Size: " << FrontierSize << std::endl;
	if (PRINT_F == 2)
		printExt::printCudaArray(devF1, FrontierSize, ss.str());
}

template<int MIN_VALUE, int MAX_VALUE>
inline int cudaGraph::logValueHost(int Value) {
	int logSize = 31 - __builtin_clz(MAX_CONCURR_TH / Value);
	if (logSize < _Log2<MIN_VALUE>::VALUE)
		logSize = _Log2<MIN_VALUE>::VALUE;
	if (logSize > _Log2<MAX_VALUE>::VALUE)
		logSize = _Log2<MAX_VALUE>::VALUE;
	return logSize;
}

		/*if (BLOCK_BFS && FrontierSize < BLOCK_FRONTIER_LIMIT) {
			BFS_BlockKernel<DUPLICATE_REMOVE><<<1, 1024, 49152>>>(devNodes, devEdges, devDistance, devF1, devF2, FrontierSize);
			cudaMemcpyFromSymbolAsync(&level, devLevel, sizeof(int));
		} else {*/

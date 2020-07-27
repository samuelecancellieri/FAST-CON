#pragma once

long long int totalFrontierNodes;
long long int totalFrontierEdges;
double totalAvgFrontierIncrease;
int numSources, visitedNodes, lastFrontierSize, avgFrontierSize;

void cudaGraph::cudaBFS4K_N(int N_OF_TESTS)
{
	srand(time(NULL));	//random
	if (DUPLICATE_REMOVE) //configuro sharedmemory per duplicate removal
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	long long int totalVisitedEdges = 0, totalFrontierSize = 0, diameter = 0;
	totalFrontierNodes = 0, totalFrontierEdges = 0; //inizializzazione variabili per statistiche BFS

	int *hostDistance = new int[V]; //distanza percorsa sui nodi
	Timer<DEVICE> TM;				//timer
	double totalTime = 0;
	totalAvgFrontierIncrease = 0;
	//std::cout << "Insert number of random source to use as BFS start: (only int values)" << std::endl;
	//std::cin >> numSources;
	numSources = NUM_SOURCES;
	int foundST = 0;

	for (int i = 0; i < N_OF_TESTS; i++)
	{
		int *source = new int[numSources]();

		for (int ff = 0; ff < numSources; ++ff)
		{
			source[ff] = N_OF_TESTS == 1 ? 0 : mt::randf_co() * V;
		}

		// totalFrontierSize += graph.visitedNodes(); //variabili aggiornate per statistiche
		totalVisitedEdges += graph.visitedEdges();
		diameter += graph.getMaxDistance();
		avgFrontierSize = 0;

		this->Reset(source, NUM_SOURCES); //reset cuda graph con inserimento del source nel frontier

		TM.start();		   //start time
		this->cudaBFS4K(); //start BFS4K
		TM.stop();		   //stop timer
		cudaError("BFS Kernel N");

		float time = TM.duration();
		totalTime += time;

		int hostFoundCommon = 0;
		cudaMemcpyFromSymbol(&hostFoundCommon, found_common, sizeof(int));

		if (hostFoundCommon)
		{
			printf("FOUND CONNECTION S-T\n");
			foundST++;
		}

		totalFrontierSize += lastFrontierSize;

		printf("visited nodes %d\n", visitedNodes);

		if (N_OF_TESTS > 1) //stampo stats
		{
			std::cout << "iter: " << std::left << std::setw(10) << i + 1 << "time: " << std::setw(10) << time << "Edges: "
					  << std::setw(10) << graph.visitedEdges() << "Source: " << source[0] << " Target " << source[1] << std::endl;
			//std::cout << ss.str() << std::endl;
			std::cout << "-----------------------------------------------------------------------------" << std::endl;
		}
		else
		{
			cudaUtil::Compare(hostDistance, devDistance, V, "Distance Check", 1);
		}
	}

	std::cout << std::setprecision(2) << std::fixed << std::endl //stats di percorrenza BFS
			  << "\t    Number of SOURCES:  " << NUM_SOURCES << std::endl
			  << "\t    Number of TESTS:  " << N_OF_TESTS << std::endl
			  << "\t    Number of ST-FOUND:  " << foundST << std::endl
			  << "\t          Avg. Time:  " << totalTime / N_OF_TESTS << " ms" << std::endl
			  << "\t         Avg. MTEPS:  " << totalVisitedEdges / (totalTime * 1000) << std::endl
			  << "\t      Avg. Diameter:  " << diameter / N_OF_TESTS << std::endl
			  << "\t      Avg. FrontierSize:  " << totalFrontierSize / N_OF_TESTS << std::endl
			  << "\t      Avg. FrontierIncrease:  " << totalAvgFrontierIncrease / N_OF_TESTS << std::endl
			  << std::setprecision(2)
			  //   << "\tAvg. Enqueued Nodes:  " << totalFrontierNodes / N_OF_TESTS << "\t\t( +" << 1 + (double)(totalFrontierNodes - totalFrontierSize) / totalFrontierSize << "x )" << std::endl
			  << std::endl;

	if (COUNT_DUP && N_OF_TESTS == 1)
	{
		int duplicates;
		cudaMemcpyFromSymbol(&duplicates, duplicateCounter, sizeof(int));
		std::cout << "\t     Duplicates:  " << duplicates << std::endl
				  << std::endl;
	}

	cudaDeviceReset();
}

//#define fun(a)		BFS_KernelMainGLOB	<BLOCKDIM, (a), false, DUPLICATE_REMOVE>\
//								<<<DIV(FrontierSize, (BLOCKDIM / (a)) * ITEM_PER_WARP), BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>\
//								(devOutNodes, devOutEdges, devDistance, devF1, devF2, FrontierSize, level);

#define fun(a) BFS_KernelMainGLOB<BLOCKDIM, (a), false, DUPLICATE_REMOVE> \
	<<<gridDim, BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>(devOutNodes, devOutEdges, devDistance, devF1, devF2, FrontierSize, level);

#define funB(a) BFS_KernelMainGLOBB<BLOCKDIM, (a), false, DUPLICATE_REMOVE> \
	<<<gridDim, BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>(devOutNodes, devOutEdges, devDistance, devF1, devF2, FrontierSize, level);

inline void cudaGraph::cudaBFS4K() //bfs4K primo avvio
{
	int SizeArray[4];
	int level = 1, FrontierSize = numSources;

	while (FrontierSize)
	{
		int size = logValueHost<MIN_VW, MAX_VW>(FrontierSize);
		const int gridDim = (MAX_CONCURR_TH / BLOCKDIM);

		// std::cout << "FrontierSize: " << FrontierSize << " level: " << level << std::endl;
		// lastFrontierSize = FrontierSize;

		// visitedNodes += (MAX_CONCURR_TH / pow(2, size)) < FrontierSize ? (MAX_CONCURR_TH / pow(2, size)) : FrontierSize;

		def_SWITCH(size);

		cudaMemcpyFromSymbolAsync(SizeArray, devF2Size, sizeof(int) * 4);
		FrontierSize = SizeArray[level & 3];

		if (FrontierSize > this->allocFrontierSize)
			error("BFS Frontier too large. Required more GPU memory. N. of Vertices/Edges in frontier: " << FrontierSize << " >  allocated: " << this->allocFrontierSize);

		std::swap<int *>(devF1, devF2);
		level++;

		// if (FrontierSize)
		// 	avgFrontierSize += FrontierSize - lastFrontierSize;
	}
	// totalAvgFrontierIncrease += (double)avgFrontierSize / (level - 1);
	// std::cout << "avg frontier size per level: " << (double)avgFrontierSize / (level - 1) << std::endl;
}

#undef fun

void cudaGraph::Reset(int Sources[], int nof_sources)
{

	visitedNodes = 0;
	cudaMemcpy(devF1, Sources, nof_sources * sizeof(int), cudaMemcpyHostToDevice);

	const int ZERO = 0;
	cudaMemcpyToSymbol(found_common, &ZERO, sizeof(int));

	cudaUtil::fillKernel<dist_t><<<DIV(V, 128), 128>>>(devDistance, V, INF);
	cudaUtil::scatterKernel<dist_t><<<DIV(nof_sources, 128), 128>>>(devF1, nof_sources, devDistance, 0); //cudaUtil.cuh

	int SizeArray[4] = {0, 0, 0, 0};
	cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);

	GReset<<<1, 256>>>();
	cudaError("Graph Reset");
}

// ---------------------- AUXILARY FUNCTION ---------------------------------------------

inline void cudaGraph::FrontierDebug(int FrontierSize, int level, bool PRINT_F)
{
	totalFrontierNodes += FrontierSize;
}

template <int MIN_VALUE, int MAX_VALUE>
inline int cudaGraph::logValueHost(int Value)
{
	int logSize = 31 - __builtin_clz(MAX_CONCURR_TH / Value); //least significant 0, ultimo 0 prima del primo 1 meno significativo
	if (logSize < _Log2<MIN_VALUE>::VALUE)
	{
		logSize = _Log2<MIN_VALUE>::VALUE;
	}
	if (logSize > _Log2<MAX_VALUE>::VALUE)
	{
		logSize = _Log2<MAX_VALUE>::VALUE;
	}
	return logSize;
}

/*if (BLOCK_BFS && FrontierSize < BLOCK_FRONTIER_LIMIT) {
			BFS_BlockKernel<DUPLICATE_REMOVE><<<1, 1024, 49152>>>(devNodes, devEdges, devDistance, devF1, devF2, FrontierSize);
			cudaMemcpyFromSymbolAsync(&level, devLevel, sizeof(int));
		} else {*/

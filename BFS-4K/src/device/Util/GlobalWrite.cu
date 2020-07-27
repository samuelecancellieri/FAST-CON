#pragma once


#include "definition.cuh"
#include "../Util/prefixSumAsm.cu"

extern __shared__ unsigned char SMem[];

namespace FrontierWrite {

	enum WriteMode { SIMPLE, SHARED_WARP, SHARED_BLOCK };

	__device__ __forceinline__ void FrontierReserve_Warp(int* GlobalCounter, int founds, int& n, int &totalWarp, int& globalOffset) {
		n = founds;
		totalWarp = warpExclusiveScan<32>(n);
		int oldCounter;
		if (LaneID() == 0)	// && totale != 0)
			oldCounter = atomicAdd(GlobalCounter, totalWarp);

		globalOffset = __shfl(oldCounter, 0);
	}

	template<int BlockDim>
	__device__ __forceinline__ void FrontierReserve_Block(int* GlobalCounter, int founds, int& n, int &totalBlock, int& globalOffset) {
		int* SM = (int*) SMem;
		n = founds;
		const int warpId = WarpID();
		SM[warpId] = warpExclusiveScan<32>(n);

		__syncthreads();
		if (Tid < BlockDim / 32) {
			int sum = SM[Tid];
			const int total = warpExclusiveScan<BlockDim / 32>(sum);

			if (Tid == 0) {
				SM[32] = total;
				SM[33] = atomicAdd(GlobalCounter, total);
			}
			SM[Tid] = sum;
		}
		__syncthreads();

		n += SM[warpId];
		totalBlock = SM[32];
		globalOffset = SM[33];
	}


	template<int BlockDim, WriteMode mode>
	__device__ __forceinline__ void Write(int* devFrontier, int* GlobalCounter, int* Queue, int founds) {
		int n, total, globalOffset;

		if (mode == SIMPLE || mode == SHARED_WARP)
			FrontierReserve_Warp(GlobalCounter, founds, n, total, globalOffset);

		if (mode == SIMPLE) {
			const int pos = globalOffset + n;
			for (int i = 0; i < founds; i++)
				devFrontier[pos + i] = Queue[i];
		}
		else if (mode == SHARED_WARP) {
			int* SM = (int*) &SMem[ WarpID() * SMem_Per_Warp ];

			int j = 0;
			while (total > 0) {
				while (j < founds && n + j < IntSMem_Per_Warp) {
					SM[n + j] = Queue[j];
					j++;
				}
				#pragma unroll
				for (int i = 0; i < IntSMem_Per_Thread; ++i) {
					const int index = LaneID() + i * 32;
					if (index < total)
						devFrontier[globalOffset + index] = SM[index];
				}
				n -= IntSMem_Per_Warp;
				total -= IntSMem_Per_Warp;
				globalOffset += IntSMem_Per_Warp;
			}
		}

		else if (mode == SHARED_BLOCK) {
			FrontierReserve_Block<BlockDim>(GlobalCounter, founds, n, total, globalOffset);
			int* SM = (int*) SMem;
			int j = 0;
			while (total > 0) {
				__syncthreads();
				while (j < founds && n + j < IntSMem_Per_Block(BlockDim) ) {
					SM[n + j] = Queue[j];
					j++;
				}
				__syncthreads();

				#pragma unroll
				for (int i = 0; i < IntSMem_Per_Thread; ++i) {
					const int index = Tid + i * BlockDim;
					if (index < total)
						devFrontier[globalOffset + index] = SM[index];
				}
				n -= IntSMem_Per_Block(BlockDim);
				total -= IntSMem_Per_Block(BlockDim);
				globalOffset += IntSMem_Per_Block(BlockDim);
			}
		}
	}
}

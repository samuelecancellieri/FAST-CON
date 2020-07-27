DEB=-DNDEBUG

# Arch Type
MYARCH = -D _USE_64_BIT_
ARCHFLAG = -arch=sm_50

THRUST_DIR="../thrust"
MPI_HOME=/usr
CUDADIR=/usr/local/cuda
MPI_INCLUDE=/usr/include/mpi
######## C, C++ VARIABLES #########
MPICC = $(MPI_HOME)/bin/mpic++ -g -Wall -O0
LIB = -L$(MPI_HOME)/lib
INCLUDE = -I./include -I./test -I$(MPI_INCLUDE)

CCLFLAGS = -lmpi $(LIB)
CCFLAGS = -c -m64 -Wall -std=c++11 $(INCLUDE)
CCFLAGS += -fpermissive $(MYARCH) $(DEB)  -D NET_MPI

######## CUDA VARIABLES #############
NVCC = $(CUDADIR)/bin/nvcc -O3
CUDAINCLUDE = -I$(CUDADIR)/include/ $(INCLUDE) -I$(THRUST_DIR) 
CUDALIB = -L$(CUDADIR)/lib64  $(LIB)
NVCCFLAGS = $(ARCHFLAG) -c -m64 --compiler-options -Wall $(CUDAINCLUDE) #-pg -g -G
NVCCLFLAGS = $(CUDALIB) -lm -lcudart -m64  
CUDAFLAGS += $(DEBGPU) -I$(CUDADIR)/include/  

####### GRAPH-500 REFERENCE #######
RCC =  $(MPI_HOME)/bin/mpicc -c
RCFLAGS = -g -O0 -m64 $(INCLUDE)
RCFLAGS += $(MYARCH)
RLDLIBS = $(CCLFLAGS) -lm -lrt

####### TARGET ########
BIN = st-con_mpi_cuda
CPUOBJ = main.o cputils.o gputils.o make_struct.o make_stcon.o make_bfs.o adj_func.o load_graph.o
CPUSRC = main.c cputils.c gputils.c make_struct.c make_stcon.c make_bfs.c adj_func.c load_graph.c
GENLIB = ../generator/libgraph_generator_mpi.a
KDIR = ./cudakernel
GPUSRC = gstruct_kernel.cu make_struct_gpufunc.cu bfs_kernel.cu stcon_kernel.cu make_bfs_func.cu make_stcon_func.cu mythrustlib.cu
GPUOBJ = make_struct_gpufunc.o make_bfs_func.o make_stcon_func.o mythrustlib.o
GPUKERNELOBJ = gstruct_kernel.o bfs_kernel.o stcon_kernel.o
TEST = YES
ifeq ($(TEST), YES)
TESTSRC = cputestfunc.c gputestfunc.cu
TESTOBJ = cputestfunc.o gputestfunc.o
else
TESTSRC = 
TESTOBJ = 
endif
CPUREFSRC = reference_functions.c
CPUREFOBJ = reference_functions.o

all: $(BIN) $(GENLIB)

$(BIN): $(CPUOBJ) $(GPUOBJ) $(GPUKERNELOBJ) $(GENLIB) $(TESTOBJ) $(CPUREFOBJ)
	 $(MPICC) -o $(BIN) $(CPUOBJ) $(GPUOBJ) $(GENLIB) $(TESTOBJ) $(CPUREFOBJ) $(CCLFLAGS) $(NVCCLFLAGS) $(GPUKERNELOBJ)
$(GENLIB):
	$(MAKE) -C ../generator -f ../generator/Makefile.PLX.mpi

###### CPU OBJ ###############
main.o: $(CPUSRC) include/header.h
	$(MPICC) $(CCFLAGS) $(CUDAFLAGS) main.c

cputils.o: cputils.c include/cputils.h include/header.h
	$(MPICC) $(CCFLAGS) $(CUDAFLAGS) cputils.c

adj_func.o: adj_func.c include/adj_func.h include/header.h
	$(MPICC) $(CCFLAGS) $(CUDAFLAGS) adj_func.c

load_graph.o: load_graph.c load_graph.h include/header.h
	$(MPICC) $(CCFLAGS) $(CUDAFLAGS) load_graph.c	
	
make_struct.o: make_struct.c include/make_struct.h include/header.h
	$(MPICC) $(CCFLAGS) $(CUDAFLAGS) make_struct.c
make_struct_host.o: make_struct.c include/make_struct_host.h include/header.h
	$(MPICC) $(CCFLAGS) $(CUDAFLAGS) make_struct_host.c	

make_bfs.o: make_bfs.c reference_functions.c include/make_bfs.h include/header.h
	$(MPICC) $(CCFLAGS) $(CUDAFLAGS) make_bfs.c
	
make_stcon.o: make_stcon.c reference_functions.c include/make_stcon.h include/header.h
	$(MPICC) $(CCFLAGS) $(CUDAFLAGS) make_stcon.c

gputils.o: gputils.c include/gputils.h include/header.h
	$(MPICC) $(CCFLAGS) $(CUDAFLAGS) gputils.c 

###### CPU REF OBJ ###########
reference_functions.o: reference_functions.c include/reference_common.h
	$(RCC) $(RCFLAGS) reference_functions.c

###### CPU TEST OBJ ##########
TESTDIR = ./test
TCC = gcc -c -g -m64 -Wall
TCCFLAG = $(CCFLAGS)
cputestfunc.o: $(TESTDIR)/cputestfunc.c $(TESTDIR)/cputestfunc.h
	$(TCC) $(TCCFLAG) $(TESTDIR)/cputestfunc.c
gputestfunc.o: $(TESTDIR)/gputestfunc.cu $(TESTDIR)/gputestfunc.h
	$(NVCC) $(NVCCFLAGS) $(TESTDIR)/gputestfunc.cu

###### GPU OBJ ###############
make_struct_gpufunc.o: make_struct_gpufunc.cu cudakernel/gstruct_kernel.cu include/make_struct_gpufunc.h include/header.h
	$(NVCC) $(NVCCFLAGS) make_struct_gpufunc.cu

make_bfs_func.o: make_bfs_func.cu cudakernel/bfs_kernel.cu include/make_bfs_func.h include/header.h
	$(NVCC) $(NVCCFLAGS) make_bfs_func.cu
	
make_stcon_func.o: make_stcon_func.cu cudakernel/bfs_kernel.cu cudakernel/stcon_kernel.cu include/make_bfs_func.h include/make_stcon_func.h include/header.h
	$(NVCC) $(NVCCFLAGS) make_stcon_func.cu

gstruct_kernel.o: cudakernel/gstruct_kernel.cu include/header.h
	$(NVCC) $(NVCCFLAGS) cudakernel/gstruct_kernel.cu

bfs_kernel.o: cudakernel/bfs_kernel.cu include/header.h
	$(NVCC) $(NVCCFLAGS) cudakernel/bfs_kernel.cu
	
stcon_kernel.o: cudakernel/stcon_kernel.cu include/header.h
	$(NVCC) $(NVCCFLAGS) cudakernel/stcon_kernel.cu	

mythrustlib.o: mythrustlib.cu include/header.h
	$(NVCC) $(NVCCFLAGS) mythrustlib.cu

####### CLEAN ############
.PHONY: clean
clean:
	-rm -f $(BIN) $(CPUOBJ) $(TESTOBJ) $(GPUOBJ) $(GPUKERNELOBJ) $(GENLIB) $(CPUREFOBJ)
	-rm -f data/*
	-rm -rf *.dSYM
	-$(MAKE) -C ../generator -f ../generator/Makefile.PLX.mpi clean

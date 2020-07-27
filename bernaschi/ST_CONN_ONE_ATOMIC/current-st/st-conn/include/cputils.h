#ifndef _CPUTILS_H_
#define _CPUTILS_H_

#ifdef __cplusplus
extern "C" {
#endif
void setup_globals_MPI();
void* callmalloc(size_t nbytes, const char* funcname);
void* callrealloc(void* p, size_t nbytes, const char* funcname);
void freemem(void *p);
void print_edges(INT_T* edges, INT_T nedges, FILE *fout, const char *func_name);
void print_edges32(INT32_T* edges, INT_T nedges, FILE *fout, const char *func_name);
void print_array_32t(INT32_T *in, INT_T nelems, FILE* fout, const char *fcaller);
void print_array(INT_T *in, INT_T nelems, FILE* fout, const char *func_name);
void print_array_uint(unsigned int *in, unsigned int nelems, FILE* fout, const char *func_name);
void print_array_64t(int64_t *in, int64_t nelems, FILE* fout, const char *func_name);
void print_fullarray_64t(int64_t *in, int64_t nelems, FILE* fout, const char *fcaller);
int checkLocalEdgeList(INT_T* edges, INT_T nedges, FILE* fout, const char *func_name);
int checkGlobalEdgeList(INT_T* edges, INT_T nedges, FILE* fout, const char *func_name);
int exclusive_scan(INT_T* count_array, INT_T* offset_array, INT_T count_nelems, INT_T offset_nelems);
void CHECK_INPUT(INT_T *pointer, INT_T nelems, const char *fcaller);
void CHECK_INPUT32(INT32_T *pointer, INT_T nelems, const char *fcaller);
void CHECK_SIZE(const char *A_name, INT_T A_value, const char *B_name, INT_T B_value, const char *fcaller);
void PRINT_TIME(const char *fcaller, FILE *fout, double time);
void PRINT_SPACE(const char *fcaller, FILE *fout, const char *myname, double space);
int exclusive_scan_pad(INT_T* count_array,  INT_T* offset_array, INT_T* padded_offset_array,
		                INT_T count_nelems, INT_T offset_nelems);
int exclusive_scan_INT32(INT32_T* count_array,  INT32_T* offset_array, INT32_T count_nelems, INT32_T offset_nelems);

void print_uncolor_array_64t(int64_t *in, int64_t nelems, FILE* fout, const char *fcaller);

#ifdef __cplusplus
}
#endif

#endif


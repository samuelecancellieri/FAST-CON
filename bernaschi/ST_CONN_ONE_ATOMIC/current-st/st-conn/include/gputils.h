/* Needs header.h for the definition of INT_T */
#ifndef _GPUTILS_H_
#define _GPUTILS_H_

#ifdef __cplusplus
extern "C" {
#endif
void checkCUDAError(const char *msg);
void warm_gpu();
void assignDeviceToProcess();
int checkMaxScale(int SCALE, FILE *fout, int coeff);
int checkFreeMemory(INT_T nelems, FILE *fout, const char *func_name);
int printDeviceFreeMemory(FILE *fout);
void print_device_array(INT_T *d_in, INT_T nelems, FILE* fout, const char *fcaller);
void print_device_array32(INT32_T *d_in, INT_T nelems, FILE* fout, const char *fcaller);
#ifdef __cplusplus
}
#endif

#endif

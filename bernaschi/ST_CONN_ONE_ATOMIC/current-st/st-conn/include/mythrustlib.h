/* Needs header.h for the definition of INT_T */
#ifndef _MYTHRUSTLIB_H_
#define _MYTHRUSTLIB_H_
#ifdef __cplusplus
extern "C" {
#endif
int call_thrust_count_unsigned(unsigned int*in, INT_T nelems, INT_T value, INT_T *tcount);
int call_thrust_exclusive_scan(INT_T *in, INT_T *new_end, INT_T nelems, INT_T *offset);
int call_thrust_sort_unique_by_key(INT_T *key, INT_T *payload, INT_T *new_end, INT_T nelems);
int call_thrust_sort_unique_host(INT_T *in, INT_T *new_end, INT_T nelems);
int call_thrust_sort_by_key(INT_T *key, INT_T *payload, INT_T nelems);
int call_thrust_sort(INT_T *in, INT_T nelems);
int call_thrust_sort_host(INT_T *in, INT_T nelems);
int call_thrust_stable_sort_by_key_and_max_host(INT32_T *key, INT32_T *payload, INT32_T nelems, INT32_T* max);
int call_thrust_sort_by_key_and_max(INT_T *key, INT_T *payload, INT_T nelems, INT_T* max);
int call_thrust_stable_sort_by_key_and_max(INT_T *key, INT_T *payload, INT_T nelems, INT_T* max);
int call_thrust_replace(INT32_T *in, INT_T nelems, const int old_value, const int new_value);
int call_thrust_fill(INT_T *in, INT_T nelems, const int value);
int call_thrust_remove(INT_T *in, INT_T nelems, INT_T *new_nelems, int value);
int call_thrust_remove_copy(INT_T *in, INT_T nelems, INT_T *out, INT_T *new_nelems, int value);
int call_thrust_remove_by_stencil(INT_T *in, INT_T nelems, INT_T *new_nelems, unsigned int *stencil);

#ifdef __cplusplus
}
#endif
#endif


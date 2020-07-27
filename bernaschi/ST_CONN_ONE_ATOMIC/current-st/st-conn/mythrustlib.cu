#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "thrust/device_vector.h"
#include "thrust/sort.h"
#include "thrust/remove.h"
#include "thrust/scan.h"
#include "thrust/unique.h"
#include "thrust/pair.h"
#include "thrust/count.h" 

#include "header.h"
#include "defines.h"
#include "mythrustlib.h"

int call_thrust_count_unsigned(unsigned int *in, INT_T nelems, INT_T value, INT_T *tcount)
{
	thrust::device_ptr<unsigned int> d_thrust_in(in);
	int result;
	result = thrust::count(in, in + nelems, value);
	tcount[0] = result;
	return 0;
}

int call_thrust_exclusive_scan(INT_T *in, INT_T *new_end, INT_T nelems, INT_T *offset)
{
	thrust::device_ptr<INT_T> d_thrust_off(offset);
	thrust::device_ptr<INT_T> d_thrust_in(in);
	thrust::exclusive_scan(d_thrust_in, d_thrust_in + nelems, 
			       d_thrust_off);
	new_end[0]  = d_thrust_off[nelems-1] 
		    + d_thrust_in[nelems-1];
	return 0;
}

int call_thrust_sort_unique_by_key(INT_T *key, INT_T *payload, INT_T *new_end, INT_T nelems)
{                                   
	// Sort array via thrust
	thrust::device_ptr<INT_T> d_thrust_key(key);
	thrust::device_ptr<INT_T> d_thrust_payload(payload);
	 
	thrust::sort_by_key(d_thrust_key, 
			    d_thrust_key + nelems,
			    d_thrust_payload);

	// Unique array
	thrust::pair
	<
	thrust::device_ptr<INT_T>,
	thrust::device_ptr<INT_T>
	>
	end;
	end = thrust::unique_by_key(d_thrust_key,
				    d_thrust_key + nelems,
				    d_thrust_payload);

	// Set the new number of elements
	new_end[0] = end.first - d_thrust_key;
	new_end[1] = end.second - d_thrust_payload;

	return 0;
}

int call_thrust_sort_unique_host(INT_T *in, INT_T *new_elems, INT_T nelems)
{
	// Sort array via thrust
	thrust::sort(in, in + nelems);

	// Unique array
	INT_T *new_end = thrust::unique(in, in + nelems);
	new_elems[0] = new_end - in;

	return 0;
}

int call_thrust_sort_by_key(INT_T *key, INT_T *payload, INT_T nelems)
{                                   
	// Sort array via thrust
	thrust::device_ptr<INT_T> d_thrust_key(key);
	thrust::device_ptr<INT_T> d_thrust_payload(payload);
	 
	thrust::sort_by_key(d_thrust_key, 
			    d_thrust_key + nelems,
			    d_thrust_payload);
	return 0;
}

int call_thrust_sort(INT_T *in, INT_T nelems)
{
	// Sort array via thrust
	thrust::device_ptr<INT_T> d_thrust_in(in);

	thrust::sort(d_thrust_in, d_thrust_in + nelems);
	return 0;
}

int call_thrust_sort_host(INT_T *in, INT_T nelems)
{
	// Sort array via thrust
	//thrust::host_ptr<INT_T> d_thrust_in(in);

	thrust::sort(in, in + nelems);
	return 0;
}

int call_thrust_stable_sort_by_key_and_max_host(INT32_T *key, INT32_T *payload, INT32_T nelems, INT32_T* max)
{                                   
	// Sort array via thrust
	thrust::sort_by_key(key, key + nelems, payload);
	
	max[0] = key[nelems-1];

	return 0;
}

int call_thrust_sort_by_key_and_max(INT_T *key, INT_T *payload, INT_T nelems, INT_T* max)
{                                   
	// Sort array via thrust
	thrust::device_ptr<INT_T> d_thrust_key(key);
	thrust::device_ptr<INT_T> d_thrust_payload(payload);
	 
	thrust::stable_sort_by_key(d_thrust_key, 
			    d_thrust_key + nelems,
			    d_thrust_payload);
	
	max[0] = d_thrust_key[nelems-1];

	return 0;
}

int call_thrust_stable_sort_by_key_and_max(INT_T *key, INT_T *payload, INT_T nelems, INT_T* max)
{                                   
	// Sort array via thrust
	thrust::device_ptr<INT_T> d_thrust_key(key);
	thrust::device_ptr<INT_T> d_thrust_payload(payload);
	 
	thrust::sort_by_key(d_thrust_key, 
			    d_thrust_key + nelems,
			    d_thrust_payload);
	
	max[0] = d_thrust_key[nelems-1];

	return 0;
}

int call_thrust_replace(INT32_T *in, INT_T nelems, const int old_value, const int new_value)
{
	thrust::device_ptr<INT32_T> d_thrust_in(in);

	thrust::replace(d_thrust_in, d_thrust_in + nelems, old_value, new_value);

	return 0;
}

int call_thrust_fill(INT_T *in, INT_T nelems, const int value)
{
	thrust::device_ptr<INT_T> d_thrust_in(in);

	thrust::fill_n(d_thrust_in, nelems, value);

	return 0;
}

int call_thrust_remove(INT_T *in, INT_T nelems, INT_T *new_nelems, int value)
{
	thrust::device_ptr<INT_T> d_thrust_in(in);
	thrust::device_ptr<INT_T> new_end;

	new_end = thrust::remove(d_thrust_in, d_thrust_in + nelems, value);
	new_nelems[0] = new_end - d_thrust_in;

	return 0;
}

int call_thrust_remove_copy(INT_T *in, INT_T nelems, INT_T* out, INT_T *new_nelems, int value)
{
	thrust::device_ptr<INT_T> d_thrust_in(in);
	thrust::device_ptr<INT_T> d_thrust_out(out);
	thrust::device_ptr<INT_T> new_end;

	new_end = thrust::remove_copy(d_thrust_in, d_thrust_in + nelems, d_thrust_out, value);
	new_nelems[0] = new_end - d_thrust_out;

	return 0;
}

struct is_uno {
	__host__ __device__
	bool operator()(const int x) {
		return (x == 1);
	}
};

int call_thrust_remove_by_stencil(INT_T *in, INT_T nelems, INT_T *new_nelems, unsigned int *stencil)
{
  	thrust::device_ptr<unsigned int> d_thrust_stencil(stencil);
	thrust::device_ptr<INT_T> d_thrust_in(in);
  	thrust::device_ptr<INT_T> new_end;
  	
	new_end = thrust::remove_if(d_thrust_in, d_thrust_in + nelems, d_thrust_stencil, is_uno());

	new_nelems[0] = new_end - d_thrust_in;

	return 0;
}

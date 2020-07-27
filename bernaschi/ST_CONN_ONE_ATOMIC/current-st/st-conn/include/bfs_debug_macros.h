#ifdef TIME_DEBUG_1
#define BFS_MULTI_TIME_VAR \
	double make_queue_offset_time_start = 0;\
	double make_queue_offset_time_stop = 0;\
	double make_queue_offset_time = 0;\
	double make_binary_time_start = 0;\
	double make_binary_time_stop = 0;\
	double make_binary_time = 0;\
	double enqueue_time_start = 0;\
	double enqueue_time_stop = 0;\
	double enqueue_time = 0;
#endif

#ifdef GPU_DEBUG_1
#define BFS_MULTI_DEBUG_VAR\
	INT_T *h_queue = NULL;\
	INT_T *hh_pred = NULL;\
	INT_T *h_queue_deg = NULL;\
	INT_T *h_queue_off = NULL;\
	INT_T *h_next_off = NULL;\
	INT_T *h_new_queue = NULL;\
	INT_T *hh_send_buff = NULL;\
	INT_T *hh_recv_buff = NULL;\
	h_queue = (INT_T*)callmalloc(nverts*sizeof(INT_T),\
															"make_bfs_multi: malloc h_queue"); 	\
	hh_pred = (INT_T*)callmalloc(nverts*sizeof(int64_t),\
															"make_bfs_multi: malloc hh_pred"); 	\
	h_queue_deg = (INT_T*)callmalloc(nverts*sizeof(INT_T),\
																	"make_bfs_multi: malloc h_queue_deg");\
	h_queue_off = (INT_T*)callmalloc(nverts*sizeof(INT_T),\
																	"make_bfs_multi: malloc h_queue_off");\
	h_next_off = (INT_T*)callmalloc((nverts)*sizeof(INT_T),\
																	"make_bfs_multi: malloc h_next_off");\
	h_new_queue = (INT_T*)callmalloc(2*nverts*sizeof(INT_T),\
																	"make_bfs_multi: malloc"\
																	" h_new_queue");\
	hh_send_buff = (INT_T*)callmalloc(h_send_size*sizeof(INT_T),\
																		"make_bfs_multi: malloc"\
																		" hh_send_buff");\
	hh_recv_buff = (INT_T*)callmalloc(h_send_size*sizeof(INT_T),\
																		"make_bfs_multi: malloc"\
																		" hh_recv_buff");

#define	BFS_MULTI_DEBUG_CHECK_QUEUE_DEG\
			cudaMemcpy(h_queue, d_queue, queue_count*sizeof(INT_T), \
								cudaMemcpyDeviceToHost);\
			checkCUDAError("make_bfs_multi: d_queue -> h_queue");\
			cudaMemcpy(h_queue_deg, d_queue_deg, queue_count*sizeof(INT_T),\
								cudaMemcpyDeviceToHost);\
			checkCUDAError("make_bfs_multi: d_queue_deg -> h_queue_deg");\
			cudaMemcpy(h_queue_off, d_queue_off, queue_count*sizeof(INT_T),\
								cudaMemcpyDeviceToHost);\
			checkCUDAError("make_bfs_multi: d_queue_off -> h_queue_off");\
			print_array(h_queue, queue_count, fp_bfs, "make_bfs_multi: QUEUE");\
			print_array(hg->offset, (2*nverts+1), fp_bfs, \
									"make_bfs_multi: EDGES OFFSET");\
			check_queue_deg(hg, h_queue, h_queue_deg, queue_count);\
			fflush(fp_bfs);

#define	BFS_MULTI_DEBUG_CHECK_QUEUE_OFF\
			cudaMemcpy(h_next_off, d_next_off, (queue_count)*sizeof(INT_T),\
								cudaMemcpyDeviceToHost);\
			checkCUDAError("make_bfs_multi: d_next_off -> h_next_off");\
			fprintf(fp_bfs, "make_bfs_multi:\n");\
			fprintf(fp_bfs, "queue_count = %"PRI64"\n", queue_count);\
			fprintf(fp_bfs, "next_level_vertices = %"PRI64"\n", \
							next_level_vertices);\
			check_queue_off(h_queue_deg, h_next_off, queue_count);	\
			fflush(fp_bfs);

#define	BFS_MULTI_DEBUG_CHECK_NEW_QUEUE\
		fprintf(fp_bfs, "make_bfs_multi: bfs_level = %d\n", bfs_level);\
		fprintf(fp_bfs, "make_bfs_multi: queue_count = %"PRI64"\n", queue_count);\
		fprintf(fp_bfs, "make_bfs_multi: new_queue_count = %"PRI64"\n", \
						new_queue_count);\
		fprintf(fp_bfs, "make_bfs_multi: nvisited = %"PRI64"\n", nvisited[0]);\
		cudaMemcpy(h_queue, d_queue, new_queue_count*sizeof(INT_T), \
							cudaMemcpyDeviceToHost);\
		checkCUDAError("make_bfs_multi: d_queue -> h_queue");\
		cudaMemcpy(hh_pred, d_pred, nverts*sizeof(int64_t), \
							cudaMemcpyDeviceToHost);\
		checkCUDAError("make_bfs_multi: d_pred -> hh_pred");\
		print_array(h_queue, new_queue_count, fp_bfs, \
								"make_bfs_multi: NEXT LEVEL QUEUE");\
		print_array(hh_pred, nverts, fp_bfs, \
								"make_bfs_multi: PRED");\
		check_new_queue(h_queue, new_queue_count, hh_pred);

#define BFS_MULTI_DEBUG_CHECK_BINARY\
			cudaMemcpy(hh_send_buff, d_send_buff, next_level_vertices*sizeof(INT_T),\
								cudaMemcpyDeviceToHost);\
			cudaMemcpy(hh_recv_buff, d_recv_buff, next_level_vertices*sizeof(INT_T),\
								cudaMemcpyDeviceToHost);\
			print_array(hh_send_buff, next_level_vertices, fp_bfs,\
									"make_bfs_multi: HH_SEND_BUFF");\
			print_array(hh_recv_buff, next_level_vertices, fp_bfs,\
									"make_bfs_multi: HH_RECV_BUFF");\
			fflush(fp_bfs);

#define	BFS_MULTI_DEBUG_FREE_VARS\
	free(h_queue); \
	free(hh_pred); \
	free(h_queue_deg);\
	free(h_queue_off);\
	free(h_next_off);\
	free(h_new_queue);\
	free(hh_send_buff);\
	free(hh_recv_buff);
#endif

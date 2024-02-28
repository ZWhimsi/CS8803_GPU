#ifndef TRACE_READER_MAIN_H
#define TRACE_READER_MAIN_H

#include <vector>
#include <queue>
#include "global_types.h"
#include "zlib.h"
#include "trace_reader.h"
#include "sim_defs.h"

using namespace std;

typedef struct kernel_info_s {
  int n_of_warp;
  int n_warp_per_block = 0;
  int n_of_block;
  vector<tuple<int, int, int>> warp_id_v; // warp_id, warp_start_inst_count, warp_total_inst

  Counter inst_count_total = 0;
} kernel_info_s;

typedef struct warp_s {
  gzFile m_trace_file;

  // Trace buffer for reading trace file
  const unsigned trace_buffer_size = 32;                  // number of instruction the buffer can hold
  std::queue <trace_info_nvbit_small_s *> trace_buffer;   // Trace buffer

  // TODO: We need to have a per-warp timestamp marker
  
  bool m_file_opened;
  bool m_trace_ended;
  // int file_pointer_offset = 0;
  int warp_id;
  int block_id; // this one is different from unique_block_id. for every kernel, the id starts from 0
} warp_s;

typedef struct warp_trace_info_node_s {
  warp_s* trace_info_ptr; /**< trace information pointer */
  int warp_id; /**< warp id */
  int unique_block_id; /**< block id */
} warp_trace_info_node_s;

typedef struct block_schedule_info_s {
  bool start_to_fetch = false; /**< start fetching */
  int dispatched_core_id = -1; /**< core id in which this block is launched */
  bool retired = false; /**< retired */
  int dispatched_thread_num = 0;; /**< number of dispatched threads */
  // int retired_thread_num; /**< number of retired threads */
  int total_thread_num = 0; /**< number of total threads */
  // int dispatch_done; /**< dispatch done */
  bool trace_exist = false; /**< trace exist */
  // Counter sched_cycle; /**< scheduled cycle */
  // Counter retire_cycle; /**< retired cycle */
} block_schedule_info_s;

#endif
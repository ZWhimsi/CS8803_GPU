#include "core.h"
#include "trace_reader.h"
#include "macsim.h"
#include <cstring>
#include <algorithm>
#include "cache.h"

using namespace std;

#define ASSERTM(cond, args...)                                    \
do {                                                              \
  if (!(cond)) {                                                  \
    fprintf(stderr, "%s:%d: ASSERT FAILED ", __FILE__, __LINE__); \
    fprintf(stderr, "%s\n", #cond);                               \
    fprintf(stderr, "%s:%d: ASSERT FAILED ", __FILE__, __LINE__); \
    fprintf(stderr, ## args);                                     \
    fprintf(stderr, "\n");                                        \
    exit(15);                                                     \
  }                                                               \
} while (0)

core_c::core_c(macsim* gpusim, int core_id, sim_time_type cur_cycle)
{
  // Initialize core object
  this->gpusim = gpusim;
  this->core_id = core_id;
  this->c_cycle = cur_cycle;

  ENABLE_CACHE = gpusim->m_gpu_params->Enable_GPU_Cache;
  ENABLE_CACHE_LOG = gpusim->m_gpu_params->GPU_Cache_Log;

  l1cache_size = gpusim->m_gpu_params->L1Cache_Size;
  l1cache_assoc = gpusim->m_gpu_params->L1Cache_Assoc;
  l1cache_line_size = gpusim->m_gpu_params->L1Cache_Line_Size;
  l1cache_banks = gpusim->m_gpu_params->L1Cache_Banks;

  // Create L1 cache
  c_l1cache = new cache_c("dcache", l1cache_size, l1cache_assoc, l1cache_line_size,
                         sizeof(uint64_t), l1cache_banks, false, core_id, CACHE_DL1, false, 1, 0, gpusim);
}

core_c::~core_c(){}

void core_c::attach_l2_cache(cache_c * cache_ptr) {
  c_l2cache = cache_ptr;
}

bool core_c::is_retired() {
  return c_retire;
}

sim_time_type core_c::get_cycle(){
  return c_cycle;
}

int core_c::get_insts(){
  return inst_count_total;
}

sim_time_type core_c::get_stall_cycles(){
  return stall_cycles;
}

int core_c::get_running_warp_num(){
  return c_dispatched_warps.size();
}

int core_c::get_max_running_warp_num(){
  return c_max_running_warp_num;
}

void core_c::run_a_cycle(){
  
  if (c_cycle > 5000000000) {
    cout << "Core " << core_id << ", warps: ";
    for (const auto& pair : c_suspended_warps) {
      cout << pair.second->warp_id << " ";
    }
    cout << endl << "Deadlock" << endl;
    c_retire = true;
  }

  c_cycle++;

  // If we have memory response, move corresponding warp to dispatch queue
  while (!c_memory_responses.empty()){
    // cout << "reply: " << c_memory_responses.front() << endl;
    if(c_suspended_warps.count(c_memory_responses.front()) > 0){
      // remove from suspended queue
      warp_s * ready_warp = c_suspended_warps[c_memory_responses.front()];
      c_suspended_warps.erase(ready_warp->warp_id);

      // cout << "#> warp_ready " << ready_warp->warp_id << "\n";
      
      // move to dispatch queue
      c_dispatched_warps.push_back(ready_warp);

      // clear memory response from memory response queue
      c_memory_responses.pop();
    } else {
      // memory response doesn't belong to any warp in dispatch queue: discard it 
      c_memory_responses.pop();
    }
  }

  // Move currently executing warp to back of dispatch queue
  if (c_running_warp != NULL) {
    c_dispatched_warps.push_back(c_running_warp);
    c_running_warp = NULL;
  }

  if (c_dispatched_warps.empty()) {
    // Schedule get warps from block scheduler into dispatched warp
    gpusim->dispatch_warps(core_id, gpusim->block_scheduling_policy);

    // Retire the core if there are no more warps to run
    if (c_dispatched_warps.empty() && c_suspended_warps.empty()){
      c_retire = true;
      cout << "core " << core_id << " retired" << endl;
      return;
    }
  }

  // Schedule a warp
  bool skip_cycle = schedule_warps(gpusim->warp_scheduling_policy);
  if(skip_cycle) {
    stall_cycles++;
    return;
  }

  // printf("#> warp_scheduled %d\n", c_running_warp->warp_id);

  if (!c_running_warp->m_file_opened)
    ASSERTM(0, "error opening trace file");

  // refill trace buffer for the warp if empty
  if(c_running_warp->trace_buffer.empty()) {
    bool reached_eof = gzeof(c_running_warp->m_trace_file);
   
    if (!reached_eof) {
      // Try to refill trace buffer
      unsigned tmp_buf_sz = c_running_warp->trace_buffer_size * TRACE_SIZE;
      char tmp_buf [tmp_buf_sz];
      unsigned bytes_read = gzread(c_running_warp->m_trace_file, tmp_buf, tmp_buf_sz);
      unsigned num_of_insts_read = bytes_read / TRACE_SIZE;

      if (num_of_insts_read == 0) // we reached end of file
        reached_eof = true;

      for(unsigned i=0; i<num_of_insts_read; i++) {
        trace_info_nvbit_small_s * trace_info = new trace_info_nvbit_small_s;
        memcpy(trace_info, &tmp_buf[i*TRACE_SIZE], TRACE_SIZE);
        c_running_warp->trace_buffer.push(trace_info);
      }
    }

  if(reached_eof) {
      // No instructions to execute in buffer and we reached end of trace file: close file
      gzclose(c_running_warp->m_trace_file);
      // cout << "#> warp_finished " << c_running_warp->warp_id << "\n";
      delete c_running_warp;
      c_running_warp = NULL;
      return;
    }
  }

  // pop one instruction, and execute it
  trace_info_nvbit_small_s *trace_info = c_running_warp->trace_buffer.front();
  c_running_warp->trace_buffer.pop();
  inst_count_total++;

  if(send_mem_req(trace_info, ENABLE_CACHE)) {
    // After sending memory request is, warp should be moved to suspended queue     
    c_suspended_warps[c_running_warp->warp_id] = c_running_warp;
    // cout << "#> warp_suspended " << c_running_warp->warp_id << "\n";
    c_running_warp = NULL;
  }
}

bool core_c::schedule_warps(Warp_Scheduling_Policy_Types policy) {
  // Select warp scheduling policy
  switch(policy) {
    case Warp_Scheduling_Policy_Types::ROUND_ROBIN:
      return schedule_warps_rr();
    case Warp_Scheduling_Policy_Types::GTO:
      return schedule_warps_gto();
    case Warp_Scheduling_Policy_Types::CCWS:
      return schedule_warps_ccws();
    default:
      ASSERTM(0, "Warp Scheduling Policy not valid!");
      return true;
  }
}

bool core_c::schedule_warps_rr() { 
  // If there are no available warps to run, skip the cycle
  if (!c_dispatched_warps.empty()) {
    c_running_warp = c_dispatched_warps.front();
    c_dispatched_warps.erase(c_dispatched_warps.begin());
    return false;
  }
  return true;
}

bool core_c::schedule_warps_gto() {
  // TODO: Implement the GTO logic here
  /*
    GTO logic goes here
  */  

  printf("ERROR: GTO Not Implemented\n");   // TODO: remove this
  c_retire = true;                          // TODO: remove this
  return true;
}



bool core_c::schedule_warps_ccws() {
  // TODO: Implement the CCWS logic here
  /*
    CCWS logic goes here
  */  

  printf("ERROR: CCWS Not Implemented\n");   // TODO: remove this
  c_retire = true;                          // TODO: remove this
  return true;
}


bool core_c::send_mem_req(trace_info_nvbit_small_s* trace_info, bool enable_cache){
  if ((is_ld(trace_info->m_opcode) || is_st(trace_info->m_opcode)) && !is_using_shared_memory(trace_info->m_opcode)) {
    gpusim->inc_n_cache_req();

    if (enable_cache){
      Addr line_addr;
      Addr repl_line_addr;
      if(ENABLE_CACHE_LOG) cout << GPU_NVBIT_OPCODE[trace_info->m_opcode] << endl;
      // Check whether the data is in l1 cache. 
      if (c_l1cache->access_cache(trace_info->m_mem_addr, &line_addr, true, 0) == NULL) { 
        // l1 miss
        if(ENABLE_CACHE_LOG) cout << "core id: " << core_id << ", address: " << trace_info->m_mem_addr << ", l1 cache miss!" << endl;
        if (c_l2cache->access_cache(trace_info->m_mem_addr, &line_addr, true, 0) == NULL) { 
          // l2 miss --> gpu has to access it from the ~SSD~ RAM
          if(ENABLE_CACHE_LOG) cout << "core id: " << core_id << ", address: " << trace_info->m_mem_addr << ", l2 cache miss! Create SSD R/W request.." << endl;
          gpusim->inst_event(trace_info, core_id, c_running_warp->block_id, c_running_warp->warp_id, c_cycle);
          c_l2cache->insert_cache(trace_info->m_mem_addr, &line_addr, &repl_line_addr, 0, false); // using cpu_cache_line in cache.cpp
          if(ENABLE_CACHE_LOG) cout << "core id: " << core_id << ", address: " << trace_info->m_mem_addr << ", written to l2 cache! - l2 cache capacity: " << c_l2cache->m_num_cpu_line << "/" << c_l2cache->m_assoc * c_l2cache->m_num_sets << endl;
          return true;
        } else {
          // l2 hit --> write it to l1 cache
          if(ENABLE_CACHE_LOG) cout << "core id: " << core_id << ", address: " << trace_info->m_mem_addr << ", l2 cache hit!" << endl;
          c_l1cache->insert_cache(trace_info->m_mem_addr, &line_addr, &repl_line_addr, 0, false); // using cpu_cache_line in cache.cpp
          if(ENABLE_CACHE_LOG) cout << "core id: " << core_id << ", address: " << trace_info->m_mem_addr << ", written to l1 cache! - l1 cache capacity: " << c_l1cache->m_num_cpu_line << "/" << c_l1cache->m_assoc * c_l1cache->m_num_sets << endl;          
        } 
      } else {
        // l1 hit
        if(ENABLE_CACHE_LOG) cout << "core id: " << core_id << ", address: " << trace_info->m_mem_addr << ", l1 cache hit!" << endl;
      }
      return false;
    } else {
      // when ENABLE_CACHE == false
      gpusim->inst_event(trace_info, core_id, c_running_warp->block_id, c_running_warp->warp_id, c_cycle);
      return true;
    }
  } else {
    return false;
  }
}

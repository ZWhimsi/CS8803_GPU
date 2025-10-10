#include "core.h"
#include "trace.h"
#include "macsim.h"
#include <cstring>
#include <algorithm>
#include "cache.h"
#include "ccws_vta.h"

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

#define CACHELOG(x) if(ENABLE_CACHE_LOG) {x}


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
                         sizeof(cache_data_t), l1cache_banks, false, core_id, CACHE_DL1, false, 1, 0, gpusim);
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
  return c_dispatched_warps.size() + c_suspended_warps.size() + (c_running_warp ? 1 : 0);
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

  WSLOG(printf("-----------------------------------\n");)

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Task 2.3: Decrement LLS scores by 1 point for all warps in the core
  // Decrement LLS for currently running warp
  if (c_running_warp != NULL) {
    if (c_running_warp->ccws_lls_score > CCWS_LLS_BASE_SCORE) {
      c_running_warp->ccws_lls_score--;
    }
  }
  
  // Decrement LLS for active warps in dispatch queue
  for (auto warp : c_dispatched_warps) {
    if (warp->ccws_lls_score > CCWS_LLS_BASE_SCORE) {
      warp->ccws_lls_score--;
    }
  }
  
  // Decrement LLS for suspended warps
  for (auto& suspended_pair : c_suspended_warps) {
    warp_s* warp = suspended_pair.second;
    if (warp->ccws_lls_score > CCWS_LLS_BASE_SCORE) {
      warp->ccws_lls_score--;
    }
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  // If we have memory response, move corresponding warp from suspended queue to dispatch queue
  while (!c_memory_responses.empty()){
    if(c_suspended_warps.count(c_memory_responses.front()) > 0){

      // remove from suspended queue
      warp_s * ready_warp = c_suspended_warps[c_memory_responses.front()];
      c_suspended_warps.erase(ready_warp->warp_id);
      
      // move to dispatch queue
      c_dispatched_warps.push_back(ready_warp);

      // clear memory response from memory response queue
      c_memory_responses.pop();

      WSLOG(printf("Warp ready: %x\n", ready_warp->warp_id);)
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
    int ndispatched_warps = gpusim->dispatch_warps(core_id, gpusim->block_scheduling_policy);
    WSLOG(if(ndispatched_warps > 0)printf("Block scheduler: %d warps dispatched\n", ndispatched_warps);)

    // Retire the core if there are no more warps to run
    if (c_dispatched_warps.empty() && c_suspended_warps.empty()){
      c_retire = true;
      cout << "core " << core_id << " retired" << endl;
      return;
    }
  }

  WSLOG(
  // Print queues
  printf("[%ld,%d]: DQ[", c_cycle, core_id);
  unsigned _indx=0;
  for (auto x: c_dispatched_warps){
    printf("%x:%d%s", x->warp_id, x->ccws_lls_score, (_indx++ != c_dispatched_warps.size()-1?", ":""));
  }
  printf("] SQ["); _indx=0;
  for (auto x: c_suspended_warps){
    printf("%x:%d%s", x.first, x.second->ccws_lls_score, (_indx++ != c_suspended_warps.size()-1?", ":""));
  }
  printf("]\n");
  )

  CCWSLOG(
  // Print VTAs
  for(auto W: c_dispatched_warps){
    printf("dVTA warp:%x: [", W->warp_id);
    W->ccws_vta_entry->print();
    printf("]\n");
  }
  for(auto W: c_suspended_warps){
    printf("sVTA warp:%x: [", W.first);
    W.second->ccws_vta_entry->print();
    printf("]\n");
  }
  )

  // Schedule a warp
  bool skip_cycle = schedule_warps(gpusim->warp_scheduling_policy);
  if(skip_cycle) {
    stall_cycles++;
    return;
  }

  WSLOG(printf("Warp scheduled: %x\n", c_running_warp->warp_id);)


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
      WSLOG(printf("Warp finished: %x\n", c_running_warp->warp_id);)
      delete c_running_warp;
      c_running_warp = NULL;
      return;
    }
  }

  // pop one instruction, and execute it
  trace_info_nvbit_small_s *trace_info = c_running_warp->trace_buffer.front();
  
  //---------- Execute instruction ----------
  if((is_ld(trace_info->m_opcode) || is_st(trace_info->m_opcode)) && !is_using_shared_memory(trace_info->m_opcode)) {
    // Load/Store Op: Send request to memory hierarchy
    CACHELOG(printf("==[Cycle: %ld]============================================\n", c_cycle);)
    CACHELOG(printf("Cache Access: Wid: %x, Addr: 0x%016lx, Wr: %d\n", c_running_warp->warp_id, trace_info->m_mem_addr, is_st(trace_info->m_opcode));)
    bool suspend_warp = send_mem_req(c_running_warp->warp_id, trace_info, ENABLE_CACHE);
    if(suspend_warp) {
      // Memory request initiated, need to suspend without committing
      WSLOG(printf("Warp suspended: %x\n", c_running_warp->warp_id);)
      c_suspended_warps[c_running_warp->warp_id] = c_running_warp;
      c_running_warp = NULL;
      return;
    } 
  }

  // Commit otherwise (non suspending ld/st OR any other instruction)
  c_running_warp->trace_buffer.pop();
  inst_count_total++;
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
  // GTO: Greedy Then Oldest scheduler
  // First try to schedule the same warp again (greedy part)
  if (c_running_warp != NULL) {
    // Check if the running warp is still in dispatch queue
    for (auto it = c_dispatched_warps.begin(); it != c_dispatched_warps.end(); ++it) {
      if (*it == c_running_warp) {
        // Found it! Schedule the same warp again (greedy)
        c_dispatched_warps.erase(it);
        return false; // success, don't stall
      }
    }
  }
  
  // If no running warp or not in queue, find oldest warp (oldest part)
  if (!c_dispatched_warps.empty()) {
    sim_time_type oldest_time = c_dispatched_warps[0]->last_dispatch_cycle;
    int oldest_idx = 0;
    
    // Find warp with oldest dispatch time
    for (int i = 1; i < c_dispatched_warps.size(); i++) {
      if (c_dispatched_warps[i]->last_dispatch_cycle < oldest_time) {
        oldest_time = c_dispatched_warps[i]->last_dispatch_cycle;
        oldest_idx = i;
      }
    }
    
    // Schedule the oldest warp
    c_running_warp = c_dispatched_warps[oldest_idx];
    c_dispatched_warps.erase(c_dispatched_warps.begin() + oldest_idx);
    return false; // success, don't stall
  }
  
  return true; // no warps available, stall cycle
}



bool core_c::schedule_warps_ccws() {
  // CCWS: Cache-Conscious Wavefront Scheduling
  if (c_dispatched_warps.empty()) {
    return true; // no warps available, stall cycle
  }

  // Task 2.4a: Calculate cumulative LLS cutoff
  int num_active_warps = c_dispatched_warps.size();
  int cumulative_lls_cutoff = num_active_warps * CCWS_LLS_BASE_SCORE;
  
  // Task 2.4b: Construct scheduleable warps set
  // Copy dispatch queue and sort by LLS score (descending)
  std::vector<warp_s*> sorted_warps = c_dispatched_warps;
  std::sort(sorted_warps.begin(), sorted_warps.end(), 
    [](warp_s* a, warp_s* b) { return a->ccws_lls_score > b->ccws_lls_score; });
  
  // Build scheduleable set by adding warps until cumulative threshold
  std::vector<warp_s*> scheduleable_warps;
  int cumulative_score = 0;
  
  for (auto warp : sorted_warps) {
    if (cumulative_score + warp->ccws_lls_score <= cumulative_lls_cutoff) {
      scheduleable_warps.push_back(warp);
      cumulative_score += warp->ccws_lls_score;
    }
  }
  
  // Ensure we have at least one scheduleable warp
  if (scheduleable_warps.empty()) {
    scheduleable_warps.push_back(sorted_warps[0]);
  }
  
  // Task 2.4c: Use Round Robin on scheduleable set
  // Find first scheduleable warp in dispatch queue
  for (auto it = c_dispatched_warps.begin(); it != c_dispatched_warps.end(); ++it) {
    warp_s* warp = *it;
    if (std::find(scheduleable_warps.begin(), scheduleable_warps.end(), warp) != scheduleable_warps.end()) {
      // Found scheduleable warp, schedule it
      c_running_warp = warp;
      c_dispatched_warps.erase(it);
      return false; // success, don't stall
    }
  }
  
  return true; // no scheduleable warps found, stall cycle
}


bool core_c::send_mem_req(int wid, trace_info_nvbit_small_s* trace_info, bool enable_cache){
  gpusim->inc_n_cache_req();

  // Check if caches are enabled
  if(!enable_cache) {
    // send request to memory directly
    gpusim->inst_event(trace_info, core_id, c_running_warp->block_id, c_running_warp->warp_id, c_cycle);
    return true; // suspend warp
  }

  //////////////////////////////////////////////////////////////////////////////
  // Access Caches
  // L1 cache: Write through - Write no allocate
  // L2 cache: Write back - write allocate.

  Addr addr = trace_info->m_mem_addr;
  bool is_read = is_ld(trace_info->m_opcode);
  Addr line_addr;
  Addr repl_line_addr;
  
  if(is_read) {
    ////////////////////////////////////////
    // READ

    // Access L1
    cache_data_t * l1_access_data = (cache_data_t*) c_l1cache->access_cache(addr, &line_addr, true, 0);
    bool l1_hit = l1_access_data ? true : false;

    if(l1_hit) {
      // *** L1 Read Hit ***
      // - Return val, continue warp
      gpusim->inc_n_l1_hits();
      
      CACHELOG(printf("L1 Read: Hit\n");)
      return false; // continue warp
    } 
    else {
      // *** L1 Read Miss ***
      CACHELOG(printf("L1 Read: Miss\n");)

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Task 2.2a: Check VTA on L1 read miss and update LLS scores
      // Get tag from the address
      Addr vta_ln_tag;
      int vta_set;
      c_l1cache->find_tag_and_set(addr, &vta_ln_tag, &vta_set);

      // Access the VTA using the tag
      CCWSLOG(printf("VTA Access: %0llx\n", vta_ln_tag);)
      bool vta_hit = c_running_warp->ccws_vta_entry->access(vta_ln_tag);
      if(vta_hit) { // VTA Hit
        // Increment VTA hits counter
        num_vta_hits++;

        // Calculate LLDS using the formula
        int num_active_warps = c_dispatched_warps.size();
        int cum_lls_cutoff = num_active_warps * CCWS_LLS_BASE_SCORE;
        int num_insts = inst_count_total;
        if (num_insts == 0) num_insts = 1; // avoid division by zero
        
        int llds = (num_vta_hits * CCWS_LLS_K_THROTTLE * cum_lls_cutoff) / num_insts;
        
        // Ensure LLS doesn't go below base score
        if (llds < CCWS_LLS_BASE_SCORE) {
          llds = CCWS_LLS_BASE_SCORE;
        }
        
        CCWSLOG(printf("VTA hit! (core:%d, warp: 0x%x, score:%d -> %d)\n", core_id, c_running_warp->warp_id, c_running_warp->ccws_lls_score, llds);)
        c_running_warp->ccws_lls_score = llds;
      }
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      // Access L2
      cache_data_t * l2_access_data = (cache_data_t*) c_l2cache->access_cache(addr, &line_addr, true, 0);
      bool l2_hit = l2_access_data ? true : false;

      if(l2_hit){
        // *** L2 Read Hit ***
        // - Insert in L1: L1 is WTWNA, so no dirty eviction
        // - Return val, continue warp
        CACHELOG(printf("L2 Read: Hit\n");)
        
        // Insert in L1
        cache_data_t* l1_ins_ln = (cache_data_t*)c_l1cache->insert_cache(addr, &line_addr, &repl_line_addr, 0, false);

        //        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Task 2.1a: Insert evicted L1 tag into VTA on L2 hit
        if(repl_line_addr) {
          // Get tag from the evicted line address
          Addr repl_ln_tag;
          int repl_set;
          c_l1cache->find_tag_and_set(repl_line_addr, &repl_ln_tag, &repl_set);
          
          // Insert evicted tag into current warp's VTA
          c_running_warp->ccws_vta_entry->insert(repl_ln_tag);
          CCWSLOG(printf("VTA insertion: %llx\n", repl_ln_tag));
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        return false; // continue warp
      }
      else {
        // *** L2 Read Miss ***
        // - Send memory request
        // - Delegate L2 update to macsim.cpp::get_mem_response()
        // - Delegate L1 update to macsim.cpp::get_mem_response()
        // - Suspend warp

        CACHELOG(printf("L2 Read: Miss, Memory request sent.. (Warp Suspended)\n");)
        gpusim->inst_event(trace_info, core_id, c_running_warp->block_id, c_running_warp->warp_id, c_cycle, true, false);
        
        return true; // suspend warp
      }
    }

  }
  else {
    ////////////////////////////////////////
    // WRITE
    
    // Access L1
    cache_data_t * l1_access_data = (cache_data_t*) c_l1cache->access_cache(addr, &line_addr, true, 0);
    bool l1_hit = l1_access_data ? true : false;

    if(l1_hit) {
      // *** L1 Write Hit ***
      // - Update value in L1: already updated LRU timestamp
      // - Write through to L2
      gpusim->inc_n_l1_hits();
      CACHELOG(printf("L1 Write: Hit, Write val in L1\n");)
    }
    else {
      // *** L1 Write Miss ***
      // - Write through to L2
      CACHELOG(printf("L1 Write: Miss, don't care\n");)
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Task 2.2b: Check VTA on L1 write miss and update LLS scores
      // Get tag from the address
      Addr vta_ln_tag;
      int vta_set;
      c_l1cache->find_tag_and_set(addr, &vta_ln_tag, &vta_set);

      // Access the VTA using the tag
      CCWSLOG(printf("VTA Access: %0llx\n", vta_ln_tag);)
      bool vta_hit = c_running_warp->ccws_vta_entry->access(vta_ln_tag);
      if(vta_hit) { // VTA Hit
        // Increment VTA hits counter
        num_vta_hits++;

        // Calculate LLDS using the formula
        int num_active_warps = c_dispatched_warps.size();
        int cum_lls_cutoff = num_active_warps * CCWS_LLS_BASE_SCORE;
        int num_insts = inst_count_total;
        if (num_insts == 0) num_insts = 1; // avoid division by zero
        
        int llds = (num_vta_hits * CCWS_LLS_K_THROTTLE * cum_lls_cutoff) / num_insts;
        
        // Ensure LLS doesn't go below base score
        if (llds < CCWS_LLS_BASE_SCORE) {
          llds = CCWS_LLS_BASE_SCORE;
        }
        
        CCWSLOG(printf("VTA hit! (core:%d, warp: 0x%x, score:%d -> %d)\n", core_id, c_running_warp->warp_id, c_running_warp->ccws_lls_score, llds);)
        c_running_warp->ccws_lls_score = llds;
      }
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }

    // Write through irrespective of L1 Hit/Miss
    CACHELOG(printf("Writing through to L2\n");)

    cache_data_t * l2_access_data = (cache_data_t*) c_l2cache->access_cache(addr, &line_addr, true, 0);
    bool l2_hit = l2_access_data ? true : false;
    if(l2_hit) {
      // *** L2 Write Hit ***
      // - Mark dirty
      // - Continue Warp
      CACHELOG(printf("L2 Write: Hit, Marking dirty\n");)
      l2_access_data->m_dirty = true;
      return false; // continue
    }
    else {
      // *** L2 Write Miss ***
      // - Send Memory request
      // - Delegate L2 insertion to macsim.cpp::get_mem_response()
      // - Delegate L2 mark dirty to macsim.cpp::get_mem_response()
      // - Suspend warp

      // L2 Miss: Get a block from memory, delegate mark dirty
      CACHELOG(printf("L2 Write: Miss, Memory request sent.. (Warp Suspended)\n");)
      gpusim->inst_event(trace_info, core_id, c_running_warp->block_id, c_running_warp->warp_id, c_cycle, false, true);

      // Need to mark the block dirty after miss repair -> handled in macsim::get_mem_response()
      return true; // suspend warp
    }
  }
}

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

  remove_insts_in_exec_buffer(c_cycle);

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


  // TODO: Task 1: Check if the instruction is a compute instruction and add it to the buffer if so.
  // If the buffer is full, stall the current running warp.
  
  if(is_compute(trace_info->m_opcode)) {
    int latency = get_latency(trace_info->m_opcode, gpusim->tensor_latency);
    int comp_cycle = c_cycle + latency;
    int dest_reg = trace_info->m_dst[0];
    
    bool exec_buf_full = add_insts_to_exec_buffer(comp_cycle, c_running_warp->warp_id, dest_reg);
    if(exec_buf_full) {
      stall_cycles++;
      return;
    }
  }
  
  c_running_warp->trace_buffer.pop();
  inst_count_total++;
}

// TODO: Task 1: Add instructions to the execution buffer (c_exec_buffer).
// If the execution buffer is full, return true.

bool core_c::add_insts_to_exec_buffer(int completion_cycle, int warp_id, int dest_reg) {
  // clean up completed instructions first
  remove_insts_in_exec_buffer(c_cycle);
  
  // check if buffer is full - execution_width allows that many instructions
  if (c_exec_buffer.size() >= gpusim->execution_width) {
    return true;
  }
  
  // add instruction to buffer
  ExecutionData exec_data;
  exec_data.timestamp = completion_cycle;
  exec_data.warp_id = warp_id;
  exec_data.dest_reg = dest_reg;
  c_exec_buffer.push_back(exec_data);
  
  return false;
}

// TODO: Task 1: Remove instructions from the execution buffer if their cycle timestamp is less than or equal to the current cycle.

void core_c::remove_insts_in_exec_buffer(int current_cycle) {
  // remove completed instructions - use erase-remove idiom for efficiency
  c_exec_buffer.erase(
    std::remove_if(c_exec_buffer.begin(), c_exec_buffer.end(),
      [current_cycle](const ExecutionData& exec_data) {
        return exec_data.timestamp <= current_cycle;
      }),
    c_exec_buffer.end()
  );
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

// TODO: Task 2: Incorporate register dependency checking when scheduling warps.
// Iterate through the dispatched warps to find one without dependencies.
// If none are found, stall.

// Note: Dependency checking is skipped when a warp's trace_buffer is empty.

bool core_c::schedule_warps_rr() { 
  // If there are no available warps to run, skip the cycle
  if (c_dispatched_warps.empty()) {
    return true;
  }
  
  // try to find a warp without dependencies
  for (auto it = c_dispatched_warps.begin(); it != c_dispatched_warps.end(); ++it) {
    warp_s* candidate_warp = *it;
    
    // temporarily set running warp for dependency check
    c_running_warp = candidate_warp;
    
    // check for dependency
    bool has_conflict = check_dependency();
    if (!has_conflict) {
      // no dependency, schedule this warp
      c_dispatched_warps.erase(it);
      return false;
    }
  }
  
  // all warps have dependencies, stall
  c_running_warp = NULL;
  return true;
}

// TODO: Task 2: This function should return true if the warp to be scheduled has a register dependency with any of the registers in the execution buffer.
// Register dependency occurs when the destination register of the executing compute instructions matches any of the valid source registers (given by trace_info->m_num_read_regs).
// The registers being compared should belong to the same warp.

bool core_c::check_dependency() {
  // skip if no warp
  if (c_running_warp == NULL) {
    return false;
  }
  
  // skip dependency check if trace buffer is empty (as per README)
  if (c_running_warp->trace_buffer.empty()) {
    return false;
  }
  
  // get next instruction
  trace_info_nvbit_small_s* trace_info = c_running_warp->trace_buffer.front();
  
  // only check dependencies for compute instructions
  if (!is_compute(trace_info->m_opcode)) {
    return false;
  }
  
  // clean up completed instructions first
  remove_insts_in_exec_buffer(c_cycle);
  
  // early exit if no executing instructions
  if (c_exec_buffer.empty()) {
    return false;
  }
  
  // check for conflicts with executing instructions from same warp only
  for (const auto& exec_inst : c_exec_buffer) {
    if (exec_inst.warp_id != c_running_warp->warp_id) {
      continue;
    }
    
    // check if dest reg matches any source reg
    for (int i = 0; i < trace_info->m_num_read_regs; i++) {
      int src_reg = trace_info->m_src[i];
      if (exec_inst.dest_reg == src_reg) {
        return true;
      }
    }
  }
  
  return false;
}

bool core_c::schedule_warps_gto() {
  // Implement the GTO logic here
  /*
    GTO logic goes here
  */  

  printf("ERROR: GTO Not Implemented\n");   // remove this
  c_retire = true;                          // remove this
  return true;
}



bool core_c::schedule_warps_ccws() {
  //Implement the CCWS logic here
  /*
    CCWS logic goes here
  */  

  printf("ERROR: CCWS Not Implemented\n");   // remove this
  c_retire = true;                          // remove this

  // determine cumulative LLS cutoff 
  int cumulative_lls_cutoff = 0; 
  
  if (!c_dispatched_warps.empty()) {
    // Construct schedulable warps set:
    // - Create a copy of the dispatch queue, and sort it in descending order.
    // - Collect the the warps with highest LLS scores (until we reach the cumulative cutoff) to construct the 
    //   schedulable warps set.

    // Copy dispatch queue

    // sort the vector by scores (descending order)

    // Construct set of scheduleable warps by adding warps till we hit the cumulative threshold
    std::vector<warp_s*> scheduleable_Warps;
  
    assert(scheduleable_Warps.size() > 0);   // We should always have atleast one schedulable warp

    // Use Round Robin as baseline scheduling logic to schedule warps from the dispatch queue only if 
    // the warp is present in the scheduleable warps set

  }

  return true;
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
      // Upon L1 Read miss, we need to check if the tag corresponding to the address is present in 
      // currently executing warp's VTA.

      // Get tag from address (see if there is any method in cache class to help with this)
      Addr vta_ln_tag;

      // Access the VTA using the tag
      CCWSLOG(printf("VTA Access: %0llx\n", vta_ln_tag);)
      bool vta_hit = false;
      if(vta_hit) { // VTA Hit
        // Increment VTA hits counter

        // Update the VTA Score to LLDS
        int llds = 0;
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

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Insert the tag in warp's VTA entry upon L1 eviction.
        // Steps:
        //  - Get tag corresponding to the address. (see if any of the cache class methods can help with this)
        //  - The warp which issued the memory request is the currently executing warp, Insert the tag in warp's VTA entry
        if(repl_line_addr) {
          // Get the tag from the address
          Addr repl_ln_tag; 
          
          // Insert tag in warp's VTA entry
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
      // Upon L1 Read miss, we need to check if the tag corresponding to the address is present in 
      // currently executing warp's VTA.

      // Get tag from address (see if there is any method in cache class to help with this)
      Addr vta_ln_tag;

      // Access the VTA using the tag
      CCWSLOG(printf("VTA Access: %0llx\n", vta_ln_tag);)
      bool vta_hit = false;
      if(vta_hit) { // VTA Hit
        // Increment VTA hits counter

        // Update the VTA Score to LLDS
        int llds = 0;
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

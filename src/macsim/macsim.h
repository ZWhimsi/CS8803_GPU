#ifndef MACSIM_H
#define MACSIM_H

#include <string>
#include <queue>
#include <vector>
#include <algorithm>

#include "sim_defs.h"
#include "utils.h"
#include "trace_reader_main.h"
#include "trace_reader.h"

#include "exec/GPU_Parameter_Set.h"
#include "ram.h"

using namespace std;

class core_c;
class GPU_Parameter_Set;

enum class Block_Scheduling_Policy_Types {
  ROUND_ROBIN = 0,
  LARGE_CHUNK = 1,
  STRIDED = 2,
  LASP = 3,
  CA_AWARE = 4,
  CA_GC = 5,
};

constexpr const char* Block_Scheduling_Policy_Types_str[] = {
  "ROUND_ROBIN"
};

enum class Warp_Scheduling_Policy_Types {
  ROUND_ROBIN = 0,
  GTO = 1,
  CCWS = 2, 
};

constexpr const char* Warp_Scheduling_Policy_Types_str[] = {
  "ROUND_ROBIN",
  "GTO",
  "CCWS"
};

struct GPU_scoreboard_entry {
  uint64_t PC;
  sim_time_type req_time = 0;
  bool is_mem;
  int core_id;
  int warp_id;
  int mem_queue_id = -1;
};

class macsim {
public:
  // Create a macsim object
  macsim(GPU_Parameter_Set* gpu_params);

  // Destroy macsim object
  ~macsim();
  
  // Set queues
  void set_queues(queue<RAM_request>* req_queue_ptr, queue<RAM_response>* resp_queue_ptr) {
    gpu_mem_request_queue = req_queue_ptr;
    gpu_mem_response_queue = resp_queue_ptr;
  }

  // Get cycles elapsed
  int get_cycle() { return m_cycle; }

  // Get number of memory responses recieved
  int get_n_responses() { return n_responses; }

  // Get number of memory requests generated
  int get_n_requests() { return n_requests;}

  // Get average memory latency
  uint32_t get_avg_latency() { return n_responses == 0 ? 0 : total_latency/n_responses; }

  // setup trace reader
  void trace_reader_setup();
  
  // Generates memory request for lower level memory model if there is a L2 miss
  void inst_event(trace_info_nvbit_small_s* trace_info, int core_id, int block_id, int warp_id, sim_time_type c_cycle);
  
  // Get memory response from memory and 
  void get_mem_response();

  // Run a cycle
  bool run_a_cycle();

  // Start kernel  
  void start_kernel();

  // End kernel
  void end_kernel();

  void create_warp_node(int kernel_id, int warp_id);
  void insert_block(warp_trace_info_node_s *node);
  warp_trace_info_node_s* fetch_warp_from_block(int block_id);
  int retire_block_helper(int core_id);

  /**
   * Dispatch warps to specified core
   * if (core_id == -1), dispatch to all cores (used for initialization)
   * warp_to_run == NULL if there is no more warp to schedule (schedule := put in the core's c_dispatched_warps queue)
  */
  void dispatch_warps(int core_id, Block_Scheduling_Policy_Types policy);

  warp_s* initialize_warp(int warp_id);

  // Block scheduler
  int schedule_blocks(int core_id, Block_Scheduling_Policy_Types policy);

  // Round Robin block scheduler
  int schedule_blocks_rr(int core_id);

  // Check if all cores retired
  bool is_every_core_retired();

  // Print simulation stats
  void print_stats();

  // Finish simulation
  void end_sim();

  // Increment cache requests
  void inc_n_cache_req() { n_cache_req++; }


  uint64_t global_memory_base_addr = 0;
  
  sim_time_type m_cycle;
  int n_of_cores;
  int max_block_per_core;
  Block_Scheduling_Policy_Types block_scheduling_policy;
  Warp_Scheduling_Policy_Types warp_scheduling_policy;
  int kernel_id = 0;
  vector<string> kernels_v;
  vector<core_c *> core_pointers_v;
  pool_c<warp_trace_info_node_s> *trace_node_pool; /**<  trace node pool */
  pool_c<warp_s> *warp_pool;
  vector<kernel_info_s> kernel_info_v;
  int m_kernel_block_start_count = 0;
  int m_num_active_warps = 0;
  int m_num_waiting_dispatched_warps = 0;
  map<int, bool> m_block_list; /**< list of block that has started execution */

  unordered_map<int, block_schedule_info_s *> m_block_schedule_info; /**< block schedule info */
  vector<int> m_block_schedule_order; /**< block schedule order */

  /**< block queue indexed by block ID, list contains queue of warps*/
  unordered_map<int, list<warp_trace_info_node_s *> *> *m_block_queue;
  unordered_map<int, sim_time_type> c_cycle_total;
  unordered_map<int, int> c_insts_total;
  unordered_map<int, sim_time_type> c_stall_cycles;
  bool gpu_retired = false;
  GPU_Parameter_Set* m_gpu_params;

  vector<pair<sim_time_type, int>> m_active_chips;
  vector<pair<sim_time_type, int>> m_chip_contention;
  bool is_GC_busy;

  int n_gpu_precondition = 0;
  bool is_precondition = false;
  int n_resp_precondition = 0;
  int m_cycle_post_precondition = 0;

private:
  string kernel_config_path;
  int n_repeat_kernel;
  sim_time_type cur_cycle;
  int m_cycle_per_period;
  
  uint64_t n_requests; // track number of GPU memory queue request, also used as unique ID to identify
  uint64_t n_responses;
  uint64_t n_timeout_req; // track number of GPU memory queue request that get a response timeout
  uint64_t n_cache_req;

  int n_blocks_total; 
  vector<int> n_blocks_per_kernel;
  int n_total_ppa_prediction, n_correct_ppa_prediction;
  sim_time_type total_latency = 0;
  
  queue<RAM_request>* gpu_mem_request_queue;
  queue<RAM_response>* gpu_mem_response_queue;

  // For strided block scheduling
  int schedule_block_offset = 0;

  bool kernel_starting = true;
  bool kernel_ending = false;

  // scoreboard to track GPU requests on the fly
  vector<GPU_scoreboard_entry> GPU_scoreboard;
  void read_trace(string trace_path, int truncate_size);

  int l2cache_size; 
  int l2cache_assoc; 
  int l2cache_line_size; // Granularity, 64-bit data
  int l2cache_banks;
};

const std::string GPU_NVBIT_OPCODE[] = {
  "FADD",
  "FADD32I",
  "FCHK",
  "FFMA32I",
  "FFMA",
  "FMNMX",
  "FMUL",
  "FMUL32I",
  "FSEL",
  "FSET",
  "FSETP",
  "FSWZADD",
  "MUFU",
  "HADD2",
  "HADD2_32I",
  "HFMA2",
  "HFMA2_32I",
  "HMMA",
  "HMUL2",
  "HMUL2_32I",
  "HSET2",
  "HSETP2",
  "DADD",
  "DFMA",
  "DMUL",
  "DSETP",
  "BMMA",
  "BMSK",
  "BREV",
  "FLO",
  "IABS",
  "IADD",
  "IADD3",
  "IADD32I",
  "IDP",
  "IDP4A",
  "IMAD",
  "IMMA",
  "IMNMX",
  "IMUL",
  "IMUL32I",
  "ISCADD",
  "ISCADD32I",
  "ISETP",
  "LEA",
  "LOP",
  "LOP3",
  "LOP32I",
  "POPC",
  "SHF",
  "SHL",
  "SHR",
  "VABSDIFF",
  "VABSDIFF4",
  "F2F",
  "F2I",
  "I2F",
  "I2I",
  "I2IP",
  "FRND",
  "MOV",
  "MOV32I",
  "MOVM",
  "PRMT",
  "SEL",
  "SGXT",
  "SHFL",
  "PLOP3",
  "PSETP",
  "P2R",
  "R2P",
  "LD",
  "LDC",
  "LDG",
  "LDL",
  "LDS",
  "LDSM",
  "ST",
  "STG",
  "STL",
  "STS",
  "MATCH",
  "QSPC",
  "ATOM",
  "ATOMS",
  "ATOMG",
  "RED",
  "CCTL",
  "CCTLL",
  "ERRBAR",
  "MEMBAR",
  "CCTLT",
  "R2UR",
  "S2UR",
  "UBMSK",
  "UBREV",
  "UCLEA",
  "UFLO",
  "UIADD3",
  "UIADD3_64",
  "UIMAD",
  "UISETP",
  "ULDC",
  "ULEA",
  "ULOP",
  "ULOP3",
  "ULOP32I",
  "UMOV",
  "UP2UR",
  "UPLOP3",
  "UPOPC",
  "UPRMT",
  "UPSETP",
  "UR2UP",
  "USEL",
  "USGXT",
  "USHF",
  "USHL",
  "USHR",
  "VOTEU",
  "TEX",
  "TLD",
  "TLD4",
  "TMML",
  "TXD",
  "TXQ", 
  "SUATOM",
  "SULD",
  "SURED",
  "SUST",
  "BMOV",
  "BPT",
  "BRA",
  "BREAK",
  "BRX",
  "BRXU",
  "BSSY",
  "BSYNC",
  "CALL",
  "EXIT",
  "JMP",
  "JMX",
  "JMXU",
  "KILL",
  "NANOSLEEP",
  "RET",
  "RPCMOV",
  "RTT",
  "WARPSYNC",
  "YIELD",
  "B2R",
  "BAR",
  "CS2R",
  "DEPBAR",
  "GETLMEMBASE",
  "LEPC",
  "NOP",
  "PMTRIG",
  "R2B",
  "S2R",
  "SETCTAID",
  "SETLMEMBASE",
  "VOTE"
};

enum GPU_NVBIT_OPCODE_ {
  FADD = 0,
  FADD32I,
  FCHK,
  FFMA32I,
  FFMA,
  FMNMX,
  FMUL,
  FMUL32I,
  FSEL,
  FSET,
  FSETP,
  FSWZADD,
  MUFU,
  HADD2,
  HADD2_32I,
  HFMA2,
  HFMA2_32I,
  HMMA,
  HMUL2,
  HMUL2_32I,
  HSET2,
  HSETP2,
  DADD,
  DFMA,
  DMUL,
  DSETP,
  BMMA,
  BMSK,
  BREV,
  FLO,
  IABS,
  IADD,
  IADD3,
  IADD32I,
  IDP,
  IDP4A,
  IMAD,
  IMMA,
  IMNMX,
  IMUL,
  IMUL32I,
  ISCADD,
  ISCADD32I,
  ISETP,
  LEA,
  LOP,
  LOP3,
  LOP32I,
  POPC,
  SHF,
  SHL,
  SHR,
  VABSDIFF,
  VABSDIFF4,
  F2F,
  F2I,
  I2F,
  I2I,
  I2IP,
  FRND,
  MOV,
  MOV32I,
  MOVM,
  PRMT,
  SEL,
  SGXT,
  SHFL,
  PLOP3,
  PSETP,
  P2R,
  R2P,
  LD,
  LDC,
  LDG,
  LDL,
  LDS,
  LDSM,
  ST,
  STG,
  STL,
  STS,
  MATCH,
  QSPC,
  ATOM,
  ATOMS,
  ATOMG,
  RED,
  CCTL,
  CCTLL,
  ERRBAR,
  MEMBAR,
  CCTLT,
  R2UR,
  S2UR,
  UBMSK,
  UBREV,
  UCLEA,
  UFLO,
  UIADD3,
  UIADD3_64,
  UIMAD,
  UISETP,
  ULDC,
  ULEA,
  ULOP,
  ULOP3,
  ULOP32I,
  UMOV,
  UP2UR,
  UPLOP3,
  UPOPC,
  UPRMT,
  UPSETP,
  UR2UP,
  USEL,
  USGXT,
  USHF,
  USHL,
  USHR,
  VOTEU,
  TEX,
  TLD,
  TLD4,
  TMML,
  TXD,
  TXQ,
  SUATOM,
  SULD,
  SURED,
  SUST,
  BMOV,
  BPT,
  BRA,
  BREAK,
  BRX,
  BRXU,
  BSSY,
  BSYNC,
  CALL,
  EXIT,
  JMP,
  JMX,
  JMXU,
  KILL,
  NANOSLEEP,
  RET,
  RPCMOV,
  RTT,
  WARPSYNC,
  YIELD,
  B2R,
  BAR,
  CS2R,
  DEPBAR,
  GETLMEMBASE,
  LEPC,
  NOP,
  PMTRIG,
  R2B,
  S2R,
  SETCTAID,
  SETLMEMBASE,
  VOTE
};

const std::string LD_LIST[] = {
  "LD",
  "LDC",
  "LDG",
  "LDL",
  "LDS",
  "LDSM"
};

const std::string ST_LIST[] = {
  "ST",
  "STG",
  "STL",
  "STS"
};

const std::string SHARED_MEM_LIST[] = {
  "LDS",
  "LDSM",
  "STS"
};

inline bool is_ld(uint8_t opcode){
  auto it = find(begin(LD_LIST), end(LD_LIST), GPU_NVBIT_OPCODE[opcode]);
  return (it != end(LD_LIST));
}
inline bool is_st(uint8_t opcode){
  auto it = find(begin(ST_LIST), end(ST_LIST), GPU_NVBIT_OPCODE[opcode]);
  return (it != end(ST_LIST));
}
inline bool is_using_shared_memory(uint8_t opcode){
  auto it = find(begin(SHARED_MEM_LIST), end(SHARED_MEM_LIST), GPU_NVBIT_OPCODE[opcode]);
  return (it != end(SHARED_MEM_LIST));
}

#endif // MACSIM_H
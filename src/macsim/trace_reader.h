/*
Copyright (c) <2012>, <Georgia Institute of Technology> All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided
with the distribution.

Neither the name of the <Georgia Institue of Technology> nor the names of its contributors
may be used to endorse or promote products derived from this software without specific prior
written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef TRACE_READER_H
#define TRACE_READER_H

// #include "global_types.h"
#include <vector>
#include <unordered_map>
#include <map>
#include <string>
#include "global_types.h"

//nvbit defines
#define NVBIT_TRACE
#define MAX_NVBIT_SRC_NUM 4
#define MAX_NVBIT_DST_NUM 4
typedef struct trace_info_nvbit_small_s {
  uint8_t m_opcode;
  bool m_is_fp;
  bool m_is_load;
  uint8_t m_cf_type;
  uint8_t m_num_read_regs;
  uint8_t m_num_dest_regs;
  uint16_t m_src[MAX_NVBIT_SRC_NUM];
  uint16_t m_dst[MAX_NVBIT_DST_NUM];
  uint8_t m_size;

  uint32_t m_active_mask;
  uint32_t m_br_taken_mask;
  uint64_t m_inst_addr;
  uint64_t m_br_target_addr;
  union {
    uint64_t m_reconv_inst_addr;
    uint64_t m_mem_addr;
  };
  union {
    uint8_t m_mem_access_size;
    uint8_t m_barrier_id;
  };
  uint16_t m_num_barrier_threads;
  union {
    uint8_t m_addr_space;  // for loads, stores, atomic, prefetch(?)
    uint8_t m_level;  // for membar
  };
  uint8_t m_cache_level;  // for prefetch?
  uint8_t m_cache_operator;  // for loads, stores, atomic, prefetch(?)
} trace_info_nvbit_small_s;

#define TRACE_SIZE sizeof(trace_info_nvbit_small_s)

class trace_reader_c
{
  public:
    trace_reader_c();
    virtual ~trace_reader_c();

    void inst_event(trace_info_nvbit_small_s* inst);

    virtual void print();
    virtual void reset();

    void init();
    static trace_reader_c Singleton;

  protected:
    std::vector<trace_reader_c*> m_tracer;
    std::string m_name;
};


class reuse_distance_c : public trace_reader_c
{
  public:
    reuse_distance_c();
    ~reuse_distance_c();
#if defined(GPU_TRACE)
    void inst_event(trace_info_gpu_small_s* inst);
#elif defined(NVBIT_TRACE)
    void inst_event(trace_info_nvbit_small_s* inst);
#elif defined(ARM64_TRACE)
    void inst_event(trace_info_a64_s* inst);
#else
    void inst_event(trace_info_cpu_s* inst);
#endif
    void print();
    void reset();

  private:
    int m_self_counter;
    std::unordered_map<Addr, int> m_reuse_map;
    std::unordered_map<Addr, int> m_reuse_pc_map;
    std::map<Addr, bool> m_static_pc;
    std::unordered_map<Addr, std::unordered_map<Addr, std::unordered_map<int, bool> *> *> m_result;
};


class static_pc_c : public trace_reader_c
{
  public:
    static_pc_c();
    ~static_pc_c();

#if defined(GPU_TRACE)
    void inst_event(trace_info_gpu_small_s* inst);
#elif defined(NVBIT_TRACE)
    void inst_event(trace_info_nvbit_small_s* inst);
#elif defined(ARM64_TRACE)
    void inst_event(trace_info_a64_s* inst);
#else
    void inst_event(trace_info_cpu_s* inst);
#endif

  private:
    std::unordered_map<Addr, bool> m_static_pc;
    std::unordered_map<Addr, bool> m_static_mem_pc;
    uint64_t m_total_inst_count;
    uint64_t m_total_load_count;
};


#endif

import os

trace_dir = os.getenv('MACSIM_TRACE_DIR')

benchmarks = {
    'gemm_float':         f'{trace_dir}/gemm_float/kernel_config.txt',
    'gemm_half':          f'{trace_dir}/gemm_half/kernel_config.txt',
    'cnn_float':          f'{trace_dir}/cnn_float/kernel_config.txt',
    'cnn_half':           f'{trace_dir}/cnn_half/kernel_config.txt',
    'ffn_float':          f'{trace_dir}/ffn_float/kernel_config.txt',
    'ffn_half':           f'{trace_dir}/ffn_half/kernel_config.txt',
    'gpt2_float':         f'{trace_dir}/gpt2_float/kernel_config.txt',
    'gpt2_half':          f'{trace_dir}/gpt2_half/kernel_config.txt',
}

gpu_configs = {
    'CC':   'xmls/gpuconfig_8c_cc.xml',
    'TC':   'xmls/gpuconfig_8c_tc.xml'
}

stats = [
    'NUM_CYCLES',
    'NUM_INSTRS_RETIRED',
    'NUM_STALL_CYCLES'
]
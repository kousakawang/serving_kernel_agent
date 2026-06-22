

cmd_probe_target_calls:没有以非cuda-graph模式运行服务和测试

serving_kernel_agent/framework_engineer/templates/probe_cuda_extension.py
这个文件的实现有问题，会有如下的错误：
FAILED: [code=1] main.o 
c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=kernel_agent_probe_cuda_extension -DTORCH_API_INCLUDE_EXTENSION_H -isystem /usr/local/lib/python3.12/dist-packages/torch/include -isystem /usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cu130/kernel_agent_probe_cuda_extension/main.cpp -o main.o 
/root/.cache/torch_extensions/py312_cu130/kernel_agent_probe_cuda_extension/main.cpp: In function ‘void pybind11_init_kernel_agent_probe_cuda_extension(pybind11::module_&)’:
/root/.cache/torch_extensions/py312_cu130/kernel_agent_probe_cuda_extension/main.cpp:4:46: error: ‘add_one’ was not declared in this scope
    4 | m.def("add_one", torch::wrap_pybind_function(add_one), "add_one");

我看了下临时生成的cpp文件，里面没有包含CUDA_SRC里定义的代码
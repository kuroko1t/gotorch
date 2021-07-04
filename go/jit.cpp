/*
MIT License

Copyright (c) 2019 kurosawa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <torch/torch.h>
#include <jit.h>
#include <torch/script.h>

JitModule Load(const char *path) {
  std::string spath(path);
  torch::jit::script::Module* module = new torch::jit::script::Module();
  *module = torch::jit::load(spath);
  return (void*)module;
}

ATensor JitForward(JitModule module, ATensor tensor) {
  at::Tensor *ori_tensor = (at::Tensor*)tensor;
  torch::jit::script::Module *ori_module = (torch::jit::script::Module*)module;
  at::Tensor *ret_tensor = new at::Tensor();
  std::vector<torch::jit::IValue> input_vec;
  input_vec.push_back(*ori_tensor);
  *ret_tensor = ori_module->forward(input_vec).toTensor();
  return (void*)ret_tensor;
}

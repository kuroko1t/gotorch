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
#include <torch/script.h>
#include <gotorch.h>

struct TorchModel : public torch::nn::Module {
  using torch::nn::Module::register_module;
};

LinearImpl linear(int a, int b) {
  torch::nn::LinearImpl *linear= new torch::nn::LinearImpl(a, b);
  return linear;
}

Tensor Randn(int* shape, int size) {
  torch::Tensor *tensor = new torch::Tensor();
  std::vector<int64_t> x;
  for (int i=0; i < size; i++) {
    x.push_back((int64_t)shape[i]);
  }
  *tensor = torch::randn(x);
  return (void*)tensor;
}

TModel modelInit() {
  TorchModel *mod= new TorchModel();
  return (void*)mod;
}

void params_size(TModel model, int *size) {
    TorchModel* tmodel = (TorchModel*) model;
    *size = tmodel->parameters().size();
}

void params(TModel model, int size, Tensor *tensor) {
    TorchModel* tmodel = (TorchModel*) model;
    for (int i = 0; i < size; i++) {
        torch::Tensor *datatensor = new torch::Tensor();
        *datatensor = tmodel->parameters()[i];
        tensor[i] = datatensor;
    }
}

SGD optimizer_sgd(Tensor *tensor, float lr, int size) {
  std::vector<torch::Tensor> tensors;
  for (int i=0; i < size; i++) {
    tensors.push_back(*(torch::Tensor*)tensor[i]);
  }
  torch::optim::SGD *optimizer = new torch::optim::SGD(tensors, lr);
  return optimizer;
}

void optimizer_zero_grad(SGD optimizer) {
  ((torch::optim::SGD*)optimizer)->zero_grad();
}

void optimizer_step(SGD optimizer) {
  ((torch::optim::SGD*)optimizer)->step();
}

int istraining(TModel model) {
  TorchModel* tmodel = (TorchModel*) model;
  return tmodel->is_training();
}

LinearImpl register_module_linear(const char *name, LinearImpl linear, TModel mod) {
  std::string str(name);
  TorchModel *mod_test = (TorchModel*)mod;
  torch::nn::LinearImpl *plinear = (torch::nn::LinearImpl*)linear;
  std::shared_ptr<torch::nn::LinearImpl> p1(plinear);
  return (void*)((mod_test->register_module(str, p1)).get());
}

Conv2dImpl register_module_conv2d(const char *name, Conv2dImpl conv2d, TModel mod) {
  std::string str(name);
  TorchModel *mod_re = (TorchModel*)mod;
  torch::nn::Conv2dImpl *conv2d_re = (torch::nn::Conv2dImpl*)conv2d;
  std::shared_ptr<torch::nn::Conv2dImpl> conv2d_re_sh(conv2d_re);
  return ((mod_re->register_module(str, conv2d_re_sh)).get());
}

Dropout2dImpl register_module_featureDropout(const char *name,
					     Dropout2dImpl featuredrop, TModel mod) {
  std::string str(name);
  TorchModel *mod_re = (TorchModel*)mod;
  torch::nn::Dropout2dImpl *featuredrop_re = (torch::nn::Dropout2dImpl*)featuredrop;
  std::shared_ptr<torch::nn::Dropout2dImpl> featuredrop_re_sh(featuredrop_re);
  return ((mod_re->register_module(str, featuredrop_re_sh)).get());
}


Conv2dImpl conv2d(int in_channels, int out_channels, int kernel_size) {
  torch::nn::Conv2dImpl *conv2d = new torch::nn::Conv2dImpl(in_channels, out_channels, kernel_size);
  return conv2d;
}

Dropout2dImpl FeatureDropout() {
  torch::nn::Dropout2dImpl *conv_drop =
    new torch::nn::Dropout2dImpl();
  return conv_drop;
}

Tensor forward_linear(LinearImpl linear, Tensor tensor) {
  torch::nn::LinearImpl* plinear = (torch::nn::LinearImpl*)linear;
  torch::Tensor* ptensor = (torch::Tensor*)tensor;
  torch::Tensor *atensor = new torch::Tensor();
  torch::Tensor* go_atensor = (torch::Tensor*)atensor;
  *go_atensor = plinear->forward(*ptensor);
  return (void*)go_atensor;
}

Tensor forward_conv2d(Conv2dImpl conv2d, Tensor tensor) {
  torch::nn::Conv2dImpl* conv2d_re = (torch::nn::Conv2dImpl*)conv2d;
  torch::Tensor* ptensor = (torch::Tensor*)tensor;
  torch::Tensor *atensor = new torch::Tensor();
  torch::Tensor* go_atensor = (torch::Tensor*)atensor;
  *go_atensor = conv2d_re->forward(*ptensor);
  return (void*)go_atensor;
}

Tensor forward_featureDropout(Dropout2dImpl featuredrop, Tensor tensor) {
  torch::nn::Dropout2dImpl* featuredrop_re = (torch::nn::Dropout2dImpl*)featuredrop;
  torch::Tensor* ptensor = (torch::Tensor*)tensor;
  torch::Tensor *atensor = new torch::Tensor();
  torch::Tensor* go_atensor = (torch::Tensor*)atensor;
  *go_atensor = featuredrop_re->forward(*ptensor);
  return (void*)go_atensor;
}

//using mnistDataset = torch::data::StatelessDataLoader<
//        torch::data::datasets::MapDataset<
//            torch::data::datasets::MNIST,
//            torch::data::transforms::Stack<torch::data::Example<>>>,
//        torch::data::samplers::RandomSampler>;

//using example_data = torch::data::Example<at::Tensor, at::Tensor>;

int data_loader_size(const char *path, int batch_size) {
  std::string spath(path);
  auto dataset = torch::data::make_data_loader(
      torch::data::datasets::MNIST(spath)
      .map(torch::data::transforms::Stack<>()),
      batch_size);
  int size = 0;
  for (auto& x : *dataset) {
    size +=1;
  }
  return size;
}

void data_loader(const char *path, int batch_size,
                 Tensor *data_vec, Tensor *target_vec) {
  std::string spath(path);
  auto dataset = torch::data::make_data_loader(
      torch::data::datasets::MNIST(spath)
      .map(torch::data::transforms::Stack<>()),
      batch_size);
  int i = 0;
  for (auto& x : *dataset) {
      torch::Tensor *datatensor = new torch::Tensor();
      torch::Tensor *targettensor = new torch::Tensor();
      *datatensor = x.data;
      data_vec[i] = datatensor;
      *targettensor = x.target;
      target_vec[i] = targettensor;
      i+=1;
  }
}

int tensor_size(Tensor tensor, int dim) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  return atensor->size(dim);
}

Tensor tensor_view(Tensor tensor, int* shape, int size) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  std::vector<int64_t> x;
  for (int i=0; i < size; i++) {
    x.push_back((int64_t)shape[i]);
  }
  c10::IntArrayRef x1 = c10::IntArrayRef(x);
  torch::Tensor *ret_tensor = new torch::Tensor();
  *ret_tensor = atensor->view(x1);
  return ret_tensor;
}


Tensor tensor_reshape(Tensor tensor, int* shape, int size) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  std::vector<int64_t> x;
  for (int i = 0; i < size; i++) {
    x.push_back((int64_t)shape[i]);
  }
  torch::Tensor *ret_tensor = new torch::Tensor();
  c10::IntArrayRef x1 = c10::IntArrayRef(x);
  *ret_tensor = atensor->reshape(x1);
  return (void*)ret_tensor;
}

int tensor_is_cuda(Tensor tensor) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  return atensor->is_cuda();
}

void backward(Tensor tensor) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  atensor->backward();
}

Tensor log_softmax(Tensor tensor, int dim) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  torch::Tensor *ret_tensor = new torch::Tensor();
  *ret_tensor = torch::log_softmax(*atensor, dim);
  return (void*)ret_tensor;
}

Tensor tensor_nll_loss(Tensor tensor, Tensor target) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  torch::Tensor *atarget = (torch::Tensor*)target;
  torch::Tensor *ret_tensor = new torch::Tensor();
  *ret_tensor = torch::nll_loss(*atensor, *atarget);
  return (void*)ret_tensor;
}

Tensor relu(Tensor tensor) {
    torch::Tensor *atensor = (torch::Tensor*)tensor;
    torch::Tensor *ret_tensor = new torch::Tensor();
    *ret_tensor = torch::relu(*atensor);
    return (void*)ret_tensor;
}

Tensor dropout(Tensor tensor, float droprate, int is_training) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  torch::Tensor *ret_tensor = new torch::Tensor();
  bool is_training_bool = false;
  if (is_training != 0) {
    is_training_bool = true;
  }
  *ret_tensor = torch::dropout(*atensor, droprate, is_training_bool);
  return ret_tensor;
}

Tensor max_pool2d(Tensor tensor, int kernel_size) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  torch::Tensor *ret_tensor = new torch::Tensor();
  *ret_tensor = torch::max_pool2d(*atensor, kernel_size);
  return (void*)ret_tensor;
}

float tensor_item(Tensor tensor) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  return atensor->item<float>();
}

Tensor tensor_to_device(Tensor tensor, CPU device) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  torch::Device* device_re = (torch::Device*)device;
  torch::Tensor *ret_tensor = new torch::Tensor();
  *ret_tensor = atensor->to(*device_re);
  return ret_tensor;
}

Tensor tensor_to_cuda(Tensor tensor, CUDA device) {
  return tensor_to_device(tensor, device);
}

Tensor tensor_to_cpu(Tensor tensor, CPU device) {
  return tensor_to_device(tensor, device);
}

void save(TModel model, const char *path) {
  std::string spath(path);
  TorchModel* tmodel = (TorchModel*) model;
  std::shared_ptr<torch::nn::Module> t1model =
    std::make_shared<torch::nn::Module>(*tmodel);
  torch::save(t1model, spath);
}

TModel load(const char *path) {
  std::string spath(path);
  torch::jit::script::Module *model = new torch::jit::script::Module();
  *model = torch::jit::load(spath);
  return (void*)model;
}

int cuda_is_available() {
  return torch::cuda::is_available();
}

CUDA cuda_device() {
  torch::Device *device = new torch::Device(torch::kCUDA);
  return device;
}

CPU cpu_device() {
  torch::Device *device = new torch::Device(torch::kCPU);
  return device;
}

void model_to_cuda(TModel model, CUDA device) {
  TorchModel* tmodel = (TorchModel*) model;
  torch::Device* device_re = (torch::Device*)device;
  tmodel->to(*device_re);
}

void model_to_cpu(TModel model, CPU device) {
  TorchModel* tmodel = (TorchModel*) model;
  torch::Device* device_re = (torch::Device*)device;
  tmodel->to(*device_re);
}

float32vec AtensorToVec(ATensor atensor) {
  //std::vector<float> *vec = new std::vector<float>();
  at::Tensor *ori_atensor = (at::Tensor*)atensor;
  //vec = ori_tensor->data_ptr();
  //return vec;
  std::vector<float> *v = new std::vector<float>((*ori_atensor).data_ptr<float>(), (*ori_atensor).data_ptr<float>() + (*ori_atensor).numel());
  return (void*)v;
} 

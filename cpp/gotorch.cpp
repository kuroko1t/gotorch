//#include <torch/nn/modules/linear.h>
//#include <torch/csrc/api/include/torch/nn/module.h>
//#include <torch/nn/module.h>
#include <torch/torch.h>
#include <gotorch.h>

struct TorchModel : public torch::nn::Module {
  using torch::nn::Module::register_module;
};

Linear torch_nn_Linear(int a, int b) {
  torch::nn::LinearImpl *linear= new torch::nn::LinearImpl(a, b);
  return linear;
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

Linear Register_module(const char *name, Linear linear, TModel mod) {
  std::string str(name);
  TorchModel *mod_test = (TorchModel*)(mod);
  torch::nn::LinearImpl *plinear = (torch::nn::LinearImpl*)linear;
  std::shared_ptr<torch::nn::LinearImpl> p1(plinear);
  return (void*)((mod_test->register_module(str, p1)).get());
}

Tensor forward(Linear linear, Tensor tensor) {
  torch::nn::LinearImpl* plinear = (torch::nn::LinearImpl*)linear;
  torch::Tensor* ptensor = (torch::Tensor*)tensor;
  torch::Tensor *atensor = new torch::Tensor();
  torch::Tensor* go_atensor = (torch::Tensor*)atensor;
  *go_atensor = plinear->forward(*ptensor);
  return (void*)go_atensor;
}

using mnistDataset = torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            torch::data::datasets::MNIST,
            torch::data::transforms::Stack<torch::data::Example<>>>,
        torch::data::samplers::RandomSampler>;

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

Tensor tensor_reshape(Tensor tensor, int* shape, int size) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  std::vector<long> x;
  for (int i = 0; i < size; i++) {
    x.push_back((long)shape[i]);
  }
  torch::Tensor *ret_tensor = new torch::Tensor();
  *ret_tensor = atensor->reshape(x);
  return (void*)ret_tensor;
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



float tensor_item(Tensor tensor) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  return atensor->item<float>();
}

// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/common/reshard_function_desc.h"

namespace phi {
class DeviceContext;

namespace distributed {

class DistTensor;
class TensorDistAttr;

class ReshardFunction {
 public:
  ReshardFunction() = default;
  virtual ~ReshardFunction() = default;

  virtual bool IsSuitable(const DistTensor& in,
                          const TensorDistAttr& out_dist_attr) = 0;

  std::shared_ptr<DistTensor> Eval(DeviceContext* dev_ctx,
                                   const DistTensor& in,
                                   const TensorDistAttr& out_dist_attr);

  virtual void Eval(DeviceContext* dev_ctx,
                    const DistTensor& in,
                    const TensorDistAttr& out_dist_attr,
                    DistTensor* out) = 0;

  std::vector<std::unique_ptr<ReshardFuncDesc>> GetReshardFuncDescs() {
    return std::move(reshard_func_descs_);
  }
  virtual std::string Name() { return "ReshardBase"; }

 protected:
  void SetValue(DistTensor* tensor, const DenseTensor& value);
  void SetDistProps(DistTensor* tensor,
                    const DDim& dims,
                    const TensorDistAttr& dist_attr);
  void SetDistProps(DistTensor* tensor, const TensorDistAttr& dist_attr);
  DenseTensor* GetMutableTensor(DistTensor* tensor);
  std::vector<std::unique_ptr<ReshardFuncDesc>> reshard_func_descs_;
};

}  // namespace distributed
}  // namespace phi

// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue_utils.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"

using AtomicVectorSizeT = std::vector<std::atomic<size_t>>;

namespace paddle {
namespace framework {
namespace interpreter {

using DeviceContext = platform::DeviceContext;

class ContextManager {
 public:
  using DeviceContextMap =
      std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>;

  static ContextManager& Instance() {
    static ContextManager* ctx_manager = new ContextManager;
    return *ctx_manager;
  }

  std::shared_future<std::unique_ptr<DeviceContext>> Get(
      const std::string& type,
      const platform::Place& place,
      int stream_priority) {
    std::lock_guard<std::mutex> lk(ctx_mtx_);
    VLOG(6) << "Get dev_ctx for " << type << " - " << place;

    DeviceContextMap& ctxs = ctx_pool_[type];
    if (ctxs.find(place) == ctxs.end()) {
      platform::EmplaceDeviceContexts(
          &ctxs,
          {place},
          /*disable_setting_default_stream_for_allocator=*/true,
          stream_priority);
    }
    return ctxs[place];
  }

 private:
  ContextManager() {}
  DISABLE_COPY_AND_ASSIGN(ContextManager);

  std::mutex ctx_mtx_;
  std::unordered_map<std::string, DeviceContextMap> ctx_pool_;
};


class AsyncWorkQueue {
 public:
  AsyncWorkQueue(size_t host_num_threads,
                 size_t deivce_num_threads,
                 EventsWaiter* waiter);

  // void WaitEmpty() { queue_group_->WaitQueueGroupEmpty(); }

  void AddTask(const OpFuncType& op_func_type, std::function<void()> fn);

  void Cancel() { queue_group_->Cancel(); }

  size_t QueueNumThreads(size_t idx) {
    return queue_group_->QueueNumThreads(idx);
  }

 private:
  size_t host_num_thread_;
  std::unique_ptr<WorkQueueGroup> queue_group_;
};

bool IsCommunicationOp(const OperatorBase* op);

bool IsCommunicationOp(const Instruction& instr);

bool IsCpuOp(const Instruction& instr);

bool IsGradOp(const std::string& op_name);

bool IsMemcpyD2H(const Instruction& instr);

bool IsMemcpyH2D(const Instruction& instr);

bool IsMemcpyOp(const Instruction& instr);

bool IsSupportedHeterPlace(const phi::Place& place);

void AddFetch(const std::vector<std::string>& fetch_names,
              framework::BlockDesc* block);

void BuildOpFuncList(const platform::Place& place,
                     const framework::BlockDesc& block,
                     const std::set<std::string>& skip_gc_vars,
                     std::vector<OpFuncNode>* vec_func_list,
                     VariableScope* scope,
                     const ExecutionConfig& execution_config,
                     bool use_local_scope = true,
                     bool static_build = false);

void BuildOpFuncList(
    const platform::Place& place,
    ::ir::Block* block,
    std::vector<OpFuncNode>* vec_func_list,
    framework::Scope* scope,
    framework::Scope* local_scope,
    const std::unordered_map<::ir::Value, std::string>& value_2_name_map,
    const ExecutionConfig& execution_config);

void BuildVariableScope(const framework::BlockDesc& block,
                        const ExecutionConfig& execution_config,
                        VariableScope* var_scope);

void LogDeviceMemoryStats(const platform::Place& place);

bool IsPhiCommOp(framework::OperatorBase* operator_base);

void SetDeviceCommContext(framework::OperatorBase* operator_base,
                          platform::DeviceContext* dev_ctx);
}  // namespace interpreter
}  // namespace framework
}  // namespace paddle

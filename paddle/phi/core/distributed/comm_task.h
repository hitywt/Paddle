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

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include "glog/logging.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/macros.h"

namespace phi {
namespace distributed {

class Store;
class CommTask {
 public:
  CommTask(const std::string& backend = "",
           const phi::Place& place = phi::Place(),
           int rank = -1,
           int size = 0,
           int gid = 0,
           uint64_t seq = 0,
           int64_t numel = 0,
           CommType comm_type = CommType::UNKNOWN)
      : backend_(backend),
        place_(place),
        rank_(rank),
        size_(size),
        gid_(gid),
        seq_(seq),
        numel_(numel),
        comm_type_(comm_type) {
    const char* global_rank = std::getenv("PADDLE_TRAINER_ID");
    PADDLE_ENFORCE_NOT_NULL(
        global_rank,
        phi::errors::NotFound(
            "The environment variable 'PADDLE_TRAINER_ID' cannot be found."));
    global_rank_ = std::atoi(global_rank);
  }
  virtual ~CommTask() = default;

  std::string GetBackend() { return backend_; }
  phi::Place GetPlace() { return place_; }
  int GetGlobalRank() { return global_rank_; }
  int GetRank() { return rank_; }
  int GetSize() { return size_; }
  int GetGid() { return gid_; }
  int64_t GetNumel() { return numel_; }
  uint64_t GetSeq() { return seq_; }
  CommType GetCommType() { return comm_type_; }
  bool GetTraceUpdated() { return start_trace_updated_; }
  void SetTraceUpdated() { start_trace_updated_ = true; }
  std::chrono::time_point<std::chrono::steady_clock> GetStartTime() {
    return start_time_;
  }
  std::shared_ptr<Store> GetStore() { return store_; }
  void SetStore(std::shared_ptr<Store> store) { store_ = store; }

  virtual std::string GetTraceMsg() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return "";
  }
  virtual void StartRecord() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }
  virtual void EndRecord() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }

  virtual std::string GetCommErrors() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }
  virtual bool IsStarted() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual bool IsTimeout() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual bool IsCompleted() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual void AbortComm() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }

 protected:
  std::string backend_;
  phi::Place place_;
  int global_rank_;
  int rank_;
  int size_;
  int gid_;
  uint64_t seq_{0};
  int64_t numel_;
  CommType comm_type_;
  bool start_trace_updated_{false};

  bool completed_ = false;
  bool aborted_{false};
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  std::shared_ptr<Store> store_;

 private:
  DISABLE_COPY_AND_ASSIGN(CommTask);
};

}  // namespace distributed
}  // namespace phi

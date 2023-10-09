// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/nccl_comm_context.h"

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/check/nccl_dynamic_check.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"
#include <stdlib.h>

namespace phi {
namespace distributed {

// set this flag to `true` and recompile to enable dynamic checks
constexpr bool FLAGS_enable_nccl_dynamic_check = false;

NCCLCommContext::NCCLCommContext(int rank, int size, ncclUniqueId nccl_id)
    : CommContext(rank, size) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclCommInitRank(&nccl_comm_, size_, nccl_id, rank_));
}

ncclComm_t NCCLCommContext::GetNcclComm() { return nccl_comm_; }

gpuStream_t NCCLCommContext::GetStream() { return dev_ctx_->stream(); }

phi::GPUContext* NCCLCommContext::GetDevContext() { return dev_ctx_.get(); }

void NCCLCommContext::SetDevContext(
    std::unique_ptr<phi::GPUContext>&& dev_ctx) {
  dev_ctx_ = std::move(dev_ctx);
}

gpuEvent_t NCCLCommContext::GetComputeEvent() { return compute_event_.get(); }

void NCCLCommContext::SetComputeEvent(
    std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type>&&
        compute_event) {
  compute_event_ = std::move(compute_event);
}

gpuEvent_t NCCLCommContext::GetCommEvent() { return comm_event_.get(); }

void NCCLCommContext::SetCommEvent(
    std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type>&& comm_event) {
  comm_event_ = std::move(comm_event);
}

int64_t MakeHang(const phi::DenseTensor& in_tensor, const std::string& op, int rank, int size) {

  if (access("/root/test/hang", F_OK) == -1){
    VLOG(1) << "debug not make hang";
    return -1;
  }

  char* ch_op = std::getenv("DEBUG_OP");
  std::string debug_op;
  if (ch_op) {
    debug_op = std::string(ch_op);
  }

  char* ch_rank = std::getenv("DEBUG_RANK");
  int debug_rank = -1;
  if (ch_rank) {
    debug_rank = atoi(ch_rank);
  }

  char* ch_nranks = std::getenv("DEBUG_NRANKS");
  int debug_nranks = -1;
  if (ch_nranks) {
    debug_nranks = atoi(ch_nranks);
  }

  int64_t numel = in_tensor.numel();

  if (debug_op.find(op) != std::string::npos && debug_rank == rank && debug_nranks == size) {
      if (numel > 5) {
          numel -= 5;
          VLOG(0) << "make hang for op " << op << ", rank " << debug_rank << ", nranks " << debug_nranks;
      } else {
          VLOG(0) << "can't make hang for op " << op << ", rank " << debug_rank << ", nranks " << debug_nranks;
      }
      return numel;
  }
  VLOG(1) << "debug not make hang for op " << op << ", rank " << debug_rank << ", nranks " << debug_nranks;
  return -1;
}

void NCCLCommContext::Broadcast(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int root,
                                gpuStream_t stream) {
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(*out_tensor, root, rank_, nccl_comm_);
  }

  int64_t numel = MakeHang(in_tensor, "Broadcast", rank_, size_);
  numel = numel != -1 ? numel : in_tensor.numel();

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclBroadcast(in_tensor.data(),
                                  out_tensor->data(),
                                  //in_tensor.numel(),
                                  numel,
                                  ToNCCLDataType(in_tensor.type()),
                                  root,
                                  nccl_comm_,
                                  stream));
}

void NCCLCommContext::AllGather(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                gpuStream_t stream) {
  phi::distributed::CommStaticCheck::GatherLikeShape(*out_tensor,
                                                     in_tensor,
                                                     /*dst_rank*/ rank_,
                                                     /*cur_rank*/ rank_,
                                                     size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ 0,
                                                   rank_,
                                                   nccl_comm_);
  }

  int64_t numel = MakeHang(in_tensor, "AllGather", rank_, size_);
  numel = numel != -1 ? numel : in_tensor.numel();

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclAllGather(in_tensor.data(),
                                  out_tensor->data(),
                                  //in_tensor.numel(),
                                  numel,
                                  ToNCCLDataType(in_tensor.type()),
                                  nccl_comm_,
                                  stream));
}
void NCCLCommContext::ReduceScatter(phi::DenseTensor* out_tensor,
                                    const phi::DenseTensor& in_tensor,
                                    ncclRedOp_t reduce_type,
                                    gpuStream_t stream) {
  phi::distributed::CommStaticCheck::ScatterLikeShape(*out_tensor,
                                                      in_tensor,
                                                      /*dst_rank*/ rank_,
                                                      /*cur_rank*/ rank_,
                                                      size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ 0,
                                                   rank_,
                                                   nccl_comm_);
  }

  int64_t numel = MakeHang(*out_tensor, "ReduceScatter", rank_, size_);
  numel = numel != -1 ? numel : out_tensor->numel();

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclReduceScatter(in_tensor.data(),
                                      out_tensor->data(),
                                      //out_tensor->numel(),
                                      numel,
                                      ToNCCLDataType(in_tensor.type()),
                                      reduce_type,
                                      nccl_comm_,
                                      stream));
}

void NCCLCommContext::Send(const phi::DenseTensor& in_tensor,
                           const int64_t& count,
                           const int& peer,
                           gpuStream_t stream) {
  phi::distributed::CommStaticCheck::CheckShape(in_tensor, rank_, size_);

  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(in_tensor, rank_, rank_, nccl_comm_);
  }

  int64_t numel = MakeHang(in_tensor, "Send", rank_, size_);
  if (numel != -1) {
    if (count > 5) {
      numel = count - 5;
	  VLOG(0) << "make hang for send";
    } else {
	  VLOG(0) << "can't make hang for send";
    }
  } else {
    numel = count;
  }

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclSend(in_tensor.data(),
                             //count,
                             numel,
                             ToNCCLDataType(in_tensor.dtype()),
                             peer,
                             nccl_comm_,
                             stream));
  VLOG(3) << "rank " << GetRank() << " send " << phi::product(in_tensor.dims())
          << " to " << peer;
}

void NCCLCommContext::Recv(phi::DenseTensor* out_tensor,
                           const int64_t& count,
                           const int& peer,
                           gpuStream_t stream) {
  phi::distributed::CommStaticCheck::CheckShape(*out_tensor, rank_, size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(*out_tensor, peer, rank_, nccl_comm_);
  }

  int64_t numel = MakeHang(*out_tensor, "Recv", rank_, size_);
  if (numel != -1) {
    if (count > 5) {
      numel = count - 5;
	  VLOG(0) << "make hang for recv";
    } else {
	  VLOG(0) << "can't make hang for recv";
    }
  } else {
    numel = count;
  }

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclRecv(out_tensor->data(),
                             //count,
                             numel,
                             ToNCCLDataType(out_tensor->dtype()),
                             peer,
                             nccl_comm_,
                             stream));
  VLOG(3) << "rank " << GetRank() << " recv "
          << phi::product(out_tensor->dims()) << " from " << peer;
}

void NCCLCommContext::AllReduce(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                ncclRedOp_t reduce_type,
                                gpuStream_t stream) {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ rank_,
                                               /*cur_rank*/ rank_,
                                               size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ 0,
                                                   rank_,
                                                   nccl_comm_);
  }

  int64_t numel = MakeHang(in_tensor, "AllReduce", rank_, size_);
  numel = numel != -1 ? numel : in_tensor.numel();

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclAllReduce(in_tensor.data(),
                                  out_tensor->data(),
                                  //in_tensor.numel(),
                                  numel,
                                  ToNCCLDataType(in_tensor.type()),
                                  reduce_type,
                                  nccl_comm_,
                                  stream));
}

void NCCLCommContext::Reduce(phi::DenseTensor* out_tensor,
                             const phi::DenseTensor& in_tensor,
                             ncclRedOp_t reduce_type,
                             int root,
                             gpuStream_t stream) {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ root,
                                               /*cur_rank*/ rank_,
                                               size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ root,
                                                   rank_,
                                                   nccl_comm_);
  }

  int64_t numel = MakeHang(in_tensor, "Reduce", rank_, size_);
  numel = numel != -1 ? numel : in_tensor.numel();

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclReduce(in_tensor.data(),
                               out_tensor->data(),
                               //in_tensor.numel(),
                               numel,
                               ToNCCLDataType(in_tensor.type()),
                               reduce_type,
                               root,
                               nccl_comm_,
                               stream));
}

void NCCLCommContext::GroupStart() {
  NCCL_CHECK(phi::dynload::ncclGroupStart());
}
void NCCLCommContext::GroupEnd() { NCCL_CHECK(phi::dynload::ncclGroupEnd()); }

}  // namespace distributed
}  // namespace phi

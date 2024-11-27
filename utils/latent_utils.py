import warnings
import torch
import torch.nn.functional as F


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def get_delta_weight(self, adapter) -> torch.Tensor:
    # This function is introduced in newer PEFT versions. we modify this function instead of modifying
    # the merge function (as we did previously for version 0.4.0 of PEFT).
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # In case users wants to merge the adapter weights that are in
    # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()

    output_tensor = transpose(
        weight_B @ self.default_lora_latent_mapping.weight @ weight_A,
        self.fan_in_fan_out
    ) * self.scaling[adapter]

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

        # cast back the weights
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)

    return output_tensor


# def forward_latent(self, x: torch.Tensor):
#     previous_dtype = x.dtype

#     if self.active_adapter[0] not in self.lora_A.keys():
#         return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
#     if self.disable_adapters:
#         if self.r[self.active_adapter[0]] > 0 and self.merged:
#             self.unmerge()
#         result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
#     elif self.r[self.active_adapter[0]] > 0 and not self.merged:
#         result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

#         x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

#         # adding latent_mapping in the forward loop
#         result += (
#             self.lora_B[self.active_adapter[0]](
#                 self.default_lora_latent_mapping(
#                     self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x))
#                 )
#             )
#             * self.scaling[self.active_adapter[0]]
#         )
#     else:
#         result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

#     result = result.to(previous_dtype)

#     return result
# def forward_latent(self, x: torch.Tensor):
#     previous_dtype = x.dtype

#     # 如果当前适配器不在 lora_A 的 keys 中，直接返回线性变换结果
#     if self.active_adapter[0] not in self.lora_A.keys():
#         return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    
#     # 如果禁用了 adapter，且 merged 状态被激活，取消 merge 并返回线性变换结果
#     if self.disable_adapters:
#         if self.r[self.active_adapter[0]] > 0 and self.merged:
#             self.unmerge()
#         result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
#     # 正常的 LoRA 流程
#     elif self.r[self.active_adapter[0]] > 0 and not self.merged:
#         result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

#         # 确保输入 x 的 dtype 与 lora_A 的权重类型一致
#         x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

#         # 1. 基本的 default_lora_latent_mapping 处理
#         latent_result = self.default_lora_latent_mapping(
#             self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x))
#         )

#         # 通过 lora_B 处理 latent_mapping 结果并缩放
#         result += (
#             self.lora_B[self.active_adapter[0]](latent_result)
#             * self.scaling[self.active_adapter[0]]
#         )

#         # 2. 多头机制的处理
#         # 我们定义一个变量来累积所有多头的结果
#         multihead_result = 0
#         for head in self.multihead_lora_mappings:
#             # 多头的处理，每个头都有独立的 A -> Head -> B 流程
#             head_result = head(
#                 self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x))
#             )
            
#             # 将每个头的结果通过 lora_B 处理，并加入缩放
#             multihead_result += (
#                 self.lora_B[self.active_adapter[0]](head_result)
#                 * self.scaling[self.active_adapter[0]]
#             )

#         # # 3. 将所有多头的结果进行平均（也可以改为其他融合方式，如加权和）
#         # multihead_result /= len(self.multihead_lora_mappings)

#         # 最终将多头的结果加入到 result 中
#         result += multihead_result
#         result = result/(len(self.multihead_lora_mappings)+1)
#     else:
#         result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

#     # 将结果恢复为原始的 dtype
#     result = result.to(previous_dtype)

#     return result

def forward_latent(self, x: torch.Tensor):
    previous_dtype = x.dtype

    if self.active_adapter[0] not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter[0]] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[self.active_adapter[0]] > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

        # adding latent_mapping in the forward loop
        result += (
            self.lora_B[self.active_adapter[0]](
                self.default_lora_latent_mapping(
                    self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x))
                )
            )
            * self.scaling[self.active_adapter[0]]
        )
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    result = result.to(previous_dtype)

    return result


import math
import types

import peft
import torch
from peft.import_utils import is_bnb_available
from peft.utils import _get_submodules
from torch.nn import init
from tqdm import tqdm

from .latent_utils import get_delta_weight, forward_latent
from .svd_utils import get_linear_rec_svd


def get_replacement_module(weight, module_name, type, writer, reconstruct_config):
    cfg = reconstruct_config[type]
    print('--------------------lz')
    print(torch.cuda.is_available())
    if type == 'svd':
        reconstructed_matrix, enc, dec = get_linear_rec_svd(weight.cpu().detach().numpy(), cfg['rank'],
                                                            cfg['n_iter'],
                                                            cfg['random_state'])
        final_enc = torch.tensor(enc, dtype=weight.dtype, device=weight.device)
        final_dec = torch.tensor(dec, dtype=weight.dtype, device=weight.device)
    else:
        raise NotImplementedError(f"{type} is currently not supported.")
    return final_enc, final_dec


def init_module_weights(target_module: torch.nn.Linear, sigma: float):
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias"):
        # Set bias to zeros
        if target_module.bias is not None:
            torch.nn.init.zeros_(target_module.bias)


def replace_module_weights(target_module, new_weight):
    device = target_module.weight.device
    target_module.weight = torch.nn.Parameter(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def update_decoder_weights(target_module, new_weight):
    device = target_module.weight.device
    with torch.no_grad():
        target_module.weight.copy_(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def kaiming_uniform_init_lower_half(matrix: torch.tensor):
    rows, _ = matrix.size()
    init.kaiming_uniform_(matrix[math.ceil(rows / 2):, :], a=math.sqrt(5))
    return matrix

def kaiming_uniform_init(matrix: torch.tensor):
    init.kaiming_uniform_(matrix, a=math.sqrt(5))
    return matrix


import torch
import types
from tqdm import tqdm

import torch
import types
from tqdm import tqdm

import torch
import types
from tqdm import tqdm


# def find_and_initialize(model, peft_config, adapter_name, reconstr_type, reconstruct_config, writer):
#     """
#     :param adapter_name: options: 'default'
#     :param reconstr_type: options: 'svd'
#     """
#     half_init_dec = reconstruct_config['half_init_dec']
#     replacement_module_random_init = reconstruct_config['replacement_module_random_init']
#     reconstruction_mode = reconstruct_config['reconstr_mode']
#     lora_config = peft_config[adapter_name]
#     r_squared = reconstruct_config['r_squared']  # whether using r*r matrix between lora_A and lora_B or not
#     loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
#
#     if loaded_in_8bit and not is_bnb_available():
#         raise ImportError(
#             "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
#             "You can install it with `pip install bitsandbytes`."
#         )
#
#     is_target_modules_in_base_model = False
#     key_list = [key for key, _ in model.named_modules()]
#     print(len(key_list))
#
#     assert (not isinstance(lora_config.target_modules, str))
#
#     print("Iterating through model's specified modules to initialize A/B matrices.")
#     for key in tqdm(key_list):
#         target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
#         if target_module_found:
#             if not is_target_modules_in_base_model:
#                 is_target_modules_in_base_model = True
#             _, target, target_name = _get_submodules(model, key)
#
#             if reconstruction_mode == 'separated':
#                 # Retrieve or compute replacement weights
#                 replacement_encoder_weight, replacement_decoder_weight = get_replacement_module(
#                     weight=target.weight.T,
#                     module_name=key,
#                     type=reconstr_type,
#                     writer=writer,
#                     reconstruct_config=reconstruct_config
#                 )
#
#                 if not isinstance(target, peft.tuners.lora.Linear):
#                     raise NotImplementedError('Only initialization for peft.tuners.lora.Linear type is implemented.')
#                 else:
#                     if half_init_dec:
#                         kaiming_uniform_init_lower_half(replacement_decoder_weight)
#                     if replacement_module_random_init:
#                         kaiming_uniform_init(replacement_encoder_weight)
#                         kaiming_uniform_init(replacement_decoder_weight)
#
#                     # Ensure tensors are on the same device
#                     device = target.lora_B.default.weight.device
#
#                     replacement_decoder_weight = replacement_decoder_weight.to(device)
#                     replacement_encoder_weight = replacement_encoder_weight.to(device)
#
#                     # Initialize scaling matrices and move them to the same device
#                     Lambda_b = torch.diag(torch.randn(target.lora_B.default.weight.shape[0], device=device))
#                     Lambda_d = torch.diag(torch.randn(target.lora_A.default.weight.shape[1], device=device))
#
#                     # Debug shapes
#                     print("Lambda_b shape:", Lambda_b.shape)
#                     print("Replacement decoder weight shape:", replacement_decoder_weight.shape)
#                     print("Lambda_d shape:", Lambda_d.shape)
#                     print("Replacement encoder weight shape:", replacement_encoder_weight.shape)
#
#                     # Check if transposition is needed
#                     if Lambda_b.shape[1] != replacement_decoder_weight.shape[0]:
#                         print("Transposing replacement_decoder_weight to align with Lambda_b.")
#                         replacement_decoder_weight = replacement_decoder_weight.T
#
#                     if Lambda_d.shape[0] != replacement_encoder_weight.shape[1]:
#                         print("Transposing replacement_encoder_weight to align with Lambda_d.")
#                         replacement_encoder_weight = replacement_encoder_weight.T
#
#                     # Check shapes again before multiplication
#                     print("Lambda_b shape:", Lambda_b.shape)
#                     print("Replacement decoder weight shape:", replacement_decoder_weight.shape)
#                     print("Lambda_d shape:", Lambda_d.shape)
#                     print("Replacement encoder weight shape:", replacement_encoder_weight.shape)
#
#                     # Fix scaling matrices application
#                     if r_squared:
#                         # Ensure correct order and shapes for multiplication
#                         if Lambda_b.shape[1] != replacement_decoder_weight.shape[0]:
#                             raise ValueError(
#                                 f"Lambda_b shape {Lambda_b.shape} and replacement_decoder_weight shape {replacement_decoder_weight.shape} are incompatible for multiplication.")
#
#                         if Lambda_d.shape[0] != replacement_encoder_weight.shape[1]:
#                             raise ValueError(
#                                 f"Lambda_d shape {Lambda_d.shape} and replacement_encoder_weight shape {replacement_encoder_weight.shape} are incompatible for multiplication.")
#
#                         scaled_lora_B = Lambda_b @ replacement_decoder_weight
#                         scaled_lora_A = replacement_encoder_weight @ Lambda_d
#
#                         # Check dimensions before applying to the module
#                         print("Scaled lora B shape:", scaled_lora_B.shape)
#                         print("Scaled lora A shape:", scaled_lora_A.shape)
#
#                         replace_module_weights(target.lora_B.default, scaled_lora_B)
#                         replace_module_weights(target.lora_A.default, scaled_lora_A)
#
#                         target.default_lora_latent_mapping = torch.nn.Linear(lora_config.r, lora_config.r, bias=False)
#                         init_module_weights(target.default_lora_latent_mapping, sigma=0.00001)
#                         target.default_lora_latent_mapping.to(device)
#
#                         target.lora_A.default.weight.requires_grad = True  # only the r*r matrix will be tuned
#                         target.lora_B.default.weight.requires_grad = True  # only the r*r matrix will be tuned
#
#                     else:
#                         init_module_weights(target.lora_A.default, sigma=0.00001)
#
#             else:
#                 raise NotImplementedError("The only supported mode is: separated.")
#     if not is_target_modules_in_base_model:
#         raise ValueError(
#             f"Target modules {lora_config.target_modules} not found in the base model. "
#             f"Please check the target modules and try again."
#         )

def find_and_initialize(model, peft_config, adapter_name, reconstr_type, reconstruct_config, writer):
    """
    :param adapter_name: options: 'default'
    :param reconstr_type: options: 'svd'
    """
    half_init_dec = reconstruct_config['half_init_dec']
    replacement_module_random_init = reconstruct_config['replacement_module_random_init']
    reconstruction_mode = reconstruct_config['reconstr_mode']
    lora_config = peft_config[adapter_name]
    r_squared = reconstruct_config['r_squared']  # whether using r*r matrix between lora_A and lora_B or not
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    if loaded_in_8bit and not is_bnb_available():
        raise ImportError(
            "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
            "You can install it with `pip install bitsandbytes`."
        )
    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]
    assert (not isinstance(lora_config.target_modules, str))
    print("Iterating through model's specified modules to initialize A/B matrices.")
    for key in tqdm(key_list):
        target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
        if target_module_found:
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, target_name = _get_submodules(model, key)

            if reconstruction_mode == 'separated':
                replacement_encoder_weight, replacement_decoder_weight = get_replacement_module(weight=target.weight.T,
                                                                                                module_name=key,
                                                                                                type=reconstr_type,
                                                                                                writer=writer,
                                                                                                reconstruct_config=reconstruct_config)

                if not isinstance(target, peft.tuners.lora.Linear):
                    raise NotImplementedError('Only initialization for peft.tuners.lora.Linear type is implemented.')
                    # TODO implement for Linear8bitLt
                else:
                    if half_init_dec:
                        kaiming_uniform_init_lower_half(replacement_decoder_weight)
                    if replacement_module_random_init:
                        kaiming_uniform_init(replacement_encoder_weight)
                        kaiming_uniform_init(replacement_decoder_weight)
                    replace_module_weights(target.lora_B.default, replacement_decoder_weight.T)
                    if r_squared:
                        target.forward = types.MethodType(forward_latent, target)
                        target.get_delta_weight = types.MethodType(get_delta_weight, target)
                        replace_module_weights(target.lora_A.default, replacement_encoder_weight.T)
                        target.default_lora_latent_mapping = torch.nn.Linear(lora_config.r, lora_config.r, bias=False)
                        init_module_weights(target.default_lora_latent_mapping, sigma=0.00001)
                        target.default_lora_latent_mapping.to(target.lora_A.default.weight.device)
                    
                        target.lora_A.default.weight.requires_grad = False  # only the r*r matrix will be tuned
                        target.lora_B.default.weight.requires_grad = False  # only the r*r matrix will be tuned
                   
                   
                    # if r_squared:
                    #     # 替换 forward 和 get_delta_weight 方法
                    #     target.forward = types.MethodType(forward_latent, target)  # 使用多头前向传播方法
                    #     target.get_delta_weight = types.MethodType(get_delta_weight, target)

                    #     # 替换 lora_A 的默认权重为重建的 encoder 权重 (转置)
                    #     replace_module_weights(target.lora_A.default, replacement_encoder_weight.T)

                    #     # 初始化原本的 r*r 映射
                    #     target.default_lora_latent_mapping = torch.nn.Linear(lora_config.r, lora_config.r, bias=False)
                    #     init_module_weights(target.default_lora_latent_mapping, sigma=0.00001)
                    #     target.default_lora_latent_mapping.to(target.lora_A.default.weight.device)
                    #     num_heads = 7
                    #     target.multihead_lora_mappings = torch.nn.ModuleList([
                    #         torch.nn.Linear(lora_config.r, lora_config.r, bias=False) for _ in range(num_heads)
                    #     ])

                    #     # 初始化多头矩阵的权重
                    #     for head in target.multihead_lora_mappings:
                    #         init_module_weights(head, sigma=0.00001)
                    #         head.to(target.lora_A.default.weight.device)

                    #     # 冻结 lora_A 和 lora_B 的权重，仅调整多头矩阵和 default_lora_latent_mapping
                    #     target.lora_A.default.weight.requires_grad = False  # 仅训练多头的 r*r 矩阵
                    #     target.lora_B.default.weight.requires_grad = False  # 仅训练多头的 r*r 矩阵
                        
                    else:
                        init_module_weights(target.lora_A.default, sigma=0.00001)

            else:
                raise NotImplementedError("The only supported mode is: separated.")

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )

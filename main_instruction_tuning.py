# Code based on https://github.com/GraphPKU/PiSSA script with minimal changes.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import threading
import copy
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import MistralForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import cairosvg
from PIL import Image
import io
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import clip
import yaml
import json
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal
import peft
import torch
import transformers
from transformers import Trainer, set_seed
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from utils.initialization_utils import find_and_initialize
from transformers import AutoModelForCausalLM, AutoConfig
import torch
import torch.nn as nn
from transformers import MistralForCausalLM, MistralConfig
from transformers import Cache
import concurrent.futures
import signal

IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="/home/liuzhe/new-files/AI-ModelScope/Mistral-7B-v0___1")
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default="", metadata={"help": "Path to the training data."})
    dataset_split: str = field(
        default="train", metadata={"help": "(`['train', 'test', 'eval']`):"}
    )
    dataset_field: List[str] = field(
        default="caption code", metadata={"help": "Fields of dataset input and output."}
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048, metadata={
        "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}, )
    lora_r: int = field(default=128, metadata={
        "help": "The rank of the low-rank adapter."})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # print('tokenizerd:',tokenizer.decode(labels))

    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def cosine_similarity(a, b):
    print(a.shape)
    print(b.shape)
    if a.device != b.device:
        b = b.to(a.device)  # Move b to the same device as a
    # Define a 1D convolutional layer to reduce the feature dimension
    # Reshape a for convolution: (batch_size, in_channels, sequence_length)
    # conv1d = nn.Conv1d(in_channels=4096, out_channels=512, kernel_size=1).to(a.device)
    # a = a.permute(0, 2, 1)  # Change shape to (batch_size, 4096, sequence_length)
    # a = self.conv1d(a)  # Apply convolution
    # a = a.permute(0, 2, 1)  # Change shape back to (batch_size, sequence_length, 512)
    # Compute cosine similarity

    similarity = F.cosine_similarity(a, b, dim=-1)
    return similarity.mean()

def get_decoded_svg_content(decoded_output: str) -> str:
    """
    提取字符串中第一个以 <svg> 开头和 </svg> 结尾的内容。

    :param decoded_output: 输入的字符串
    :return: 第一个 SVG 内容，如果未找到则返回空字符串
    """
    # 查找第一个 <svg> 和 </svg> 的位置
    start_index = decoded_output.find('<svg')
    end_index = decoded_output.find('</svg>', start_index)

    # 提取内容
    if start_index != -1 and end_index != -1:
        svg_content = decoded_output[start_index:end_index + len('</svg>')]
        return svg_content.strip()
    
    return "-"  # 如果未找到，返回空字符串
class TimeoutException(Exception):
    pass

# 超时处理函数
def timeout_handler(signum, frame):
    raise TimeoutException("Function execution exceeded time limit.")

def svgTopng(svg_text,unsafe=True):
    # 设置超时处理
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3)  

    try:
        png_data = cairosvg.svg2png(bytestring=svg_text)
        # Create an image from the PNG data
        image = Image.open(io.BytesIO(png_data))
        # Create a new image with a light gray background
        background = Image.new('RGB', image.size, (240, 240, 240))  # Light gray color
        background.paste(image, (0, 0), image)
        return background
    finally:
        signal.alarm(0)  # 取消超时
class CustomMistralForCausalLM(MistralForCausalLM):

    # def __init__(self, config):
    #     super().__init__(config)
    #     # Define a 1D convolutional layer
    #     # self.clip_linear_layer = nn.Linear(512, 4096)
    #     # self.clip_linear_layer = nn.Conv1d(in_channels=4096, out_channels=512, kernel_size=1)
    #     self.align_layer= buildMLP(4096,1024,512)
    #     # Initialize weights and apply final processing
    #     self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        # print("Hidden states size:", hidden_states.shape)

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()
        loss = None
        if labels is not None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                '/home/liuzhe/new-files/AI-ModelScope/Mistral-7B-v0___1',
                model_max_length=2048,
                padding_side="right",
                use_fast=True,
            )
                # 创建一个新变量，过滤掉 -100
            filtered_labels = [label for label in labels.tolist()[0] if label != -100]
                # print("input_ids",print(input_ids))
            svg_text = tokenizer.decode(filtered_labels).replace('</s>',"")
            new_svg_text = tokenizer.decode(filtered_labels)

            print("label",new_svg_text)
            # input_id_decode = tokenizer.decode(f_input_ids).replace('</s>',"")
            # print("Input",input_id_decode)

            # 2. 加载并预处理图像
            image = clip_preprocess(svgTopng(svg_text)).unsqueeze(0).to("cuda")
            # Encode the image to get the image features
            label_embeddings = clip_model.encode_image(image) 
#           获得svg last_hidden_state
            # svg_inputs = tokenizer(svg_text, return_tensors="pt").to("cuda")
            # svg_hidden_outputs = self.model(**svg_inputs)
            # svg_hidden_states = svg_hidden_outputs.last_hidden_state

            # align_hidden_states=self.align_layer(svg_hidden_states)
            # similarity = cosine_similarity(align_hidden_states,image_embeddings)
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()

            predicted_token_ids = logits.argmax(dim=-1)  # 获取整个序列的预测 token IDs
            decoded_output = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
            res_svg = get_decoded_svg_content(decoded_output)

            try:
                print("res_svg",res_svg)
                # 尝试将 SVG 转换为 PNG
    
                res_img = clip_preprocess(svgTopng(res_svg)).unsqueeze(0).to("cuda")
                res_embeddings = clip_model.encode_image(res_img)

                loss = loss_fct(shift_logits, shift_labels)*0.7 +(1 - cosine_similarity(res_embeddings, label_embeddings))*0.3
            except Exception as e:
                print(f"Error : {e}")
                loss = loss_fct(shift_logits, shift_labels)
            # loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Now you can use the model for training or inference
def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    # script_args = TrainingArguments(
    #     model_name_or_path="/home/liuzhe/new-files/AI-ModelScope/Mistral-7B-v0___1",
    #     output_dir="output_32",
    #     lora_r=128,
    #     data_path="/home/liuzhe/new-files/LoRA-XS/utils/dataset-1024-everypath-10-26.json",
    #     dataset_split="train",
    #     dataset_field=['caption', 'code'],
    #     num_train_epochs=4,
    #     per_device_train_batch_size=1,
    #     gradient_accumulation_steps=1,
    #     save_strategy="steps",
    #     save_steps=48,
    #     save_total_limit=20,
    #     learning_rate=4e-3,
    #     weight_decay=0.,
    #     warmup_ratio=0.02,
    #     lr_scheduler_type="cosine",
    #     logging_steps=1,
    #     bf16=False,
    #     tf32=False,
    #     fp16=True,
    # )
    set_seed(script_args.seed)

    model = CustomMistralForCausalLM.from_pretrained(
        script_args.model_name_or_path, device_map="auto"
    )

    if script_args.lora_r is not None:
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    else:
        raise ValueError("LoRA rank should be provided.")

    now = datetime.datetime.now()
    now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

    adapter_name = "default"
    peft_config_dict = {adapter_name: lora_config}

    with open("config/reconstruct_config.yaml", 'r') as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    reconstr_type = reconstr_config['reconstruction_type']
    reconstr_config[reconstr_type]['rank'] = peft_config_dict[adapter_name].r

    script_args.output_dir = f"{script_args.output_dir}/{script_args.model_name_or_path}/" \
                             f"{script_args.data_path}_split_{script_args.dataset_split}/" \
                             f"LoRA_init_{reconstr_type}_rank_{peft_config_dict[adapter_name].r}_lr_" \
                             f"{script_args.learning_rate}_seed_{script_args.seed}/output_{now}"
    os.makedirs(script_args.output_dir)

    with open(os.path.join(script_args.output_dir, 'reconstr_config.json'), 'w') as fp:
        json.dump(reconstr_config, fp)

    find_and_initialize(model, peft_config_dict, adapter_name=adapter_name, reconstr_type=reconstr_type,
                        writer=None, reconstruct_config=reconstr_config)
    

    # print(model)
    # for name, module in model.named_modules():

    #     if isinstance(module,peft.tuners.lora.Linear):
    #     # Check if the module has the clip_conv1d attribute
    #         if hasattr(module, 'clip_conv1d'):
    #             clip_conv1d_layer = module.clip_conv1d
    #             print(f"Found clip_conv1d in module: {name}")
    #             print(clip_conv1d_layer)

    for param in model.parameters():
        param.data = param.data.contiguous()
    model.print_trainable_parameters()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    raw_train_datasets = load_dataset("json",data_files=script_args.data_path,split='train')
        # remove_columns=raw_train_datasets.column_names,
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=1,
        num_proc=32,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0],
                   "response": script_args.dataset_field[1]}
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)

    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir, 'ft'))
if __name__ == "__main__":
    train()

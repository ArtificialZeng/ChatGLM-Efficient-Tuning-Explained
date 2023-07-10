import torch

from typing import Dict, Optional, Sequence, Union  #导入Python的类型提示模块，这些类型会在函数定义中用来说明参数和返回值的类型。

from transformers import DataCollatorWithPadding, BatchEncoding  #从transformers库中导入DataCollatorWithPadding和BatchEncoding。#DataCollatorWithPadding是一个数据整理器，用于将一批样本组合在一起并进行填充以形成一个统一的批量；BatchEncoding是一个类，用于保存由tokenizer返回的编码结果。
from transformers.modeling_utils import PreTrainedModel   #从transformers库中导入PreTrainedModel，表示预训练的模型。
from transformers.tokenization_utils import PreTrainedTokenizer  #从transformers库中导入PreTrainedTokenizer，表示预训练的分词器。

from .other import IGNORE_INDEX  ##从同级目录下的other模块中导入IGNORE_INDEX，它是一个特殊的索引值，用于在损失计算时忽略某些特定的标记。

#新的类DataCollatorForChatGLM，它继承自DataCollatorWithPadding，这意味着它拥有DataCollatorWithPadding的所有方法和属性，并可以添加或修改某些方法和属性。
class DataCollatorForChatGLM(DataCollatorWithPadding):
    r"""
    Data collator for ChatGLM. It is capable of dynamically padding for batched data.
    """
    def __init__( #类的构造函数，它在类的实例化时被调用。构造函数接收预训练的分词器，模型，和两个可选参数ignore_pad_token_for_loss和use_v2。
            self,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            ignore_pad_token_for_loss: Optional[bool] = False,
            use_v2: Optional[bool] = False
    ):
        super().__init__(tokenizer, padding=True)
        self.model = model
        self.label_pad_token_id = IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id
        if use_v2:
            self.get_attention_masks = self.get_attention_masks_v2
            self.get_position_ids = self.get_position_ids_v2
        else:
            self.get_attention_masks = self.get_attention_masks_v1
            self.get_position_ids = self.get_position_ids_v1

    def get_attention_masks_v1(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates attention masks for left-padded sequences.

        Note that ChatGLM assigns False on token to be attended in attention mask. In general settings, it should be True.

        According to: https://huggingface.co/THUDM/chatglm-6b/blob/v1.1.0/modeling_chatglm.py#L680
        """
        batch_size, seq_length = input_ids.size()
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()

        for i, seq in enumerate(input_ids):
            attention_mask[i, :, :(seq == self.tokenizer.bos_token_id).nonzero()[0].item()] = 1 # context
            attention_mask[i, :, :(seq != self.tokenizer.pad_token_id).nonzero()[0].item()] = 0 # padding

        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()
        return attention_mask

    def get_position_ids_v1(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates position ids for left-padded sequenes.

        According to: https://huggingface.co/THUDM/chatglm-6b/blob/v1.1.0/modeling_chatglm.py#L692
        """
        batch_size, seq_length = input_ids.size()
        mask: int = self.model.config.mask_token_id
        gmask: int = self.model.config.gmask_token_id
        position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
        block_position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)

        for i, seq in enumerate(input_ids):
            mask_token = gmask if gmask in seq else mask
            context_length = (seq == self.tokenizer.bos_token_id).nonzero()[0].item()
            padding_length = (seq != self.tokenizer.pad_token_id).nonzero()[0].item()
            position_ids[i, padding_length:] = torch.arange(
                seq_length - padding_length,
                dtype=torch.long,
                device=device
            )
            if self.model.position_encoding_2d or (mask_token != gmask): # 2d position encoding or not gMASK
                position_ids[i, context_length:] = (seq == mask_token).nonzero()[0].item() - padding_length # mask position
            block_position_ids[i, context_length:] = torch.arange(
                seq_length - context_length,
                dtype=torch.long,
                device=device
            ) + 1

        if self.model.position_encoding_2d:
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)

        return position_ids

    def get_attention_masks_v2(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates attention masks for left-padded sequences.
        """
        batch_size, seq_length = input_ids.size()
        attention_mask = torch.ones((batch_size, seq_length), device=device)

        for i, seq in enumerate(input_ids):
            attention_mask[i, :(seq != self.tokenizer.pad_token_id).nonzero()[0].item()] = 0 # padding

        return attention_mask

    def get_position_ids_v2(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates position ids for left-padded sequenes.
        """
        batch_size, seq_length = input_ids.size()
        position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)

        for i, seq in enumerate(input_ids):
            padding_length = (seq != self.tokenizer.pad_token_id).nonzero()[0].item()
            position_ids[i, padding_length:] = torch.arange(seq_length - padding_length, dtype=torch.long, device=device)

        return position_ids

    def __call__(self, features: Sequence[Dict[str, Union[torch.Tensor, Sequence[int]]]]) -> BatchEncoding:
        r"""
        Pads batched data to the longest sequence in the batch.

        We adopt left-padding in both training and evaluation.
        """
        if isinstance(features[0]["input_ids"], torch.Tensor):
            input_ids = [feature["input_ids"].clone().detach().flip(0) for feature in features]
        else:
            input_ids = [torch.tensor(feature["input_ids"]).flip(0) for feature in features]

        if "labels" in features[0]:
            if isinstance(features[0]["labels"], torch.Tensor):
                labels = [feature["labels"].clone().detach().flip(0) for feature in features]
            else:
                labels = [torch.tensor(feature["labels"]).flip(0) for feature in features]
            input_ids = input_ids + labels # pad them to the same length

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).flip(-1)

        batch = {}

        if "labels" in features[0]:
            input_ids, labels = input_ids.split(len(features), dim=0)
            labels = torch.where(labels != self.tokenizer.pad_token_id, labels, self.label_pad_token_id)
            batch["labels"] = labels

        batch["input_ids"] = input_ids
        batch["attention_mask"] = self.get_attention_masks(input_ids, device=input_ids.device)
        batch["position_ids"] = self.get_position_ids(input_ids, device=input_ids.device)

        return BatchEncoding(batch)

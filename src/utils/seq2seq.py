import os
import json
import torch
import numpy as np
import torch.nn as nn
from dataclasses import dataclass  #导入Python的数据类（dataclass）模块，该模块提供了一种装饰器和函数来自动添加特殊方法到用户定义的类，包括 __init__，__repr__等。
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from transformers.trainer import PredictionOutput  #从transformers库中导入PredictionOutput，这是一个存储预测输出的数据类。
from transformers.tokenization_utils import PreTrainedTokenizer  #从transformers库中导入PreTrainedTokenizer，用于处理预训练模型的tokenization。

import jieba：导入jieba模块，用于进行中文分词。

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  #从nltk库的translate.bleu_score模块中导入sentence_bleu和SmoothingFunction，用于计算BLEU评价指标。

from .peft_trainer import PeftTrainer  #导入PeftTrainer，这是一个用于训练模型的类。

from .other import get_logger, IGNORE_INDEX  #从同目录下的other模块中导入get_logger和IGNORE_INDEX。


logger = get_logger(__name__)  #使用get_logger函数创建一个日志器。


@dataclass  #用数据类装饰器装饰下面的类。
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqTrainerForChatGLM.

    Borrowed from: https://github.com/THUDM/ChatGLM-6B/blob/0c2806fea82683349194e21996dd6b3acc3c265b/ptuning/main.py#L307
    """

    tokenizer: PreTrainedTokenizer  #为类定义一个成员，名为tokenizer，它是一个PreTrainedTokenizer。

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""  #定义了类的__call__方法，使得这个类的实例可以像函数一样被调用，输入参数是一个序列，这个序列的元素可以是numpy数组或者是包含numpy数组的元组。
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds  #从输入的eval_preds中提取预测值和标签。
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}  #初始化一个字典，用于存储评价指标的得分。

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id) #如果预测或者标签中的某个token不是忽略索引，则保持原样，否则替换为填充token的id。

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)#使用tokenizer的batch_decode方法将预测和标签的token id解码为原始文本。

        for pred, label in zip(decoded_preds, decoded_labels): #遍历预测值和标签。
            hypothesis = list(jieba.cut(pred))  #使用jieba对预测和标签进行分词。
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0: #如果预测或者标签为空，则设置rouge-1，rouge-2和rouge-l的得分为0.
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items(): #将rouge的得分放入score_dict。
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3) #计算bleu得分。
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))  #将bleu的得分放入score_dict。

        return {k: float(np.mean(v)) for k, v in score_dict.items()}  #返回每个评价指标的平均得分。


class Seq2SeqTrainerForChatGLM(PeftTrainer): #定义一个名为Seq2SeqTrainerForChatGLM的类，这个类继承自PeftTrainer类。

    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def prediction_step( #定义一个名为prediction_step的方法，这个方法接收四个参数：一个模型，一个输入字典，一个布尔值用于决定是否只计算损失，以及一个可选的字符串列表用于忽略某些键。
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        input_ids = inputs["input_ids"] #从输入中提取输入id。
        loss, generated_tokens, labels = super().prediction_step(  #调用父类的prediction_step方法，计算损失，生成tokens，并获取标签。
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        #如果generated_tokens不为None，那么取其第二维度的从input_ids.size(-1)开始之后的部分。
        generated_tokens = generated_tokens[:, input_ids.size(-1):] if generated_tokens is not None else None
        return (loss, generated_tokens, labels)  #返回损失，生成的tokens，和标签。
    
    def save_predictions(  #定义一个名为save_predictions的方法，这个方法接收一个预测结果并且不返回任何值。
            self,
            predict_results: PredictionOutput
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero(): #如果当前的进程不是主进程，直接返回。
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl") #定义预测结果的保存路径。
        logger.info(f"Saving prediction results to {output_prediction_file}")  #打印日志，告诉用户预测结果将被保存到何处。
        
        #如果预测或者标签中的某个token不是忽略索引，则保持原样，否则替换为填充token的id。
        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        #使用tokenizer的batch_decode方法将预测和标签的token id解码为原始文本。
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer: #打开预测文件，准备写入预测结果。
            res: List[str] = [] #定义一个空列表，用于保存预测结果。
            for pred, label in zip(decoded_preds, decoded_labels):  #遍历预测值和标签。
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))  #将标签和预测值以字典的形式转化为json字符串，然后添加到res列表中。
            writer.write("\n".join(res))  #将res列表中的所有元素用换行符连接起来，然后写入到预测文件中。

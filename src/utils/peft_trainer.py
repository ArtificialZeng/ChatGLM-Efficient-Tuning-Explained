#彩色版：https://zengxiaojian.blog.csdn.net/article/details/131489790
#ChatGLM源码解析系列（专栏）：https://blog.csdn.net/sinat_37574187/category_12365053.html

import os
import json
import time
import torch
from typing import Dict, Optional
from datetime import timedelta

from transformers import (
    Seq2SeqTrainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments
)

from transformers.trainer import TRAINING_ARGS_NAME
from transformers.modeling_utils import unwrap_model

from .config import FinetuningArguments

from .other import (
    get_logger,
    get_state_dict,
    load_trainable_params,
    load_valuehead_params,
    FINETUNING_ARGS_NAME,
    VALUE_HEAD_FILE_NAME
)


logger = get_logger(__name__)  #此行代码使用logger的工厂函数get_logger，使用了传入的__name__（也就是当前模块名）作为logger的名称。

#这段代码中定义了两个类，一个是LogCallback，另一个是PeftTrainer。
class LogCallback(TrainerCallback):
    r"""
    TrainerCallback includes the state function during training, for more details refer to the TrainerCallback class.
    The on_log function primarily collects process parameters during training, such as training loss, learning rate,
    and training epochs, as well as progress parameters like the current percentage progress and estimated remaining
    time. Every time a log is triggered, a new record is appended to the file "messages.log" for dynamic visualization
    purposes.
    """

    def __init__(self):
        self.start_time = time.time() #在LogCallback的构造函数中，我们记录了该实例创建的时间，作为训练的开始时间。

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None: #这是一个在每次日志记录后被调用的函数，它包含了训练参数、训练状态和训练控制等信息。
        r"""                                                                                                  #然后根据这些信息进行一些操作，比如计算训练进度、剩余时间等，并将这些信息存储在日志文件中。
        Event called after logging the last logs.
        """
        if "loss" not in state.log_history[-1]:
            return
        cur_time = time.time()
        cur_steps = state.log_history[-1].get("step")
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_steps = state.max_steps - cur_steps
        remaining_time = remaining_steps * avg_time_per_step
        log_dict = {
            "current_steps": cur_steps,
            "total_steps": state.max_steps,
            "loss": state.log_history[-1].get("loss", None),
            "reward": state.log_history[-1].get("reward", None),
            "learning_rate": state.log_history[-1].get("learning_rate", None),
            "epoch": state.log_history[-1].get("epoch", None),
            "percentage": round(cur_steps / state.max_steps * 100, 2) if state.max_steps != 0 else 100,
            "elapsed_time": str(timedelta(seconds=int(elapsed_time))),
            "remaining_time": str(timedelta(seconds=int(remaining_time)))
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "trainer_log.jsonl"), "a") as f:
            f.write(json.dumps(log_dict) + "\n")


class PeftTrainer(Seq2SeqTrainer):  #定义了一个新的类PeftTrainer，它继承了Seq2SeqTrainer类。
    r"""
    Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
    """

    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):  #在PeftTrainer的构造函数中，我们接受一个FinetuningArguments实例作为输入，该实例包含了一些微调模型所需的参数。
        super().__init__(**kwargs)   #在函数内部，我们检查是否存在前一个训练过程的日志文件，如果存在则删除。
        self.finetuning_args = finetuning_args
        #此段代码在 PeftTrainer 构造函数内部。它首先检查当前进程是否为主进程（对于并行训练来说，只有主进程负责保存模型和日志等）。然后它检查是否存在前一个训练过程的日志文件，如果存在则发出警告并删除。
        if self.is_world_process_zero() and os.path.exists(os.path.join(self.args.output_dir, "trainer_log.jsonl")):
            logger.warning("Previous log file in this folder will be deleted.")
            os.remove(os.path.join(self.args.output_dir, "trainer_log.jsonl"))
    
    #这是一个保存模型参数的方法，它会在指定的输出目录中保存训练后的模型。
    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""   
        Saves trainable parameters as model checkpoint.

        This function will only be executed at the process zero.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        #这行代码的功能是提取模型中的主要组件，移除一些可能包装模型的容器或者封装器（例如 DataParallel 或者 DistributedDataParallel）。
        model = unwrap_model(self.model)

        if hasattr(model, "pretrained_model"): # for models with valuehead (currently using LoRA only)
            backbone_model = getattr(model, "pretrained_model")
            torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))
        else:
            backbone_model = model

        #这段代码根据微调类型（例如 "lora"）保存模型，其中 "lora" 是一种训练方式，其全称是"Layer-wise Optimizer Rate Adaptation"。
        if self.finetuning_args.finetuning_type == "lora":
            backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model))
        else: # freeze/full tuning
            backbone_model.config.use_cache = True
            backbone_model.save_pretrained(
                output_dir,
                state_dict=get_state_dict(backbone_model),
                safe_serialization=self.args.save_safetensors
            )
            backbone_model.config.use_cache = False
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
            f.write(self.args.to_json_string() + "\n")
        self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))
      
    #这是一个加载模型参数的方法，它会加载之前保存的最优模型。具体加载哪个模型，是根据最优模型的checkpoint路径来决定的。这个路径通常是在训练过程中，验证得分最高的模型的保存路径。
    def _load_best_model(self):
        r"""
        Loads trainable parameters from model checkpoint.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        #在开始加载模型前，会先记录一条日志，标明要加载的模型的路径和该模型的分数。
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")

        model = unwrap_model(self.model)
        backbone_model = getattr(model, "pretrained_model") if hasattr(model, "pretrained_model") else model

        #这段代码根据微调类型来加载模型，如果微调类型为 "lora"，则加载对应的模型。在加载模型时，需要知道模型的checkpoint路径和当前激活的模型适配器。
        if self.finetuning_args.finetuning_type == "lora":
            backbone_model.load_adapter(self.state.best_model_checkpoint, getattr(backbone_model, "active_adapter"))
            #这部分代码是加载"价值头"（value head）的参数，这些参数在之前保存的模型中。这里，"价值头"通常用于强化学习模型，用于预测某个动作的价值。
            if hasattr(model, "v_head") and load_valuehead_params(model, self.state.best_model_checkpoint):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })
        #如果微调类型不是 "lora"，则加载可以训练的参数。这里的 freeze/full-tuning 或 p_tuning 指的是不同的微调方式。freeze 指的是冻结一部分层只训练其他层，full-tuning 指的是全部层都参与训练，p_tuning 是一种部分参数调整策略。
        else: # freeze/full-tuning or p_tuning
            load_trainable_params(backbone_model, self.state.best_model_checkpoint)

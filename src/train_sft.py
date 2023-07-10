
# 彩色版：ChatGLM Efficient Tuning源码解析train_sft.py   https://zengxiaojian.blog.csdn.net/article/details/131458667

# coding=utf-8
# Implements several parameter-efficient supervised fine-tuning method for ChatGLM.
# This code is inspired by https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py


from utils import (
    DataCollatorForChatGLM,  #用于处理聊天模型ChatGLM的数据整理器。在模型训练和评估的过程中，数据整理器的作用是将多个数据样本整理成一个批量（batch），并准备好模型可以接受的输入。
    Seq2SeqTrainerForChatGLM,  #专门用于聊天模型ChatGLM的训练器，它在原始的Seq2SeqTrainer的基础上可能进行了一些针对ChatGLM模型的定制。
    ComputeMetrics,  #计算指标的工具。在模型训练和评估过程中，我们需要监控一些指标以了解模型的性能，如准确率，召回率，F1值等。
    LogCallback,  #一个回调函数，它用于在模型训练过程中的某些特定时间点执行特定任务，例如记录日志。
    load_pretrained,  #用于加载预训练的模型。在许多深度学习任务中，我们通常会先从一个预训练模型开始，然后对它进行微调，以适应特定的任务。
    prepare_args,  #用于准备训练参数。
    prepare_data,
    preprocess_data,  #用于对训练数据进行预处理，例如清洗、规范化、分词、编码等。
    get_logits_processor,  #返回一个对模型输出logits进行处理的处理器，例如温度缩放，顶部k采样，顶部p采样等。
    plot_loss
)


def main():

    # Prepare pretrained model and dataset
    #这行从 prepare_args 函数获取四个参数：模型参数（model_args），数据参数（data_args），训练参数（training_args），微调参数（finetuning_args）。其中 stage="sft" 是传递给 prepare_args 的参数。
    model_args, data_args, training_args, finetuning_args = prepare_args(stage="sft")
    dataset = prepare_data(model_args, data_args) #根据上述的模型和数据参数准备数据集。
    model, tokenizer = load_pretrained(model_args, finetuning_args, training_args.do_train, stage="sft") #加载了一个预训练的模型和对应的 tokenizer。
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="sft") #对加载的数据集进行预处理，这包括使用 tokenizer 对文本数据进行编码等操作。
    data_collator = DataCollatorForChatGLM(   #创建一个 DataCollator，用于在训练过程中处理/对齐批处理数据。
        tokenizer=tokenizer,
        model=model,
        ignore_pad_token_for_loss=(data_args.ignore_pad_token_for_loss and not training_args.predict_with_generate),
        use_v2=model_args.use_v2
    )

    # Override the decoding parameters of Seq2SeqTrainer
    #这行代码检查training_args.generation_max_length是否为None。如果它不为None，那么就保持不变；如果为None，那么就将其设为data_args.max_target_length的值。
    training_args.generation_max_length = training_args.generation_max_length if \
                training_args.generation_max_length is not None else data_args.max_target_length
  
    #检查data_args.eval_num_beams是否为None。如果它不为None，那么training_args.generation_num_beams就会被设为data_args.eval_num_beams的值；如果为None，那么training_args.generation_num_beams就保持不变。
    training_args.generation_num_beams = data_args.eval_num_beams if \
                data_args.eval_num_beams is not None else training_args.generation_num_beams

  
    #接下来的 if 代码块会根据 training_args.do_train 决定是否分割数据集，以及如何分割。如果进行训练，那么会根据设定的比例分割训练和验证数据集。如果只进行评估或预测，那么全集将作为评估数据集。
    # Split the dataset
    if training_args.do_train:
        if data_args.dev_ratio > 1e-6:
            dataset = dataset.train_test_split(test_size=data_args.dev_ratio)
            trainer_kwargs = {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            trainer_kwargs = {"train_dataset": dataset}
    else: # do_eval or do_predict
        trainer_kwargs = {"eval_dataset": dataset}

    # Initialize our Trainer
    trainer = Seq2SeqTrainerForChatGLM(  #通过上述的参数来创建一个 Seq2SeqTrainer。
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **trainer_kwargs
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = {
        "do_sample": True,
        "top_p": 0.7,
        "max_new_tokens": data_args.max_target_length + 1,
        "temperature": 0.95,
        "logits_processor": get_logits_processor()
    }

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

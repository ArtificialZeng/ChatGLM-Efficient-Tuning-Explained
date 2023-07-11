# ChatGLM-Efficient-Tuning-Explained

ChatGLM-Efficient-Tuning-相关代码，逐行详解版。


* [src/](./src)
  * [utils/](./src/utils)
    * [common.py（init_adapter（）、load_pretrained()、prepare_args()](./src/utils/common.py)
    * [peft_trainer.py](./src/utils/peft_trainer.py)
    * [data_collator.py（DataCollatorForChatGLM类）](./src/utils/data_collator.py)
    * [seq2seq.py](./src/utils/seq2seq.py)
  * [train_sft.py(DataCollatorForChatGLM、Seq2SeqTrainerForChatGLM)](./src/train_sft.py)
* [examples/](./examples)
  * [ads_generation.md](./examples/ads_generation.md)
* [README.md](./README.md)



# CSDN彩色博客版：
* [src/](./ChatGLM-Efficient-Tuning-Explained/src)
  * [utils/](./ChatGLM-Efficient-Tuning-Explained/src/utils)
    * [common.py](./ChatGLM-Efficient-Tuning-Explained/src/utils/common.py)
    * [peft_trainer.py](./ChatGLM-Efficient-Tuning-Explained/src/utils/peft_trainer.py)
  * [CSDN彩色源码解析train_sft.py](https://zengxiaojian.blog.csdn.net/article/details/131458667)
* [README.md](./ChatGLM-Efficient-Tuning-Explained/README.md)

ChatGLM Efficient Tuning源码解析train_sft.py   https://zengxiaojian.blog.csdn.net/article/details/131458667


## 引用 - 源项目

```bibtex
@Misc{chatglm-efficient-tuning,
  title = {ChatGLM Efficient Tuning},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/ChatGLM-Efficient-Tuning}},
  year = {2023}
}
```

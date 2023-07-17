# ChatGLM-Efficient-Tuning-Explained
注：2023.7月以后，原作者更改了目录结构。 我的这个源码解释版本是2023.7月以前的。原来的目录结构可以根据该项目的Fork日期和commit仔细查看；
最开始 utils 在根目录， 前几天还在/src/utils ，utils文件下有common.py、init等等。 现在我都有点找不到了

新的2023.7月该项目的源码解释，可以在这里看。

ChatGLM-Efficient-Tuning-相关代码，逐行详解版。


* [src/](./src)
  * [utils/](./src/utils)
    * [common.py](./src/utils/common.py)
      * init_adapter（）
      * load_pretrained()
      * prepare_args()
    * [peft_trainer.py  （定义LogCallback、PeftTrainer）](./src/utils/peft_trainer.py)
    * [data_collator.py（DataCollatorForChatGLM类）](./src/utils/data_collator.py)
    * [seq2seq.py  （ComputeMetrics、Seq2SeqTrainerForChatGLM)](./src/utils/seq2seq.py)
  * [train_sft.py（导入DataCollatorForChatGLM、Seq2SeqTrainerForChatGLM)](./src/train_sft.py)
* [examples/](./examples)
  * [ads_generation.md（分布式运行范例）](./examples/ads_generation.md)
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

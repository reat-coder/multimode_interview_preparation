### 目前开源的主流LLM models
- prefix Decoder
  - 输入是双向注意力，输出是单向注意力
  - chatGLM, chatGLM2, U-PaLM
- causal Decoder
  - 从左到右的单向注意力
  - LLaMA-7B, LLaMa衍生
- Encoder-Decoder
  - 输入双向注意，输出单向注意
  - T5, Flan_T5, BART
- **三者区别：**在于注意力机制（attention mask）不同
  - Encoder-Decoder:
    - 适用偏理解的NLP任务，在长文本生成任务效果较差，效率较低
    - 输入上采用双向注意力，对问题的编码理解更充分
  - causal Decoder
    - 回归语言模型，预训练和下游应用是完全一致的，严格遵守只有后面的token才能看到前面的token的规则
    - 适用：生成文本任务效果好，训练效率高，zero-shot能力更强，具有涌现能力
  - prefix Decoder
    - prefix部分的token互相能看到，causal-D and ED的折中
    - 训练效率低
### 大模型训练目标
- **预测下一个词**训练目标：最大似然函数
- **去噪自编码器**（随机替换掉一些文本，训练模型恢复被打乱的文本段），目标函数：不认识》。《

### ?????? 涌现能力原因
- 任务的评价指标不够平滑
- 复杂任务&子任务随着模型增长


### LLMs简单介绍

### LLMs 后的175B 60B 540B指什么？
参数的个数，billion

### LLMs优点
- 利用大量无标注数据训练通用模型，用少量标注数据微调模型适应特殊任务。这种预训练和微调的方法可以减少数据标注时间和成本，提高模型泛化能力。
- 利用生成式GAN AI生成图片、文本、音乐等，解决用户在不同领域的需求
- 利用涌现能力完成以前完成不了的任务：数学应用题，常识推理，符号操作

### LLMs缺点
- 大量训练和运行的资源需求
- 数据质量和安全性问题，数据偏见、数据泄露、数据滥用
- 可解释性、可靠性、可持续性，理解和控制模型行为、保证模型正确性和稳定性，如何平衡模型的效益和风险

### layer normalization 层归一化
是深度学习中的一种归一化神经网络层输入的技术，layer norm公式包括：平均数$\mu$，方差$\sigma$，预测值$y$ \
**公式？**

### RMS norm
**公式**

### layer norm vs RMS norm
- RMS norm 简化了 layer norm, 去除掉计算平均值进行平移的部分
- 对比LN, RMS norm计算速度更快，效果相当，略有提升

### deep norm


### 各个模型采用归一化方法
|GPT3       |Pre layer Norm   |
|LLaMa      |pre RMS norm     |
|baichuan   | pre RMS norm    |
|chatGLM-6B |post deep norm   |
|chatGLM2-6B| post RMS norm   |
|Bloom      | pre layer norm  |
|Falcon     |pre layer norm   |

### 激活函数
类型，公式
各模型使用激活函数
- FFN
- GeLU
- Swich

### attention
- **传统attention问题**
  - 上下文长度约束
  - 传统速度慢，内存占用大
- **attention**优化方向
  - 上面的反面
- attention 变体
  - 稀疏。稀疏偏差引入attention机制降低了复杂性
  - 线性化
  - 原型和内存压缩。减少注意力矩阵<---通过减少查询或键值记忆对的数量
  - 低阶self-attention
  - attention 与先验，采用先验attention分布来补充或替代attention
  - 改进多头机制
 - 不会影响训练过程，训练速度，但是会引起非常细微的模型效果损失
- 推理过程：反复加载巨大的KV cache， 导致内存开销大，性能是内存受限
### multi- query-attention(多查询注意力机制）
传统注意力机制，没个头都有自己的query, key, and value.多个头共享相同的键和值向量，只保留不同的query vector
- **多头注意力机制multi-head-attention**:没个头上都有各自的query, key, value
- MQA：在多个头上共享key,value

### grouped query attention

### flash-attention

### 并行transformer block
优点好处

### 怎么用transformer加载bert模型
bert是什么：google2018提出的一种NLP的预训练模型，基于transformer架构，主要由多个堆叠的编码层组成，每个编码层包含自注意力机制和前馈神经网络两个主要部分。\
- 能从左到右，从右到左双向理解文本，比单向更容易捕捉语义
- 有较好的迁移学习能力
- 强大的特征提取能力

transformer要不要下载
具体代码&代码逻辑

### 如何用 transformer输出bert指定hidden_state
bert默认12层，预训练有时候不需要全部利用，只需要训练前面几层，how:
- bert-base-uncased模型目录下config.json文件中output_hidden_states可以设置编码器内隐藏层层数

### bert获取最后一层或者每一层网络的向量输出
- transformer最后一层
  - last_hidden_state，细看其中代码各个参数作用
- 获取每一层向量输出
  - outputs.last_hidden_state, 最后一层token vector
  - outputs.pooler_output,cls vector
  - hidden_state, 有13层，第一层0是输入的embedding向量，后1-12索引是每层的输出向量。
### 损失函数
##### KL (Kullback-Leibler)散度，衡量了两个概率分布间的差异，公式？
##### 交叉熵损失函数
衡量预测分布与实际标签分布的差异，可以用来衡量模型实际标签分布和预测分布之间的信息差。损失函数越大，差异越大，预测错误程度越大
- 两者区别：
  - KL散度是相对熵，越小表示分布越接近，KL散度值是非负数
  - 交叉熵，二分类问题常用损失函数
  - KL衡量两分布间差异的指标，交叉熵是KL散度的一种特殊形式，二分类问题中交叉熵只有一项，多分类问题有多项


### 多任务学习
多任务，损失差异大，动态调整损失权重，用任务特定损失函数，改变模型架构或引入正则化，平衡各任务贡献

### 五分类问题，用交叉熵损失函数不用均方差（MSE)

________lue_________


### GAN?
### LLMs复读机
what?why?How?
how?
- unlikelihood training（抑制重复词）
- 引入噪声（加入随机词or短语）
- repetition penalty（重复性惩罚）
- contrastive search
- beam search
- topk sampling
- nucleus sampler
- temperature
- 重复率指标检测
- 后处理和过滤
- 人工干预和控制
### llama，bert, chatglm模型怎么选择

### llama输入句子长度

### 各专业领域是否需要各自的大模型来服务

### 怎么让大模型处理更长的文本
- longchat
- 稀疏化
- moe
- mqa
- linear attention,减少复杂度

## 微调
### 在*模型上做全参数微调，要多少显存
- nB:16-20nG(不加载cpu)
- vicuna-7B, 4*A100 40G
- (FSDP,梯度累积，梯度检查点，降显存）


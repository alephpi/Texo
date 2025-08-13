# Datasets
- TexTeller
- img2latex

# Vision Encoders
- vit
- vitdet
- vary
- CLIP
- SigLIP
- ConvNeXt

# Decoding strategy
- Sequential:
    - GPT2
    - LLaMa3
    - Qwen2
- Tree-based:
  - relative tasks: [Semantic Parsing](https://paperswithcode.com/task/semantic-parsing), [Code Generation](https://paperswithcode.com/task/code-generation), 
  - [TreeGen](https://github.com/zysszy/TreeGen)
  - [TSDNet](https://github.com/zshhans/TSDNet)
- diffusion

# processing
- LaTeX normalizer: remove unsupported commands and simplify structure from given latex commands, e.g. https://github.com/OleehyO/TexTeller/blob/main/texteller/api/katex.py
- LaTeX to AST parser: use AST parser to further simplify the codes with equivalent expression.
- **Synthesize LaTeX expressions smartly (How?)** 

# Materials
- [KaTeX supported commands](https://katex.org/docs/supported.html)
- [MathJax supported commands](https://www.onemathematicalcat.org/MathJaxDocumentation/TeXSyntax.htm)

# LaTeX-OCR relevant project
- [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)
- [TexTeller](https://github.com/OleehyO/TexTeller)
- [MixTeX](https://github.com/RQLuo/MixTeX-Latex-OCR)
- [GOT-OCR-2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [UniMERNet](https://github.com/opendatalab/UniMERNet)

# LaTeX parsers
- [pylatexenc](https://github.com/phfaist/pylatexenc)
- [plastex](https://github.com/plastex/plastex)
- [KaTeX](https://github.com/KaTeX/KaTeX/blob/main/src/Parser.js)
- [mathjax](https://github.com/mathjax/MathJax-src/blob/master/ts/input/tex/TexParser.ts)

# Misc
- [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr)
- [chineseocr_lite](http://github.com/DayBreak-u/chineseocr_lite): a dbnet 0.9M(text region detection), anglenet 0.19M(text orientation detection), crnn_lite_lstm 1.31M(text recognition).
- [paddle to pytorch](https://zhuanlan.zhihu.com/p/335753926)

# Design
- A [blog](https://lilianweng.github.io/posts/2022-06-09-vlm/) that explains 4 types of VLM paradigm, including the two choices we have here: 
  1. learn to embed image to a prefix prompt and use decoder-only model
  2. cross-attend the image embedding, like a general machine translation task

Except for GOT-OCR2.0, which uses linear layer to project image embedding to Language decoder embedding space as a prefix prompt, all the other models implements the cross-attention paradigm and all of them has a HF `VisionEncoderDecoderModel` implementation.

Since we focus on the math expression recognition, UniMERNet is a good start point.

# Stats
| model   | MixTeX      | MixTeX2     | TexTeller3     | GOT-OCR-2.0    | UniMERNet-B    | UniMERNet-S   | UniMERNet-T   | PPFormulaNet-S   | PPFormulaNet-Plus-M |
| ------- | ----------- | ----------- | -------------- | -------------- | -------------- | ------------- | ------------- | ---------------- | ------------------- |
| encoder | SwinT 27M   | SwinT 48M   | ViT 86M        | ViTDet 95M     | SwinT 103M     | SwinT 58M     | SwinT  25M    | PPHGNetV2-B4 20M | PPHGNetV2-B6 75M    |
| decoder | Roberta 58M | GPT2 80M    | Roberta 211M   | Qwen2 463M     | MBart 221M     | MBart 144M    | MBart 81M     | MBart 44M        | MBart 78M           |
| lm head | 略          | 30K*768=23M | 15K*1024=15.4M | 152K*1024=155M | 50K*1024=51.2M | 50K*768=38.4M | 50K*512=25.6M | 50K*384=19.2M    | 50K*512=25.6M       |
| total   | 85M         | 128M        | 298M           | 560M           | 325M           | 202M          | 107M          | 64M              | 154M                |

where TexTeller is based on TrOCR, UniMERNet is based on Donut, which means they have similar architecture and differs only on training data.
After reading all the paper or readme file, in my intuition, the MER ability of these models should have the following rank:
GOT > PPFormulaNet-L ~ UniMERNet-B > UniMERNet-S ~ PPFormulaNet-S ~ TexTeller3 > UniMERNet-T > MixTeX
Especially, when CMD paper demonstrates the mismatch between the BLEU score and the MER performance, it convinces me that the actual difference between large model and its small variety would be even marginal.

First of all, apparently by looking at the table, we may immediately find out that, except for GOT where general OCR ability is emphasized, the other models hold a far bigger vocabulary than necessary for MER - even 15K is way much than need, e.g. for im2LaTeX(Deng) model, the vocab size is only ~500, this means we can greatly reduce the model size by removing a huge amount of unnecessary tokens from the vocabulary. (In fact the reason why these models use such a large vocabulary is because that they often inherit it from a large pretrained model).

The second possibility to reduce the model size is by reducing the hidden dimension of the model, e.g. from UniMERNet B to T, the model size is greatly reduced, without hurting too much the performance.

The main difference between various size of UniMERNet is the hidden dimension: 1024->768->512
The main difference between various size of PPFormulaNet is the hidden dimension and network depth: 384->512，decoder layers 2->6。

I suppose that due to the lack of paired img-latex data, the SOTA models(PPFormulaNet, UniMERNet and GOT) all need general OCR(instead of MathOCR) pretraining, which aims to force the encoder pay attention to the fine-grained features on optical characters. And then the MathOCR finetuning is to specialize its ability on ME. The objective of the pretraining and the finetuning is the same, but for pretraining, weak or pseudo labels can be tolerated since they are just used to make the model find to an easy start point for later convergence.

Of course, we will skip the pretraining stage as for the limited access to large document data. And we will mainly focus on how to modify the decoder and finetuning it.

In conclusion, we choose the PP-FormulaNet as our architecture, which has the best efficiency-performance trade-off.

# Workflow
- [x] 1. Determine the necessary tokens - hard coding into the tokenizer.
- [x] 2. Use a latex math mode parser to parse and normalize the mark-up expression.
- [x] 3. Collect a massive mono-corpus of latex math mode codes.
- [x] 4. Train a tokenizer (with hard-coded tokens) on the normalized mono-corpus.
- [ ] 5. Train a vision encoder-decoder model on normalized paired-corpus(image-latex). 
  - [x] Encoder: convert PP-HGNetV2 to pytorch
  - [x] Decoder: ~~migrate the UniMERNet decoder(customized MBart), refer to https://github.com/ParaN3xus/my-unimernet~~ no need to migrate since I finally find that PP-FormulaNet just uses the vanilla MBart decoder without the SqueezeAttention operation.
  - [ ] Adapt to transformers.VisionEncoderDecoder
  - [ ] Adapt image preprocessing for training and inference

[optional] Synthetic paired data:
1. Synthesize **smartly** latex math mode expressions.
2. Render it with an efficient engine(katex, mathjax, mathpix, latex).
3. Augment it with typical operations: 
  - affine transformation(rotation, translation, scaling)
  - deformations
  - noise

# Virtue
This tool IS made for those who have the basic knowledge on LaTeX, i.e. know how to manually compose LaTeX math formula, but typing everything by hand would simply break their note-taking mind flow.

This tool IS NOT made for those who lack the basic knowledge on LaTeX and pretend to learn math or any other knowledge by simply copy-paste to their notes. Since this tool doesn't support text-math mix OCR, these guys will not benefit from the convenience.

# Would do
1. latex math mode one liner
2. latex math mode multi liners

Possibly:
1. handwritten math
1. tikz-cd(for commutative graphs)
2. squeeze the model size to be run inside a browser

# Would not do
There can be a whole spectrum of a image-to-text task, from specific to general, easy to hard: 
1. single letter recognition (classification)
2. single-line recognition (sequential classification)
3. multi-line recognition (2D sequential classification)
4. page recognition (layout analysis + 2D sequential classification)
5. vision language model.

Any project should posit itself clearly on the spectrum according to the goal it wants to fullfill, comprising to the costs it can afford.

Level 4 (e.g. a massive recognition of PDF pages) is of course out of our scope, since I don't have a strong need for convert scanned PDF to editable one, which actually is a data collection step for training LLM. People can easily find commercial solutions such as [Quark Scanner](https://scan.quark.cn/). Not to mention level 5.

Hence the following features are would not be considered in the project's scope:
- text OCR or text-math mix OCR(except for \text in math mode, but only for common latin characters): as we want to compress the size of the model, supporting those has no fundamental difficulties but only an issue of the amount of training data and the model size.
- full latex recognition: the project sticks to the math mode only.
- image-to-markdown conversion
- PDF document recognition

We leave the community to extend them.

## Evaluation
Use UniMERNet metric:
MixTex TexTeller UniMERNet-tiny,small,base

## Notes
UniMERNet 的论文是比较详细的，并且它参考的前作 Donut 的论文亦如是。Donut 论文中提到 text reading 预训练（其实就是 OCR）对涨点帮助较大，但 Donut 相当于是文档理解模型，它的下游任务是比较多的，而它又自称是 OCR-free 模型，其实不妨说是把传统多步的文档理解流程中的 OCR 模块嵌入到预训练知识中。
而对于 UniMERNet 而言，因为它本质上就是 Math OCR，因此预训练并没有它说的那么大提升（只是 BLEU score 上小数点后第三位的提升），何况后作 CDM 上已经指出 BLEU score 作为 Math OCR 任务的度量标准是有失公允的。因此我个人认为就这个任务而言，预训练的帮助有限。

UniMER-1M 数据集的问题：
1. 下载数据集后，发现 train.txt 有 1061790 行，并非 dataset card 上说的 1061791 条数据。同时，train/images 下只有 986122 张图片。即便除去需要单独下载的 HME-100K-train 中的 74502 条数据，仍然不符合，有 1061790-74502-986122=987288-986122=1166 条缺口，经过查找在 [此处](https://github.com/opendatalab/UniMERNet/issues/2) 发现了数据集的渲染问题，删除了这 1166 条数据。
2. 参考 [这里](https://github.com/opendatalab/UniMERNet/issues/14) 合并 UniMER-1M 与 HME-100K 数据集。
3. 合并数据集后，按照官网仓库的 dataloader 配对（用 image 的编号去配对标注的行号）
4. 预处理的脚本 https://github.com/harvardnlp/im2markup/tree/master/scripts/preprocessing.

粗看了一下 UniMER-1M 的数据集，合成数据集的可行度相当高。特别是 SPE 和 CPE 其实也都是从现有数据集特别是 img2LaTeX-100K 中采样得到。

可以考虑参考一下 https://github.com/LinXueyuanStdio/LaTeX_OCR/ 虽然他的仓库比较古老，但其中的学习笔记值得一看。

注意到 PPFormulaNet-S 的 decoder 参数量为 44M，其中 embedding layer 和 lm head 各占 384*50000 = 19.2M，因此单单把词汇量从 50k 缩小到 1k 就能节省 37.6M 的参数，此时 decoder 剩余的参数量为 6M。

PaddleOCR 中的 PPHGNetv2 实现非常费解，我们找到了 D-FINE 的实现，虽然二者任务不同但骨干网络相同。

PPFormulaNet 并未使用 ESE 层和 LAB 层，故从代码中删去。

PPFormulaNet-S 的架构及模型参数量（单位 M）：
```
(Encoder)pphgnet_b4 13.59
  (stem) 0.03
  (stages) 13.57

(enc_to_dec_proj) 0.79 (2048*384)

(Decoder) MBart 43.53
  (embed_tokens) 19.20 (50000*384)
  (embed_positions) 0.40
  (layers) 4.73
    (layer 0) 2.37
      (self_attn) 0.59(384 * 384 * 4)
      (encoder_attn) 0.59(384 * 384 * 4)
      (fc1) 384*1536 0.59
      (fc2) 384*1536 0.59
    (layer 1) 2.37
  (lm_head) 19.20 (50000*384)
```
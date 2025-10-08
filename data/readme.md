`dataset` stores the all kinds of datasets during my experiments:
1. UniMER-1M and HME100K to merge to UniMER-1M_merged for training
2. UniMER-Test for test
3. simple for debugging
4. hf_datasets for making it to hf format and push to hf.

`tokenizer` stores the handicraft tokenizer I made for LaTeX (more precisely for KaTeX) commands.

`unimernet_tokenizer` stores the original tokenizer used by UniMERNet and PPFormulaNet

`unimernet_tokenizer_distill` stores the distilled version of the above one.

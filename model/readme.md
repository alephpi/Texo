This directory stores the pretrained model checkpoints:

1. formulanet.pt: the Pytorch version of the PPFormulanet-S which is built on Paddle.
2. formulanet_distill.pt: vocabulary distilled of 1 by distilling the embed_token and the lm_head layer. No finetuning.
3. formulanet_distill_best.pt: finetuned version of 2 on UniMER-1M.
4. formulanet_distill_best_transfer.pt: vocabulary transfered from 3.

# [Paper PDF](doc/your-paper.pdf)

# Probing Chest X-ray Representations in MedGemma

This project studies where chest X-ray finding information is most linearly accessible inside MedGemma 4B. The main experiment compares logistic-regression probes trained on standalone MedSigLIP embeddings, pre-decoder MedGemma image tokens, decoder-layer residual states, final prompt-token states, and MedGemma yes/no next-token scores.

The current report focuses on CheXpert+ with five findings: atelectasis, cardiomegaly, consolidation, edema, and pleural effusion. The main result is that the standalone MedSigLIP embedding performs best by mean AUROC, while MedGemma decoder representations and zero-shot generative classification do not improve linear separability in this setup.

## Environment

Install dependencies with either:

```bash
conda env create -f environment.yml
```

or:

```bash
pip install -r requirements.txt
```

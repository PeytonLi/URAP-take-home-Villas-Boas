# Appendix figure captions

These figures describe a small proof-of-concept experiment with four held-out synthetic review bundles and four abstract visual cues. They measure sensitivity to irrelevant visual variation, not demographic bias.

**Figure A1. Fine-tuning loss.** Training and held-out cross-entropy loss across the two-epoch LoRA run. Both decreased over four optimizer steps, showing that the model learned the small synthetic task without a rising held-out loss.

**Figure A2. Image-cue sensitivity before and after fine-tuning.** Bars show the mean across four held-out review bundles; points show individual bundles, and error bars are bundle-bootstrap 95% confidence intervals. Exact wording changed for 70.8% of base-model cue pairs and 41.7% of fine-tuned pairs. Mean TF-IDF distance fell from 0.173 to 0.029.

**Figure A3. Which summary properties changed across cues.** For each bundle, the max–min difference across four cues is calculated for transparent lexical proxies of evidence support, sentiment, recommendation language, and omission. These are screening measures, not substitutes for human claim-level annotation.

**Figure A4. Stability–faithfulness tradeoff.** Each arrow follows one held-out bundle from the base model to the fine-tuned model. Moving left indicates less cue sensitivity; moving up indicates greater lexical overlap with the review evidence. All four bundles moved left and up in this toy test.

**Figure A5. Pairwise cue sensitivity by review bundle.** Each cell reports TF-IDF distance between summaries generated from identical reviews under a pair of abstract cues. Darker cells indicate larger wording changes.

**Figure A6. Example outputs.** Base and fine-tuned summaries for the held-out bundle with the largest reduction in cue sensitivity. The fine-tuned outputs remain closer to the shared reference and vary less across visual cues.

Because the evaluation contains only four bundles, the estimates are unstable and should not be generalized. A full study needs many more review bundles, licensed or consented seller cues, preregistered thresholds, and blinded human checks of support, omissions, sentiment, and recommendation strength.

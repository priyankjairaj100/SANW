# Final rewrite: novelty and closest prior work

Verified 2026-10-03 against primary papers, official proceedings pages, and author arXiv records. This is an editorial audit. It changes no scientific result, experiment, training code, or protocol. Search coverage extends to work available on this date, rather than claiming an exhaustive literature search or establishing a first-ever priority claim.

## Recommended central story

The strongest supported contribution is a controlled account of the incremental value of relation labels. The design holds the annotated candidate pool, frozen backbone, adapter capacity, tuning grid, and validation rule fixed. It separates three uses of a label: promoting a supported caption, upweighting a contradiction, and excluding an unresolved statement. The distinctive question is whether the latter two operations add useful transfer beyond positive expansion.

Recommended title: **What Do Relation Labels Add Beyond Positive Captions?**

The existing title, *What Do Relation Labels Add to Image–Text Adaptation?*, is accurate but less specific about the essential comparison. Avoid a title announcing a universal retrieval tradeoff, equivalence of semantic and random weighting, or an impossibility result.

Suggested lead:

> An image can have several correct descriptions. Relation labels also identify captions that are wrong or unresolved. Do these distinctions help an image search model beyond marking more captions as correct? We answer this question with twelve training policies that share the same captions, model capacity, and selection budget. The main gain comes from recognizing supported statements. Transfer to edited captions is small, and retrieval loses correct matches.

This is a research finding about selected training procedures in the tested adaptation setting. It is not a new method for achieving state-of-the-art compositionality. The fixed-score identities explain what changes in the loss; they do not establish a learned-representation mechanism or a generalization guarantee.

## Five closest foundations and the exact boundary

| Work and verified publication | What is already established | Distinction available to this paper |
|---|---|---|
| **Kamath, Hsieh, Chang, and Krishna. The Hard Positive Truth about Vision-Language Compositionality. ECCV 2024.** [Paper](https://arxiv.org/html/2409.17958v1) | Hard-negative gains can conceal failures on meaning-preserving captions. Section 4.2 and Table 1 explicitly test SVLC+Pos and show that ordinary extra positives can fail to improve hard-positive recognition. | Neither retention failures nor the insufficiency of generic positives is new here. Our comparison isolates positive promotion, contradiction weighting, and neutral exclusion in one unchanged annotation pool, then measures their selected-policy outcomes. |
| **Oh, Cho, Kim, Kweon, and Kim. Preserving Multi-Modal Capabilities of Pre-trained VLMs for Improving Vision-Linguistic Compositionality. EMNLP 2024.** [Official paper and metadata](https://aclanthology.org/2024.emnlp-main.1062/) | FSC-CLIP identifies loss of general multimodal performance under global hard-negative training. It proposes local hard-negative supervision and selective calibrated regularization to preserve capabilities. | Retention-aware evaluation and the observation that hard-negative adaptation can damage retrieval are prior work. Our contribution concerns which annotated relation operation adds value under a common pool and capacity, without proposing the FSC-CLIP preservation mechanism. |
| **Peleg, Singh, and Hein. Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning. NeurIPS 2025.** [Paper, version 2](https://arxiv.org/html/2505.24424v2) | CLIC constructs training examples by combining images and captions. It evaluates SugarCrepe, SugarCrepe++, and retrieval, and reports joint improvements across multiple architectures and pretraining variants. | Joint semantic and retrieval evaluation is established. A retrieval penalty is not inherent to compositional learning. Our narrower question is the marginal contribution of explicit relation-label treatments within a fixed adaptation regime. CLIC is literature context, not an executed baseline in our twelve-policy experiment. |
| **Byun, Kim, and Moon. MAFA: Managing False Negatives for Vision-Language Pre-training. CVPR 2024.** [Official paper and metadata](https://openaccess.thecvf.com/content/CVPR2024/html/Byun_MAFA_Managing_False_Negatives_for_Vision-Language_Pre-training_CVPR_2024_paper.html) | MAFA mines missing positive connections and applies contrastive label smoothing. Its ablations already compare turning false negatives into positives with eliminating them. | Do not claim the first promotion-versus-removal comparison or the first use of many-to-many positives. Our distinction is the three-way supported/contradicted/neutral annotation policy, its factorial comparison after positive expansion, matched nonspecific controls, and measured retention costs. Neutral pairs here are unresolved annotations, not MAFA's mined positive matches. |
| **Liu et al. Improving Medical Vision-Language Contrastive Pretraining With Semantics-Aware Triage. IEEE Transactions on Medical Imaging, 2023.** [Paper DOI](https://doi.org/10.1109/TMI.2023.3294980), [author implementation](https://github.com/liubo105/SAT) | SAT uses inter-report similarity to divide image-report pairs into positive, negative, and neutral groups. | Three-way triage itself is prior work. Our categories come from explicit supported/contradicted/unresolved annotations, and the contribution is their controlled decomposition relative to positive expansion, including retrieval and valid-caption consequences. SAT's similarity-defined neutral category must not be equated with visual-inference neutrality. |

The five descriptions above are deliberately limited to the closest conceptual overlaps. The full-text checks of Kamath, CLIC, and MAFA are important: their contribution overlap is broader than their titles suggest. For SAT, identity and metadata were verified using the author repository and the paper's [deposited abstract record](https://pubmed.ncbi.nlm.nih.gov/37440389/); IEEE's direct paper retrieval was blocked. No unseen full-text detail is attributed to SAT.

## 2026 checks that affect the rewrite

Three current primary records further narrow broad novelty claims:

- **Pham, Hoffmann, Guerrero, and Martinez. No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models. CVPR 2026.** [Primary record](https://arxiv.org/abs/2603.25722), [official proceedings PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Pham_No_Hard_Negatives_Required_Concept_Centric_Learning_Leads_to_Compositionality_CVPR_2026_paper.pdf). Concept-centric caption parts and visual pooling support compositional improvements while preserving broader capabilities. This is another reason to avoid presenting the observed retrieval loss as inevitable.
- **Miranda, Salaberria, Agirre, and Azkune. Revisiting Compositionality in Dual-Encoder Vision-Language Models: The Role of Inference. 2026 preprint.** [Primary record, version 2](https://arxiv.org/abs/2604.11496v2). The authors study local alignment over frozen representations and the limits of global-score inference. This makes our global pooled-feature scope consequential. We do not test their alternative alignment mechanism.
- **Jeong, Jung, Choi, Jang, and Zhang. CS-CLIP: Compositional Scene Graph-guided CLIP for Robust Compositional Reasoning. September 2026 preprint.** [Primary record](https://arxiv.org/abs/2609.08242). The indexed author abstract reports element-specific biases and scene-graph negative construction with contradiction filtering. Full-text retrieval failed during this audit, so no detailed method comparison is claimed. Its abstract is already sufficient to rule out novelty claims based solely on discovering category imbalance or using contradiction-sensitive negatives.

These are targeted checks, not three additional methods to add experimentally. Add a short 2026 citation to the introduction if space permits; the complete positioning can remain in the appendix or this record. CS-CLIP is especially relevant if the final introduction emphasizes gains that concentrate in only some edit categories.

## Claim map for the fresh evidence

All numbers below come from `results/analysis/`, not the lost execution.

| Candidate headline | Editorial decision |
|---|---|
| Explicit relation operations have little incremental benefit beyond positive expansion in this study. | Lead with the controlled comparison. Grounded minus positive expansion is −0.255 points on e-SNLI-VE and −0.155 on SugarCrepe. Both adjusted intervals include zero, so use the measured differences and intervals rather than declaring equivalence. |
| Positive expansion produces strong supervised gains but weak transfer. | Strongest result sentence: +5.81 points on supervised hypothesis ranking, +0.14 points on SugarCrepe, and −8.56 points on COCO image-to-text R@1 relative to source-positive adaptation. |
| The aggregate benchmark change hides a directional split. | Keep the descriptive +1.92-point added-content result and −0.90-point combined replacement/swap result. Present them as a diagnostic of this intervention, not a new general discovery about benchmark categories. |
| Relation weighting inevitably damages retrieval. | Reject. The current study covers one backbone and global residual adapters; CLIC, FSC-CLIP, and 2026 concept-centric work demonstrate broader alternatives. |
| Semantic weights are equivalent to shuffled or constant weights. | Reject. The fresh fixed and shuffled controls differ substantially. The median and constant means are close, but there is no equivalence test. |
| The theory explains the observed representation failure. | Replace with a narrower statement: the identities separate target redistribution from negative-strength/allocation changes. They motivate the comparisons but do not identify the empirical failure mechanism. |
| This is the first test of extra positives versus false-negative removal. | Reject because MAFA already contains this comparison. Our neutral labels are also semantically different from known extra positives. |
| Supported/contradicted/neutral triage is itself a new contrastive-learning idea. | Reject as a broad formulation because SAT already distinguishes positive, negative, and neutral pairs. Emphasize the annotation semantics and controlled decomposition. |

The practical decision is whether fine-grained annotation earns its added training complexity while retaining useful retrieval. Quantify it with the changed retrieval successes and the three relation operations. Do not describe neutral labeling or contradiction labeling as cost-effective: this project does not measure labeling costs.

The constant-weight control makes this decision more concrete. It retains 55.21% COCO image-to-text R@1 and reaches 83.67% SugarCrepe accuracy, compared with 47.29% and 83.39% for positive expansion. Its supervised relation accuracy is lower, 82.15% versus 87.00%. This supports a practical choice between supervised discrimination and retained transfer in the tested setting, not a claim that constant weighting dominates every objective. The control does not use relation classes in its training loss, but the shared validation selection rule uses relation labels. Therefore **annotation-free training and selection is not a supported claim**.

## Main-text priority

1. Establish a photo-search problem and define supported, contradicted, and neutral in ordinary language.
2. Make the fixed candidate pool explicit. Supported hypotheses are already candidates in the source-positive baseline and are treated as negatives there. Thus the large supervised gain partly removes an intentional mismatch in that baseline; it cannot be sold as a pure improvement over ordinary caption-only CLIP adaptation.
3. State the central marginal comparison and the practical retrieval change. Give the two primary intervals once.
4. Retain the edit-group breakdown because it interprets the small aggregate transfer gain. Keep broad retention motivation short because prior work already establishes it.
5. Put the two exact identities after the empirical question is clear. Keep their assumptions visible, and move full derivations and additional controls to the appendix.

Suggested compact related-work transition, paraphrased rather than quoted:

> Prior work has shown that compositional benchmark gains can coexist with failures on valid rewordings or retrieval. Other methods improve these capabilities together. We ask a different, more specific question: when the candidate captions and adaptation budget are fixed, what does treating contradictions and unresolved captions separately add beyond recognizing more supported captions?

## Bibliographic checks

The existing Kamath, FSC-CLIP, and CLIC author lists, titles, and venue years in `manuscript/references.bib` are correct. CLIC is a **NeurIPS 2025** paper, not merely a 2025 preprint; volume **38** can be added. Its listing is verified in the [official 2025 proceedings](https://proceedings.nips.cc/paper_files/paper/2025/vol38-main-conference).

If adding the missing references:

- MAFA: Jaeseok Byun, Dohoon Kim, Taesup Moon; CVPR **2024**, pages **27314–27324**. Do not use 2023 simply because the arXiv identifier starts with 2312.
- SoftCLIP: Yuting Gao, Jinfeng Liu, Zihan Xu, Tong Wu, Enwei Zhang, Ke Li, Jie Yang, Wei Liu, Xing Sun; AAAI **2024**, volume **38**, issue **3**, pages **1860–1868**, DOI **10.1609/aaai.v38i3.27955**. Prefer the official proceedings author ordering over malformed search exports.
- Pham et al.: CVPR **2026**, author spelling on the paper is Hai X. Pham, David T. Hoffmann, Ricardo Guerrero, Brais Martinez.
- CS-CLIP: preserve the title and author capitalization from the primary record; cite as a 2026 arXiv preprint unless a venue is independently verified.
- SugarCrepe++: the present six-author list and arXiv identifier **2406.11171** match the [primary record](https://arxiv.org/abs/2406.11171). No conference venue was verified in this check.

No manuscript bibliography or scientific artifact was changed by this audit.

## Resolved historical citation keys

These four old keys were supplied by the recovered inventory. All four now have identified primary references; no title was guessed from a key alone.

| Recovered key | Verified identity and source | Preservation and novelty note |
|---|---|---|
| `jiang2023regulation` | Chaoya Jiang, Wei Ye, Haiyang Xu, Songfang Huang, Fei Huang, Shikun Zhang. **Vision Language Pre-training by Contrastive Learning with Cross-Modal Similarity Regulation.** ACL 2023, pages 14660–14679. DOI 10.18653/v1/2023.acl-long.819. [Official PDF](https://aclanthology.org/2023.acl-long.819.pdf) | The published PDF calls the method **Similarity-Regulated Contrastive Learning (SRCL)**. The Anthology landing-page abstract uses SACL, so prefer the PDF's name. Its negative weights derive from cross-modal similarity and use a mean-one normalization condition. Our fixed suppression and frozen-text weighting controls are not an SRCL reproduction. |
| `gao2024softclip` | Yuting Gao, Jinfeng Liu, Zihan Xu, Tong Wu, Enwei Zhang, Ke Li, Jie Yang, Wei Liu, Xing Sun. **SoftCLIP: Softer Cross-Modal Alignment Makes CLIP Stronger.** AAAI 2024, 38(3):1860–1868. DOI 10.1609/aaai.v38i3.27955. [Official proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/27955) | Similarity-informed soft targets and many-to-many alignment are prior art. Preserve as background for semantic alignment. The reconstructed negative-weight controls do not reproduce this objective. |
| `liu2023sat` | Bo Liu, Donghuan Lu, Dong Wei, Xian Wu, Yan Wang, Yu Zhang, Yefeng Zheng. **Improving Medical Vision-Language Contrastive Pretraining With Semantics-Aware Triage.** IEEE Transactions on Medical Imaging 42(12):3579–3589, 2023. DOI 10.1109/TMI.2023.3294980. [Publisher record](https://ieeexplore.ieee.org/document/10182304), [author repository](https://github.com/liubo105/SAT) | Include when discussing prior neutral-group treatment. Full publisher text was inaccessible during this check; the abstract's three-group construction is verified, but no detailed loss comparison is made. |
| `byun2024mafa` | Jaeseok Byun, Dohoon Kim, Taesup Moon. **MAFA: Managing False Negatives for Vision-Language Pre-training.** CVPR 2024, pages 27314–27324. [Official proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Byun_MAFA_Managing_False_Negatives_for_Vision-Language_Pre-training_CVPR_2024_paper.html) | Preserve the positive-conversion/removal and smoothing connection. Do not describe our work as the first comparison of these choices. |

The bibliography for SRCL must use the six authors in the official ACL PDF. Search exports for its arXiv version contain a different author list, so those should not be copied into the ACL citation.

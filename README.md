# MedGemma Vision Fine-Tuning

`unsloth/medgemma-4b-it-bnb-4bit` fine-tuned with LoRA on BBBC006 to estimate the number of Hoechst-stained U2OS nuclei in fluorescence microscopy images.
MedGemma is a collection of Gemma 3 variants that are trained for performance on medical text and image comprehension.

<img src="medgemma.jpg" width="400"> 


The model takes an image and an instruction then returns JSON:

```json
{"nuclei_count": 87}
```

> **Research use only.** This is a proof-of-concept image-counting adapter, not a clinical, diagnostic, or production microscopy-analysis system.


## Training data

The training set was created from [BBBC006 v1](https://bbbc.broadinstitute.org/BBBC006): Hoechst-stained U2OS microscopy images from 384 wells, with two fields of view per well. The adapter uses the locally available `z=00`, `w1` (Hoechst/DAPI) TIFF images.

Targets are BBBC006 `Image_Count_Nuclei` values. BBBC006 documents these as automated counts generated at the optimal-focus `z=16` plane with a CellProfiler-style `IdentifyPrimaryObjects` pipeline; they are reference labels, not independent manual annotations.

| Item | Value |
| --- | ---: |
| Total labelled images | 768 |
| Training images / wells | 614 / 307 |
| Evaluation images / wells | 154 / 77 |
| Split | Random, well-disjoint, seed 42 |

### Data processing

1. Match z=00 TIFF files to BBBC006 count rows using the stable well and site identifiers. The image UUID differs across focal planes.
2. Retain only `w1` Hoechst/DAPI images; exclude `w2` images.
3. Convert each 16-bit grayscale TIFF to 8-bit RGB using per-image 1st--99.5th percentile scaling.
4. Split by well with seed 42 and an 80/20 ratio. Both sites from each well remain together.
5. Train the model to answer the fixed instruction with `{"nuclei_count": <integer>}`.

The implementation is in [scripts/fine_tune_medgemma_bbbc006.py](scripts/fine_tune_medgemma_bbbc006.py).

## Training

| Setting | Value |
| --- | --- |
| Base model | `unsloth/medgemma-4b-it-bnb-4bit` |
| Method | Supervised fine-tuning (LoRA) |
| LoRA rank / alpha / dropout | 16 / 16 / 0.05 |
| Adapted layers | Vision and language layers |
| Epochs | 5 |
| Learning rate | 2e-4, cosine schedule |
| Batch size / gradient accumulation | 1 / 4 |
| Maximum sequence length | 512 |
| Precision | BF16 where supported; 4-bit base-model loading |
| Hardware | One NVIDIA H100 GPU |

The run uses `load_best_model_at_end`, selecting epoch 4 by validation loss (`eval_loss=0.30`).


## Result

The final adapter was evaluated on a held-out, well-disjoint BBBC006 split. Both fields of view from a well are assigned to the same split.

| Metric | Result |
| --- | ---: |
| Training / held-out images | 614 / 154 |
| Training / held-out wells | 307 / 77 |
| Valid JSON outputs | 154 / 154 (100%) |
| MAE | **6.21 nuclei** |
| RMSE | **7.79 nuclei** |
| MAPE | 18.79% |
| Predictions within +/-5 nuclei | 77 / 154 (50.0%) |
| Predictions within +/-10 nuclei | 124 / 154 (80.5%) |
| R-squared | 0.96 |

The reference-count mean baseline has MAE 30.62 and RMSE 39.35 on the same split. This comparison shows that the model uses image content effectively within this dataset; it is not an external benchmark comparison.

## Evaluation output

After training, the script generates `eval_predictions.json` in the adapter output directory. It includes:

- aggregate MAE, RMSE, MAPE, and valid-output count;
- every held-out image path and reference count;
- the raw model response;
- the parsed `predicted_nuclei_count`.

The results for the final run are at [eval_predictions.json](eval_predictions.json).

##  Use case

Use this adapter for research experiments that investigate whether a vision-language model can estimate BBBC006 automated nucleus counts from out-of-focus Hoechst images. Outputs should be parsed as JSON and reviewed alongside the image.

It is not intended for cell segmentation, instance-level nucleus localization, clinical microscopy, general cell counting, or measurements without quality control.

## License and attribution

The adapter is a derivative of MedGemma and is subject to the applicable [Health AI Developer Foundations terms](https://huggingface.co/google/medgemma-4b-it). Cite BBBC006 when using its images or labels, and verify all applicable terms before redistribution.

## Citation

If you use this adapter, cite BBBC006 and MedGemma. A project-specific citation can be added here after publication.

```bibtex
@article{ljosa2012annotated,
  title={Annotated high-throughput microscopy image sets for validation},
  author={Ljosa, Vebjorn and others},
  journal={Nature Methods},
  year={2012}
}
```

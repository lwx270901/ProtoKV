# ProtoKV
[ICML'26] Streaming Video Understanding under Delayed Query with Summary-State Memory

## Installation

```bash
pip install -r requirements.txt
```

Tested with `torch==2.8` and CUDA 12.6 environment.

---

## Preparation

- Download benchmarks under `data/`
  - [MLVU-dev-mc](https://huggingface.co/datasets/MLVU/MVLU)
  - [RVS](https://huggingface.co/datasets/Becomebright/RVS)
  - The `data/` folder should be arranged as:
    ```
    data
    ├── mlvu
    │   ├── dev_debug_mc.json
    │   └── videos
    ├── streamingbench
    │   ├── questions_real_stream.json
    │   └── data/
    ├── ovo_bench
    │   ├── ovo_bench_new.json
    │   └── src_videos
    └── rvs
        ├── ego
        │   ├── ego4d_oe.json
        │   └── videos
        └── movie
            ├── movienet_oe.json
            └── videos
    ```

## Evaluation on online benchmark

-RVS
```bash
bash scripts/qwen_inference_online.sh 
```
-OVO
```bash
bash scripts/qwen_inference_online.sh 
```
-StreammingBench
```bash
bash scripts/qwen_inference_online.sh 
```

## Evaluation on MLVU (offline benchmark)

```bash
bash scripts/qwen_inference_online.sh 
```


## Key Arguments

| Argument | Description |
|----------|-------------|
| 'model_path' | Specifies the Video-LLM backbone used for inference|
|'data_path'  | Path to the benchmark annotation file |
| 'video_root'| Root directory where the videos are stored|
| 'output_csv' | Path where the prediction results will be saved. |
| 'experiment' |Runs the query-delay streaming setting, where the model answers after different delay times. |
|'deltas' | List of delay times (in seconds) for the query-delay streaming setting. |
|'max_frames_num' | Maximum number of sampled video frames. This belongs to the evaluation/data sampling setup.|
|'block_size' | Number of frames processed per streaming block. This relates to online/block-wise streaming inference.|
|'compress_frame_num'| Total compressed KV budget in frame units.|
|'per_frame'| Enables streaming-style frame-wise updates, matching the paper’s online update setting.|
|'prototrack_proto_frames'| Number of far-memory prototype frame slots |
|'prototrack_pq_subspaces'| Number of PQ subquantizers G |
|'prototrack_pq_codebook_size' | Number of codewords per PQ subquantizer C|
|'prototrack_pq_kmeans_iters'| Number of k-means iterations used to initialize PQ codebooks. This is an implementation detail of the paper’s PQ codebook initialization.|
|'prototrack_pq_sample_size'| Number of residual samples used to fit PQ codebooks. This is an implementation detail for residual-codebook training.|
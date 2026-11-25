# STAR 原始模型基线

模型：`/data/wyt/Qwen/Qwen3.5-0.8B`。输入为 STAR 标注时间段内均匀采样的 8 帧、问题和四个选项；图像最长边 336 像素，贪心解码。视频位于 `/data/wyt/charades/Charades_v1_480`。

## 运行

在项目目录下用 4 张 GPU 分片评测完整验证集：

```bash
mkdir -p outputs/star_val_baseline
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src OMP_NUM_THREADS=2 \
    uv run --no-sync python -m video_reason.evaluate_star \
    --annotations /data/wyt/STAR/STAR_val.json \
    --device cuda --limit 0 --frames 8 --image-size 336 \
    --num-shards 4 --shard-index "$gpu" \
    --output "outputs/star_val_baseline/star_val_${gpu}.jsonl" \
    > "outputs/star_val_baseline/star_val_${gpu}.log" 2>&1 &
done
wait
```

## Baseline 结果

STAR val 全量 7,098 题；预测无缺失、重复或无效选项。准确率按题目数统计，错误和无效选项均计为未答对。

| 题型 | 答对 / 总数 | 准确率 |
| --- | ---: | ---: |
| Interaction | 1,270 / 2,398 | 52.96% |
| Sequence | 1,952 / 3,586 | 54.43% |
| Prediction | 288 / 624 | 46.15% |
| Feasibility | 225 / 490 | 45.92% |
| **总体** | **3,735 / 7,098** | **52.62%** |

四类准确率的宏平均为 **49.87%**。逐题预测、运行日志和汇总指标保存在 `outputs/star_val_baseline/`，汇总文件为 `metrics.json`。

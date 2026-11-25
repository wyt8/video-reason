"""Load Qwen3.5 and generate an answer for sampled video frames."""

from pathlib import Path


def load_model(model_path: Path, device_choice: str):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = (
        "cuda"
        if device_choice == "cuda"
        or (device_choice == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device == "cuda"
        else torch.float32
    )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = (
        AutoModelForMultimodalLM.from_pretrained(
            model_path, dtype=dtype, local_files_only=True
        )
        .to(device)
        .eval()
    )
    return model, processor, device


def generate_answer(
    model,
    processor,
    device: str,
    frames,
    clip_duration: float,
    question: str,
    max_new_tokens: int,
) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": question},
            ],
        }
    ]
    video_metadata = {
        "total_num_frames": len(frames),
        "fps": len(frames) / clip_duration,
        "duration": clip_duration,
        "frames_indices": list(range(len(frames))),
        "height": frames.shape[1],
        "width": frames.shape[2],
    }
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"video_metadata": video_metadata, "do_sample_frames": False},
    ).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return processor.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()

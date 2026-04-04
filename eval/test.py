import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

checkpoint_path = '/root/autodl-tmp/output/qwen2.5-vl-poirot-gdpo/checkpoint-2250'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(checkpoint_path)

messages = [
    {
        "role": "system",
        "content": "You are an expert visual reasoning assistant. Please strictly output in the required XML format including <observe>, <think>, <action>."
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/root/autodl-tmp/videos/_3YCVVagLSI/105.jpg"},
            {"type": "image", "image": "/root/autodl-tmp/videos/_3YCVVagLSI/110.jpg"},
            {"type": "text", "text": "Please describe what is happening in the images and provide the bounding boxes of the main subjects."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.1
)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(output_text[0])
import json
import torch
import soundfile as sf
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from jiwer import wer
from transformers import Qwen3ForCausalLM, AutoTokenizer
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

import argparse
import re
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_json_from_markdown(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json.loads(json_str)
    else:
        try:
            json_str = json.loads(text)
            return json_str
        except Exception as e:
            print(e)
            return None

def compute_Chinese_wer(ref, hyp, is_chinese):
    from jiwer import wer
    if is_chinese:
        ref = " ".join(ref.strip())
        hyp = " ".join(hyp.strip())
    else:
        ref = normalize_text(ref)
        hyp = normalize_text(hyp)
    return wer(ref, hyp)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--test_data', required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--is_Chinese', action='store_true')
args = parser.parse_args()
print('args:', args)

model_path = args.model_path
judge_model_path = "/mnt/general/share/model/Qwen/Qwen3-14B"
test_jsonl_path = args.test_data


testset_name = os.path.splitext(os.path.basename(test_jsonl_path))[0]
csv_output_path = f"test_result/{args.model_name}_{testset_name}.csv"

with open("./prompt.txt") as prompt_file:
    prompt = prompt_file.read()

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", low_cpu_mem_usage=False
).to('cuda:0')
model.eval()

judge_model = Qwen3ForCausalLM.from_pretrained(judge_model_path).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(judge_model_path)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

wer_list = []
result_rows = []
badcase_list = []

if args.is_Chinese:
    question = "请将音频内容转录为文本，直接输出转录的文本即可，严禁输出任何解释！请输出中文汉字，禁止输出汉语拼音或英文"
else:
    question = "What does the audio say? Directly give your transcription result, DO NOT add any explanation"

with open(test_jsonl_path, "r", encoding="utf-8") as fin:
    for line in tqdm(fin, desc="Evaluating"):
        data = json.loads(line.strip())
        reference = data["messages"][1]["content"].strip()
        audio_path = data["audios"][0]
        audio_id = os.path.splitext(os.path.basename(audio_path))[0]

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "audio", "audio": audio_path}
                ],
            },
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        ).to(model.device).to(model.dtype)

        with torch.no_grad():
            text_ids, _ = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=256)
        # import pdb
        # pdb.set_trace()
        output_text = processor.batch_decode(
            [text_ids[0][len(inputs.input_ids[0]):]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

        judge_input = {
            "ground_truth": reference,
            "transcripted": output_text
        }

        judge_prompt = prompt + "\n\n" + json.dumps(judge_input, ensure_ascii=False)
        judge_messages = [
            {"role": "user", "content": judge_prompt}
        ]
        import pdb
        # pdb.set_trace()
        judge_inputs = tokenizer.apply_chat_template(
            judge_messages, 
            tokenize = False,
            add_generation_prompt = True,
            enable_thinking=False
        )
        judge_inputs = tokenizer([judge_inputs], return_tensors="pt").to(judge_model.device)
        with torch.no_grad():
            judge_output_ids = judge_model.generate(
                **judge_inputs,
                max_new_tokens=256,
                do_sample=False
            )
        judge_output_text = tokenizer.decode(
            judge_output_ids[0][judge_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        print ("output: ", judge_output_text)
        try:
            judge_result = extract_json_from_markdown(judge_output_text)
            if judge_result is not None:
                is_success = judge_result.get("is_extraction_success", False)
                pred = judge_result.get("extracted_content", "")
            else:
                is_success = False
                pred = output_text  # fallback：提取失败就直接用原始输出
                
        except Exception as e:
            print(f"[Warning] Failed to parse judge output JSON: {e}")
            is_success = False
            pred = output_text

        row = {
            "audio_id": audio_id,
            "reference": reference,
            "raw_output": output_text,
            "is_extract_success": is_success,
            "extracted_content": pred,
            "wer": ""
        }

        if is_success:
            norm_pred = normalize_text(pred)
            norm_ref = normalize_text(reference)
            current_wer = compute_Chinese_wer(norm_ref, norm_pred, args.is_Chinese)
            wer_list.append(current_wer)
            row["wer"] = current_wer
        else:
            badcase_list.append(audio_id)

        result_rows.append(row)

        print(f"[{audio_id}]")
        print(f" Pred: {normalize_text(pred) if is_success else '[FAIL]'}")
        print(f" GT  : {normalize_text(reference)}")
        if is_success:
            print(f" WER : {current_wer:.4f}")
        print("-" * 60)



# 输出评测结果
df = pd.DataFrame(result_rows)
df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
print(f"\n[Output] Saved evaluation results to: {csv_output_path}")

if wer_list:
    print(f"[Summary] Average WER (on successful transcriptions): {np.mean(wer_list):.4f}")
else:
    print("[Summary] No successful transcription, WER not available.")

print(f"[Badcase] Total failed transcriptions: {len(badcase_list)}")

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
from vllm import LLM, SamplingParams
import argparse
import re
import csv
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

def inference_stage(model, processor, test_jsonl_path, is_chinese, inference_csv_path):
    os.makedirs(os.path.dirname(inference_csv_path), exist_ok=True)

    if os.path.exists(inference_csv_path):
        df_existing = pd.read_csv(inference_csv_path)
        already_processed_ids = set(df_existing["audio_id"].tolist())
        results = df_existing.to_dict("records")
    else:
        already_processed_ids = set()
        results = []

    with open(test_jsonl_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Inference Stage"):
            data = json.loads(line.strip())
            reference = data["messages"][1]["content"].strip()
            audio_path = data["audios"][0]
            audio_id = os.path.splitext(os.path.basename(audio_path))[0]

            if audio_id in already_processed_ids:
                continue
            if args.is_Chinese:
                question = "请将音频内容转录为文本，直接输出转录的文本即可，严禁输出任何解释！请输出中文汉字，禁止输出汉语拼音或英文"
            else:
                question = "What does the audio say? Directly give your transcription result, DO NOT add any explanation"


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
                output_ids, _ = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=256)
            output_text = processor.batch_decode(
                [output_ids[0][len(inputs.input_ids[0]):]],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()

            results.append({
                "audio_id": audio_id,
                "reference": reference,
                "raw_output": output_text
            })


            pd.DataFrame(results).to_csv(inference_csv_path, index=False, encoding="utf-8-sig")


def judge_stage_vllm(prompt_path, inference_csv_path, judge_csv_path, model_path, batch_size):
    os.makedirs(os.path.dirname(judge_csv_path), exist_ok=True)

    with open(prompt_path, encoding="utf-8") as pf:
        prompt_template = pf.read()

    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    already_judged_ids = set()
    write_header = not os.path.exists(judge_csv_path)

    with open(judge_csv_path, "a", encoding="utf-8-sig", newline='') as fout:
        writer = csv.writer(fout)

        if write_header:
            writer.writerow(["audio_id", "reference", "raw_output", "is_extract_success", "extracted_content"])
        else:
            try:
                df_existing = pd.read_csv(judge_csv_path)
                already_judged_ids = set(df_existing["audio_id"].tolist())
            except Exception as e:
                print(f"[WARN] Failed to load existing judge CSV: {e}")

        df_infer = pd.read_csv(inference_csv_path)

        batch_messages = []
        batch_meta = []

        for idx, row in tqdm(df_infer.iterrows(), total=len(df_infer), desc="Preparing Judge Inputs"):
            audio_id = row["audio_id"]
            if audio_id in already_judged_ids:
                continue

            reference = row["reference"]
            raw_output = row["raw_output"]

            judge_input = {
                "ground_truth": reference,
                "transcripted": raw_output
            }
            prompt = prompt_template + "\n\n" + json.dumps(judge_input, ensure_ascii=False)

            messages = [{"role": "user", "content": prompt}]
            batch_messages.append(messages)
            batch_meta.append((audio_id, reference, raw_output))

            if len(batch_messages) == batch_size:
                outputs = llm.chat(
                    batch_messages,
                    sampling_params=sampling_params,
                    chat_template_kwargs={"enable_thinking": False}
                )

                for output, (audio_id, reference, raw_output) in zip(outputs, batch_meta):
                    output_text = output.outputs[0].text.strip()
                    try:
                        result = extract_json_from_markdown(output_text)
                        if result is not None:
                            is_success = result.get("is_extraction_success", False)
                            pred = result.get("extracted_content", "")
                        else:
                            is_success = False
                            pred = raw_output
                    except Exception as e:
                        print(f"[Judge Parse Error] {audio_id}: {e}")
                        is_success = False
                        pred = raw_output

                    writer.writerow([audio_id, reference, raw_output, is_success, pred])
                    fout.flush()

                batch_messages = []
                batch_meta = []

        # 处理剩余不足一批的样本
        if batch_messages:
            outputs = llm.chat(
                batch_messages,
                sampling_params=sampling_params,
                chat_template_kwargs={"enable_thinking": False}
            )

            for output, (audio_id, reference, raw_output) in zip(outputs, batch_meta):
                output_text = output.outputs[0].text.strip()
                try:
                    result = extract_json_from_markdown(output_text)
                    if result is not None:
                        is_success = result.get("is_extraction_success", False)
                        pred = result.get("extracted_content", "")
                    else:
                        is_success = False
                        pred = raw_output
                except Exception as e:
                    print(f"[Judge Parse Error] {audio_id}: {e}")
                    is_success = False
                    pred = raw_output

                writer.writerow([audio_id, reference, raw_output, is_success, pred])
                fout.flush()


def wer_stage(judge_csv_path, is_chinese, wer_csv_path):
    os.makedirs(os.path.dirname(wer_csv_path), exist_ok=True)

    df_judge = pd.read_csv(judge_csv_path)
    wer_list = []
    rows = []

    for idx, row in tqdm(df_judge.iterrows(), total=len(df_judge), desc="WER Stage"):
        if not bool(row["is_extract_success"]):
            continue

        ref = normalize_text(row["reference"])
        hyp = normalize_text(row["extracted_content"])
        w = compute_Chinese_wer(ref, hyp, is_chinese)

        rows.append({
            "audio_id": row["audio_id"],
            "reference": row["reference"],
            "prediction": row["extracted_content"],
            "wer": w
        })
        wer_list.append(w)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(wer_csv_path, index=False, encoding="utf-8-sig")

    success_count = df_judge["is_extract_success"].sum()
    failure_count = (~df_judge["is_extract_success"]).sum() 

    WER_result_file = f"results/{args.model_name}_{base_name}_final_result.csv"
    with open(WER_result_file, "w", encoding = 'utf-8') as result_file:
        result_file.write(f"Total samples: {len(df_judge)}\n")
        result_file.write(f"Successful extractions: {success_count}\n")
        result_file.write(f"Failed extractions: {failure_count}\n")
        if wer_list:
            result_file.write(f"Average WER: {sum(wer_list)/len(wer_list):.4f}")
        else:
            result_file.write("No valid results to compute WER.")
        print(f"结果已经保存到：{WER_result_file}")

    bad_cases = df_judge[df_judge["is_extract_success"] == False]
    if not bad_cases.empty:
        bad_case_csv = wer_csv_path.replace("_wer.csv", "_bad_case.csv")
        bad_cases.to_csv(bad_case_csv, index=False, encoding="utf-8-sig")
        print(f"[Bad Cases] Saved {len(bad_cases)} bad cases to {bad_case_csv}")
    else:
        print("[Bad Cases] No bad cases found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--is_Chinese', action='store_true')
    parser.add_argument('--stage', choices=['inference', 'judge', 'wer', 'all'], default='all')
    parser.add_argument('--prompt_path', default="./prompt.txt")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.test_data))[0]
    inference_csv = f"results/{args.model_name}_{base_name}_inference.csv"
    judge_csv = f"results/{args.model_name}_{base_name}_judge.csv"
    wer_csv = f"results/{args.model_name}_{base_name}_wer.csv"

    if args.stage in ['inference', 'all']:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", low_cpu_mem_usage=False
        ).to("cuda:0").eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
        inference_stage(model, processor, args.test_data, args.is_Chinese, inference_csv)
        del model
        del processor
        torch.cuda.empty_cache()

    if args.stage in ['judge', 'all']:
        judge_model_path = "/mnt/general/share/model/Qwen/Qwen3-14B"
        judge_stage_vllm(args.prompt_path, inference_csv, judge_csv, judge_model_path, batch_size=3)

    if args.stage in ['wer', 'all']:
        wer_stage(judge_csv, args.is_Chinese, wer_csv)

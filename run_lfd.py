import argparse
import json
from tqdm import tqdm
from llm_lfd import LLM
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_data(data_path):
    new_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Samples total num：" + str(len(data)))
    for i, datum in enumerate(data):
        new_data.append({
            "id": i,
            "question": datum["question"],
            "answers": datum["answers"],
            "prompt": datum["prompt"]
        })

    return new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model list:
    # NousResearch/Llama-2-7b-chat-hf
    # mistralai/Mistral-7B-v0.1
    # deepseek-ai/deepseek-llm-7b-base
    # Qwen/Qwen3-8B
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-chat-hf")
    parser.add_argument("--data_path", type=str, default="./data/nq/nq_noise_0.json")
    parser.add_argument("--output_path", type=str, default="./output/llama2-7b/nq/nq_lfd.json")
    args = parser.parse_args()

    data = load_data(args.data_path)

    llm = LLM(args.model_name, 'cuda', quantization_bits=4, model_max_length=4096)

    try:
        results = json.load(open(args.output_path, "r", encoding="utf-8"))
    except FileNotFoundError:
        results = []

    index = len(results)
    for datum in tqdm(data[len(results):]):
        question = datum["question"]
        prompt = datum["prompt"]
        generated_text, premature_layer_dist = llm.generate(prompt, max_new_tokens=50)
        results.append({
            "id": index,
            "prompt": prompt,
            "question": question,
            "answers": datum["answers"],
            "output": generated_text,
            "premature_layer_dist": premature_layer_dist
        })
        index += 1
        json.dump(results, open(args.output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
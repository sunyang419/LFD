import argparse
import json
from tqdm import tqdm
from llm_ef import LLM
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
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-chat-hf")
    parser.add_argument("--data_path", type=str, default="./data/nq/nq_noise_0.json")
    parser.add_argument("--output_path", type=str, default="./output/llama2-7b/nq/nq_lfd_ef.json")
    args = parser.parse_args()

    data = load_data(args.data_path)[:500]

    llm = LLM(args.model_name, 'cuda', quantization_bits=4, model_max_length=4096)

    try:
        results = json.load(open(args.output_path, "r", encoding="utf-8"))
    except FileNotFoundError:
        results = []

    print("Warming up model...")
    warmup_prompt = "This is a warmup prompt."
    for _ in range(3):  # 3 times
        _ = llm.generate(warmup_prompt, max_new_tokens=10)
    print("Warmup completed.\n")

    index = len(results)
    for datum in tqdm(data[len(results):]):
        question = datum["question"]
        prompt = datum["prompt"]
        
        metrics = llm.generate(prompt, max_new_tokens=50)
        results.append({
            "id": index,
            "prompt": prompt,
            "question": question,
            "answers": datum["answers"],
            "metrics": metrics,
        })
        index += 1
        json.dump(results, open(args.output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
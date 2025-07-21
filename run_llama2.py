import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def format_race(example: dict) -> str:
    return f"{example['question']}\n(a) {example['ans0']} (b) {example['ans1']} (c) {example['ans2']}\n{example['context']}"

def format_arc(example: dict) -> str:
    return f"{example['context']}{example['question']}\n(a) {example['ans0']} (b) {example['ans1']} (c) {example['ans2']}"

def format_qonly(example: dict) -> str:
    return f"{example['question']}\n(a) {example['ans0']} (b) {example['ans1']} (c) {example['ans2']}"

def load_model(model_name: str, device: int = -1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    return generator

def generate_answer(generator, prompt: str, max_new_tokens: int = 20) -> str:
    out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]['generated_text']
    return out[len(prompt):].strip()

def process_file(input_path: Path, output_dir: Path, generator, model_tag: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"preds_{input_path.stem}.jsonl"
    with input_path.open() as f_in, out_path.open('w') as f_out:
        for line in tqdm(f_in, desc=f"Processing {input_path.name}"):
            ex = json.loads(line)
            pred_race = generate_answer(generator, format_race(ex))
            pred_qonly = generate_answer(generator, format_qonly(ex))
            pred_arc = generate_answer(generator, format_arc(ex))

            ex[f"{model_tag}_pred_race"] = pred_race
            ex[f"{model_tag}_pred_qonly"] = pred_qonly
            ex[f"{model_tag}_pred_arc"] = pred_arc
            f_out.write(json.dumps(ex) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run BBQ with Llama2 model")
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-chat-hf', help='HF model name')
    parser.add_argument('--data_dir', default='data', help='Directory with BBQ jsonl files')
    parser.add_argument('--out_dir', default='results/Llama2', help='Where to write predictions')
    parser.add_argument('--device', type=int, default=-1, help='GPU id or -1 for cpu')
    args = parser.parse_args()

    generator = load_model(args.model, device=args.device)
    model_tag = args.model.split('/')[-1]
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    for path in sorted(data_dir.glob('*.jsonl')):
        process_file(path, out_dir, generator, model_tag)

if __name__ == '__main__':
    main()

#!/usr/bin/env python

# python evaluation.py \
#   --model kakaocorp/kanana-1.5-8b-instruct-2505 \
#   --dataset ArenaHard \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --reasoning False \
#   --max_tokens 1024

import argparse
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
# from eval_utils import system_prompt, safe_parse
# from litellm import batch_completion
import os
os.environ['OPENAI_API_KEY'] = ""

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a vLLM model on a dataset and save its responses.")
    p.add_argument("--model", required=True,
                   help="Hugging Face model ID or local path")
    p.add_argument("--dataset", required=True,
                   help="Dataset ID or local path")
    p.add_argument("--split", default="test",
                   help="Dataset split (default: test)")
    p.add_argument("--revision", default=None,
                   help="Model revision / commit hash (optional)")
    p.add_argument("--max_tokens", type=int, default=32768,
                   help="Maximum tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="Top-p / nucleus sampling (default: 1.0 = disabled)")
    p.add_argument("--output", default=None,
                   help="Output CSV path (auto-generated if omitted)")
    p.add_argument("--reasoning", type=bool, default=True, 
                   help="Whether the model uses system 2 thinking.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- build LLM ----------------------------------------------------------
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": torch.cuda.device_count(),
    }
    if args.revision:
        llm_kwargs["revision"] = args.revision

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # ---- load data ----------------------------------------------------------
    df = load_dataset('HAERAE-HUB/KoSimpleEval',args.dataset, split=args.split).to_pandas()

    # ---- craft prompts ------------------------------------------------------
    prompts = []
    
    if args.dataset in ['ArenaHard']:
        for _, row in df.iterrows():
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row['prompts']}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
    else:
        for _, row in df.iterrows():
            query = (
                f"{row['question']}\n"
                "문제 풀이를 마친 후, 최종 정답을 다음 형식으로 작성해 주세요: \\boxed{{N}}."
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
    
    # ---- generate -----------------------------------------------------------
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling)
    df["response"] = [o.outputs[0].text for o in outputs]

    # ---- evaluate -----------------------------------------------------------
    if args.dataset in ['ArenaHard']:
        print('Starting Evaluation...')
        qrys = []
        for _,row in df.iterrows():
            response = row.response
            
            if args.reasoning:
                try:
                    response = row.response.split('</think>')[-1].strip()
                except:
                    response = "The model failed to return a response."
                
            query = [{'role':'system','content':system_prompt}]
            content = (
                "### Response A\n\n:"
                f"{row.ref}\n\n"
                "### Response B\n\n:"
                f"{response}\n\n"
            )
            query.append(
                {'role':'user','content':content}
            )
            qrys.append(query)
        responses = batch_completion(model='gpt-4.1',messages = qrys)
        df['judge'] = [safe_parse(res) for res in responses]
    
    # ---- save ---------------------------------------------------------------
    if args.output is None:
        safe_model = args.model.replace("/", "_")
        safe_data = args.dataset.replace("/", "_")
        args.output = f"{safe_data}--{safe_model}.csv"

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows ➜ {args.output}")


if __name__ == "__main__":
    main()



"""
Run OpenAI models on the generated world-model benchmark prompts.

Compatible with:
- wm_hiddenstate_benchmark_v2.py
- run_hiddenstate_inverse_ablation.py outputs

Main single-run use:
    python GPToutput_hiddenstate_v2.py \
        --prompts_dir wm_hiddenstate_main_outputs \
        --model gpt-4o

Ablation use: run every subfolder under the ablation root that contains prompts_all.jsonl:
    python GPToutput_hiddenstate_v2.py \
        --experiment_root wm_hiddenstate_ablation_outputs \
        --model gpt-4o

Outputs are written into the same prompt folder by default:
    all_llm_outputs.jsonl

Environment:
    export OPENAI_API_KEY="sk-..."
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm


TASK_FILES = {
    "forward": "prompts_forward.jsonl",
    "inverse": "prompts_inverse.jsonl",
    "rollout": "prompts_rollout.jsonl",
}


def load_existing_task_ids(output_file: Path, model: str) -> Set[str]:
    done: Set[str] = set()
    if not output_file.exists():
        return done
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if str(obj.get("model")) == model:
                done.add(str(obj.get("task_id")))
    return done


def iter_prompt_items(prompts_dir: Path, tasks: List[str], limit_per_task: Optional[int]) -> Iterable[Dict[str, object]]:
    for task_type in tasks:
        path = prompts_dir / TASK_FILES[task_type]
        if not path.exists():
            print(f"Warning: missing {path}; skipping {task_type}")
            continue
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                item.setdefault("task_type", task_type)
                yield item
                count += 1
                if limit_per_task is not None and count >= limit_per_task:
                    break


def call_openai_json(client, model: str, prompt: str, max_retries: int = 4, retry_sleep: float = 3.0) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise numerical prediction engine. Return only valid JSON matching the requested format.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or "{}"
        except Exception as exc:
            last_error = exc
            wait = retry_sleep * (2 ** attempt)
            print(f"OpenAI call failed on attempt {attempt + 1}/{max_retries}: {exc}. Waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_error}")


def run_prompts_dir(client, prompts_dir: Path, model: str, tasks: List[str], limit_per_task: Optional[int], overwrite: bool, output_name: str) -> Path:
    output_file = prompts_dir / output_name
    prompts_dir.mkdir(parents=True, exist_ok=True)
    done = set() if overwrite else load_existing_task_ids(output_file, model)

    items = list(iter_prompt_items(prompts_dir, tasks, limit_per_task))
    todo = [item for item in items if str(item["task_id"]) not in done]

    print(f"\nPrompt folder: {prompts_dir}")
    print(f"Model: {model}")
    print(f"Total prompts selected: {len(items)}; remaining to run: {len(todo)}")
    if not todo:
        print("Nothing to run.")
        return output_file

    mode = "w" if overwrite else "a"
    with output_file.open(mode, encoding="utf-8") as out:
        for item in tqdm(todo, desc=f"{prompts_dir.name}"):
            answer = call_openai_json(client, model, str(item["prompt"]))
            record = {
                "task_id": str(item["task_id"]),
                "task_type": str(item.get("task_type", "")),
                "difficulty": str(item.get("difficulty", "")),
                "model": model,
                "output": answer,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    print("Wrote:", output_file)
    return output_file


def find_prompt_dirs(experiment_root: Path) -> List[Path]:
    if (experiment_root / "prompts_all.jsonl").exists():
        return [experiment_root]
    dirs = []
    for child in sorted(experiment_root.iterdir()):
        if child.is_dir() and (child / "prompts_all.jsonl").exists():
            dirs.append(child)
    return dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI API on benchmark prompts")
    parser.add_argument("--prompts_dir", type=str, default=None, help="Folder containing prompts_forward/inverse/rollout.jsonl")
    parser.add_argument("--experiment_root", type=str, default=None, help="Root folder containing multiple prompt folders, for ablations")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_FILES.keys()), default=list(TASK_FILES.keys()))
    parser.add_argument("--limit_per_task", type=int, default=None, help="Smoke-test limit for each task type")
    parser.add_argument("--output_name", type=str, default="all_llm_outputs.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='sk-...' before using this script.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK is not installed. Run: pip install openai tqdm") from exc
    client = OpenAI()

    if args.prompts_dir is None and args.experiment_root is None:
        raise ValueError("Provide either --prompts_dir or --experiment_root")

    if args.prompts_dir is not None:
        prompt_dirs = [Path(args.prompts_dir)]
    else:
        prompt_dirs = find_prompt_dirs(Path(args.experiment_root))

    if not prompt_dirs:
        raise FileNotFoundError("No prompt folders found.")

    for d in prompt_dirs:
        run_prompts_dir(
            client=client,
            prompts_dir=d,
            model=args.model,
            tasks=args.tasks,
            limit_per_task=args.limit_per_task,
            overwrite=args.overwrite,
            output_name=args.output_name,
        )

    print("\nDone. Next step: re-run the benchmark with --llm_outputs pointing to the generated all_llm_outputs.jsonl file.")


if __name__ == "__main__":
    main()

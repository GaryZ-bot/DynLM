from openai import OpenAI
import json
from tqdm import tqdm

client = OpenAI()

MODEL = "gpt-4o"

TASK_FILES = {
    "forward":
    "simple_latent_material_wm_outputs/prompts_forward.jsonl",

    "inverse":
    "simple_latent_material_wm_outputs/prompts_inverse.jsonl",

    "rollout":
    "simple_latent_material_wm_outputs/prompts_rollout.jsonl"
}

all_results = []

for task_name, file_path in TASK_FILES.items():

    with open(file_path,"r") as f:

        for line in tqdm(
            f,
            desc=task_name
        ):

            item = json.loads(line)

            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={
                    "type":"json_object"
                },
                messages=[
                    {
                        "role":"user",
                        "content":item["prompt"]
                    }
                ]
            )

            answer = response.choices[0].message.content

            all_results.append({
                "task":task_name,
                "task_id":item["task_id"],
                "model":MODEL,
                "output":answer
            })

with open(
    "simple_latent_material_wm_outputs/all_llm_outputs.jsonl",
    "w"
) as f:

    for r in all_results:
        f.write(
            json.dumps(r)
            + "\n"
        )
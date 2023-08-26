# Author: Wei Lu (mailwlu@gmail.com)
from threading import Thread
from typing import Iterator
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = 'WizardLM/WizardCoder-15B-V1.0'
load_8bit = False

if torch.cuda.is_available():
    config = AutoConfig.from_pretrained(model_id)
    config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map='auto',
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.config.pad_token_id = tokenizer.pad_token_id

if not load_8bit:
    model.half()

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def get_prompt(message: str, chat_history: list[tuple[str, str]]) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{message}

### Response:"""


def get_input_token_length(message: str, chat_history: list[tuple[str, str]]) -> int:
    prompt = get_prompt(message, chat_history)
    input_ids = tokenizer(prompt, return_tensors='np', add_special_tokens=False)['input_ids']
    return input_ids.shape[-1]


def run(message: str,
        chat_history: list[tuple[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 50) -> Iterator[str]:
    prompt = get_prompt(message, chat_history)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).to('cuda')

    streamer = TextIteratorStreamer(tokenizer,
                                    timeout=10.,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs)

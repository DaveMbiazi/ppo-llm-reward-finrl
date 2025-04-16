import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import torch
from datasets import load_dataset
from transformers import TrainingArguments, HfArgumentParser, BitsAndBytesConfig
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel
from trl import RewardTrainer, RewardConfig, ModelConfig

# Define dataset sources.
DATASETS = {
     "flare_es_m2sum": "ChanceFocus/m2sum",
    "flare_es_german": "ChanceFocus/flare-german",
    "flare_es_fiqasa": "ChanceFocus/flare-fiqasa",
     "flare_es_instruction_tuning": "ChanceFocus/flare-es-instruction-tuning",
    "flare_es_stock": "ChanceFocus/flare-es-stock",
    "flare_es_fpb": "ChanceFocus/en-fpb",
     "flare_pubmedsum": "ChanceFocus/pubmedsum",
    "flare_headlines": "ChanceFocus/flare-headlines",
    "flare_finqa": "ChanceFocus/flare-finqa",
    "flare_convfinqa": "ChanceFocus/flare-convfinqa",
    "flare_ner": "ChanceFocus/flare-ner",
    "flare_cikm": "ChanceFocus/flare-sm-cikm",
    "flare_finer_acl": "ChanceFocus/flare-sm-acl",
}

print("Loading model and tokenizer...")
base_model = "meta-llama/Meta-Llama-3-8B"
peft_model = "FinGPT/fingpt-mt_llama3-8b_lora"

tokenizer = LlamaTokenizerFast.from_pretrained(
    base_model,
    trust_remote_code=True,
    legacy=False
)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_use_double_quant=False
)

model = LlamaForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quant_config
)

reward_model = PeftModel.from_pretrained(model, peft_model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    reward_model.config.pad_token_id = tokenizer.pad_token_id

reward_model.train()

for name, param in reward_model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

training_args = TrainingArguments(
    output_dir="./reward_model",
    per_device_train_batch_size=16,          
    gradient_accumulation_steps=16,           
    evaluation_strategy="steps",
    logging_steps=10,
    num_train_epochs=2,
    fp16=True,                            
    report_to=None,
)


if not hasattr(training_args, "disable_dropout"):
    training_args.disable_dropout = False
if not hasattr(training_args, "max_length"):
    training_args.max_length = 16
if not hasattr(training_args, "dataset_num_proc"):
    training_args.dataset_num_proc = 4
if not hasattr(training_args, "center_rewards_coefficient"):
    training_args.center_rewards_coefficient = 0.0

print("After setting training arguments.")


def select_chosen_and_rejected(example):
    prompt = example.get("query") or example.get("prompt", "")
    chosen_response = example.get("answer") or example.get("chosen", "")
    choices = example.get("choices") or example.get("choice", "")
    if not choices:
        return {"prompt": prompt, "chosen": chosen_response, "rejected": chosen_response}
    if chosen_response:
        rejected_candidates = [c for c in choices if c != chosen_response]
    else:
        if len(choices) >= 1:
            chosen_response = choices[0]
            rejected_candidates = choices[1:] if len(choices) > 1 else []
        else:
            rejected_candidates = []
    rejected_response = random.choice(rejected_candidates) if rejected_candidates else chosen_response
    return {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response}


def dummy_compute_metrics(eval_preds):
    return {"dummy_metric": 0.0}


parser = HfArgumentParser((RewardConfig, ModelConfig))
reward_config, model_config = parser.parse_args_into_dataclasses()
print("After parsing reward and model configuration.")

for task, dataset_identifier in DATASETS.items():
    try:
        print(f"\nTraining Reward Model for task: {task} using dataset: {dataset_identifier}...")
        dataset = load_dataset(dataset_identifier)
        print("Loaded dataset:", dataset)
    
        if "train" not in dataset:
            print(f"Dataset {dataset_identifier} doesn't have a 'train' split. Skipping...")
            continue
    
        train_dataset = dataset["train"].map(select_chosen_and_rejected)
        eval_dataset = None
        if training_args.evaluation_strategy != "no" and "test" in dataset:
            eval_dataset = dataset["test"].map(select_chosen_and_rejected)
    
        # Initialize the RewardTrainer.
        reward_trainer = RewardTrainer(
            args=training_args,
            model=reward_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=dummy_compute_metrics,
        )
    
        reward_trainer.visualize_samples = lambda num_print_samples: None
    
        print(f"Starting training on task: {task}...")
        reward_trainer.train()
        print(f"Finished training on task: {task}.")
    
        reward_trainer.save_model(f"./reward_model_final_{task}")
        reward_model = reward_trainer.model 
        
    except Exception as e:
        print(f"Error encountered for dataset {dataset_identifier}: {e}\nSkipping this dataset and moving to the next one.")
        continue

reward_trainer.save_model("./reward_model_final")
print("Final model saved.")

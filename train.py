from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.trainer_utils import SaveStrategy
from datasets import load_dataset, Features, Value
import wandb

# Wandb 初始化
wandb.init(project="Qwen2.5-7B-Instruct-Lora-FineTuning-Exp", name="25-3-24/2")  # 你可以自定义项目名称和运行名称
#本地模型目录
cache_dir = '/root/.cache/modelscope/hub/models'
model_name = f'{cache_dir}/Qwen/Qwen2.5-7B-Instruct'

#加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    model_type='qwen2',
    cache_dir=cache_dir
)

# 训练数据集
train_dataset_path = './datasets/alpaca_train_dataset.jsonl'

EOS_TOKEN = tokenizer.eos_token # 必须添加 EOS_TOKEN

_train_features = Features({
    'instruction': Value('string'),
    'input': Value('string'),
    'output': Value('string')
})

train_dataset = load_dataset(
    "json", 
    data_files=train_dataset_path, 
    split="train", 
    features=_train_features,
)

def formatting_alpaca_prompts_func(example):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {}
    ### Input:
    {}
    ### Output:
    {}"""

    instruction = example['instruction']
    input_text = example['input']
    output_text = example['output']
    
    alpaca_prompt = alpaca_prompt.format(instruction, input_text, output_text)
    
    return {"text": alpaca_prompt}

train_dataset = train_dataset.map(formatting_alpaca_prompts_func, remove_columns=['instruction','input','output'])

#设置训练参数
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=322,
    max_seq_length=max_seq_length,
    use_rslora=False,
    loftq_config=None,
)

checkpoint_save_path='/data/healthai/checkpoints'
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # max_steps=60, # 训练迭代次数
        num_train_epochs=3, # 基于 Epoch 训练
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy=SaveStrategy.EPOCH,
        output_dir=checkpoint_save_path, # checkpoints 存放目录
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=322,
        report_to="wandb",  #  启用 Wandb 集成
    ),
)

#开始训练
trainer.train()

#继续上一次检查点训练
# trainer.train(resume_from_checkpoint=True)

#保存微调模型
model.save_pretrained("Qwen-2.5-7B-Instruct-Lora")
tokenizer.save_pretrained("Qwen-2.5-7B-Instruct-Lora")

# Wandb 结束
wandb.finish()
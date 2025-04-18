from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
from transformers import TextStreamer


model_name = "../models/gemma-3-4b-it"

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048, # 可以选择任意长度用于长文本处理！
    full_finetuning = False, # [新功能！] 我们现在支持全参数微调！
    # token = "hf_...", # 如果使用需要授权的模型，请在此处添加token
)

# 使用LoRA，以便只更新模型中的少量参数！
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # 对于纯文本任务关闭视觉层微调！
    finetune_language_layers   = True,  # 应该保持开启！
    finetune_attention_modules = True,  # 注意力机制对GRPO很有效
    finetune_mlp_modules       = True,  # 应该始终保持开启！

    r = 8,           # 更大的值 = 更高的准确率，但可能过拟合
    lora_alpha = 8,  # 建议alpha至少等于r
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


dataset = load_dataset('wmt18', 'zh-en', split = "train[:30000]")
print('原始数据集 第10条数据：',dataset[10])
# {'translation': {'en': 'The end of the East-West ideological divide and the end of absolute faith in markets are historical turning points.',
#   'zh': '东西方意识形态鸿沟的结束，以及对市场绝对信心的后果，都是历史的转折点。'}}


# 格式化数据集，将中英文翻译对转换为对话格式
def format_translation_to_chat(example):
    # 使用英文作为用户输入，英文作为模型输出（按照指令要求）
    conversations = [
        {"role": "system", "content": "You are a helpful assistant. help me translate the text from English to Chinese."},
        {"role": "user", "content": example["translation"]["en"]},
        {"role": "assistant", "content": example["translation"]["zh"]}
    ]
    return {"conversations": conversations}

# 应用格式化函数
dataset = dataset.map(format_translation_to_chat)
print('格式化后 第10条数据：',dataset[10])

# {'translation': {'en': 'The end of the East-West ideological divide and the end of absolute faith in markets are historical turning points.',
#   'zh': '东西方意识形态鸿沟的结束，以及对市场绝对信心的后果，都是历史的转折点。'},
#  'conversations': [{'from': 'user', 'value': 'The end of the East-West ideological divide and the end of absolute faith in markets are historical turning points.'},
#   {'from': 'gpt', 'value': 'The end of the East-West ideological divide and the end of absolute faith in markets are historical turning points.'}],
#  'text': ''}

# 我们现在使用`standardize_data_formats`来尝试将数据集转换为适合微调的正确格式！
dataset = standardize_data_formats(dataset)

# 现在我们需要将Gemma-3的聊天模板应用到conversations，并保存到text
def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return { "text" : texts }

dataset = dataset.map(apply_chat_template, batched = True)
print('标准格式化后 第100条数据：',dataset[100]["text"])


# 使用Huggingface TRL的SFTTrainer！更多文档请参考：TRL SFT文档。我们执行50步以加快训练速度，但您可以设置num_train_epochs=1进行完整训练，并关闭max_steps=None。
print('开始训练')
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # 可以设置评估数据集！
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # 使用梯度累积来模拟更大的批量大小！
        warmup_steps = 5,
        # num_train_epochs = 1, # 设置此项进行完整训练。
        max_steps = 50,
        learning_rate = 2e-4, # 对于长时间训练，可以降低到2e-5
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # 使用此项可以启用WandB等工具
    ),
)


# 使用Unsloth的train_on_completions方法，仅对助手的输出进行训练，而忽略用户输入的损失。这有助于提高微调的准确性！
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


if __name__ == "__main__":
    print('训练中...')
    trainer_stats = trainer.train()
    print('训练结束')

    print('保存模型')
    # 保存 lora
    model.save_pretrained("../models/gemma-3-4b-it-finetune-lora", tokenizer)  # Local saving
    tokenizer.save_pretrained("../models/gemma-3-4b-it-finetune-lora") 

    # 保存完整模型
    # model.save_pretrained_merged("../models/gemma-3-4b-it-finetune", tokenizer)

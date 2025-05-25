import os
import json
import logging
import argparse
from typing import Dict, List, Any, Optional

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging as transformers_logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from datasets import load_dataset
from trl import SFTTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_info()

def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Дообучение модели Llama-2-7b-chat-hf")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Llama-2-7b-chat-hf", 
        help="Название базовой модели из HuggingFace"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Путь к JSONL-файлу с данными для дообучения"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./llama2_finetuned_model", 
        help="Директория для сохранения дообученной модели"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Количество эпох обучения"
    )
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=4, 
        help="Размер батча для обучения на одном устройстве"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=2, 
        help="Количество шагов для накопления градиента"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-4, 
        help="Скорость обучения"
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16, 
        help="Ранг для LoRA адаптеров"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32, 
        help="Альфа для LoRA"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05, 
        help="Dropout для LoRA"
    )
    
    return parser.parse_args()

def prepare_dataset(data_path: str):
    """
    Подготовка набора данных для дообучения из JSONL-файла.
    
    Args:
        data_path: Путь к JSONL-файлу с парами вопрос-ответ
    
    Returns:
        Dataset: Подготовленный набор данных
    """
    # Загружаем данные из JSONL-файла
    dataset = load_dataset("json", data_files=data_path)
    
    # Преобразуем данные в формат необходимый для SFTTrainer
    def format_instruction(example):
        conversations = example["conversations"]
        formatted = {"input": conversations[0]["value"], "output": conversations[1]["value"]}
        return formatted
    
    return dataset["train"].map(format_instruction)

def prepare_model_and_tokenizer(args):
    """
    Подготовка модели и токенизатора для дообучения.
    
    Args:
        args: Аргументы командной строки
    
    Returns:
        tuple: Подготовленная модель и токенизатор
    """
    # Настройка квантизации для уменьшения использования памяти
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Загрузка модели с квантизацией
    logger.info(f"Загрузка модели {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Подготовка модели для обучения с k-битной квантизацией
    model = prepare_model_for_kbit_training(model)
    
    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Настройка LoRA для эффективного дообучения
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    return model, tokenizer, peft_config

def train_model(args, model, tokenizer, peft_config, dataset):
    """
    Дообучение модели с использованием SFTTrainer.
    
    Args:
        args: Аргументы командной строки
        model: Модель для дообучения
        tokenizer: Токенизатор
        peft_config: Конфигурация PEFT
        dataset: Набор данных для дообучения
    
    Returns:
        SFTTrainer: Обученный тренер
    """
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=3,
        report_to="tensorboard",
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="input",
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=2048,
    )
    
    logger.info("Начало дообучения модели...")
    trainer.train()
    
    # Сохраняем модель и адаптеры
    logger.info(f"Сохранение модели в {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    return trainer

def test_model(args, trainer):
    """
    Тестирование дообученной модели.
    
    Args:
        args: Аргументы командной строки
        trainer: Обученный тренер
    """
    logger.info("Тестирование дообученной модели...")
    
    # Загружаем PEFT модель для инференса
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ),
        args.output_dir,
        torch_dtype=torch.float16,
    )
    
    # Для инференса можно объединить базовую модель и адаптеры
    merged_model = model.merge_and_unload()
    
    # Загружаем токенизатор для инференса
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    
    # Создаем pipeline для проверки
    pipe = pipeline(
        "text-generation",
        model=merged_model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
    )
    
    # Тестовый вопрос
    test_question = "Нужно ли обращаться к нотариусу при дарении доли квартиры?"
    
    logger.info(f"Тестовый вопрос: {test_question}")
    
    result = pipe(f"<s>[INST] {test_question} [/INST]")
    
    logger.info(f"Ответ модели: {result[0]['generated_text']}")

def main():
    """Основная функция"""
    # Парсинг аргументов
    args = parse_args()
    
    # Проверка наличия GPU
    if not torch.cuda.is_available():
        logger.warning("ВНИМАНИЕ: GPU не обнаружен! Обучение будет очень медленным на CPU.")
    else:
        logger.info(f"Доступно GPU устройств: {torch.cuda.device_count()}")
    
    # Подготовка данных
    logger.info(f"Загрузка данных из {args.data_path}...")
    dataset = prepare_dataset(args.data_path)
    logger.info(f"Загружено {len(dataset)} примеров для обучения")
    
    # Подготовка модели и токенизатора
    model, tokenizer, peft_config = prepare_model_and_tokenizer(args)
    
    # Дообучение модели
    trainer = train_model(args, model, tokenizer, peft_config, dataset)
    
    # Тестирование модели
    test_model(args, trainer)
    
    logger.info("Дообучение завершено успешно!")

if __name__ == "__main__":
    main()
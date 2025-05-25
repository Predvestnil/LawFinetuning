#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для дообучения модели Llama-2 на парах вопрос-ответ из юридических форумов.
"""

import os
import argparse
import logging
import torch
from finetuning import (
    prepare_dataset,
    prepare_model_and_tokenizer,
    train_model,
    create_generation_pipeline,
    generate_response
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Дообучение модели Llama-2-7b-chat-hf на юридических вопросах и ответах"
    )
    
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
        default="./models/llama2_legal", 
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
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Запустить тестирование модели после обучения"
    )
    
    return parser.parse_args()

def test_model(model_name, output_dir):
    """
    Тестирование дообученной модели.
    
    Args:
        model_name: Название базовой модели
        output_dir: Директория с сохраненными адаптерами
    """
    logger.info("Тестирование дообученной модели...")
    
    try:
        # Загружаем базовую модель
        model = torch.load(
            AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ),
        )
        
        # Загружаем адаптеры LoRA
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            output_dir,
            torch_dtype=torch.float16,
        )
        
        # Для инференса можно объединить базовую модель и адаптеры
        merged_model = model.merge_and_unload()
        
        # Загружаем токенизатор для инференса
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Создаем pipeline для проверки
        pipe = create_generation_pipeline(
            merged_model,
            tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
        )
        
        # Тестовые вопросы
        test_questions = [
            "Нужно ли обращаться к нотариусу при дарении доли квартиры?",
            "Как оформить наследство после смерти родителей?",
            "Какие документы нужны для оформления развода?",
        ]
        
        for question in test_questions:
            logger.info(f"Тестовый вопрос: {question}")
            response = generate_response(pipe, question)
            logger.info(f"Ответ модели: {response}")
            logger.info("-" * 50)
            
    except Exception as e:
        logger.error(f"Ошибка при тестировании модели: {e}")

def main():
    """Основная функция"""
    args = parse_args()
    
    # Проверяем наличие GPU
    if not torch.cuda.is_available():
        logger.warning("ВНИМАНИЕ: GPU не обнаружен! Обучение будет очень медленным на CPU.")
    else:
        logger.info(f"Доступно GPU устройств: {torch.cuda.device_count()}")
    
    # Проверяем наличие файла с данными
    if not os.path.exists(args.data_path):
        logger.error(f"Файл с данными {args.data_path} не существует!")
        return 1
    
    try:
        # Подготовка данных
        logger.info(f"Загрузка данных из {args.data_path}...")
        dataset = prepare_dataset(args.data_path)
        logger.info(f"Загружено {len(dataset)} примеров для обучения")
        
        # Подготовка модели и токенизатора
        model, tokenizer, peft_config = prepare_model_and_tokenizer(
            args.model_name,
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout
        )
        
        # Дообучение модели
        trainer = train_model(
            model,
            tokenizer,
            dataset,
            peft_config,
            args.output_dir,
            args.num_train_epochs,
            args.per_device_train_batch_size,
            args.gradient_accumulation_steps,
            args.learning_rate
        )
        
        # Тестирование модели если указан флаг --test
        if args.test:
            test_model(args.model_name, args.output_dir)
        
        logger.info("Дообучение завершено успешно!")
        return 0
        
    except Exception as e:
        logger.error(f"Произошла ошибка при дообучении модели: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

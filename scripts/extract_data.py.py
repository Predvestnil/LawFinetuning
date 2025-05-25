#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для извлечения пар вопрос-ответ из HTML-файлов юридических форумов.
"""

import os
import argparse
import logging
from extraction import process_html_files

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Извлечение пар вопрос-ответ из HTML-файлов юридических форумов"
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Директория с HTML-файлами"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Путь к файлу для сохранения результатов"
    )
    
    return parser.parse_args()

def main():
    """Основная функция"""
    args = parse_args()
    
    # Проверяем наличие директории с HTML-файлами
    if not os.path.exists(args.input_dir):
        logger.error(f"Директория {args.input_dir} не существует!")
        return 1
    
    # Создаем директорию для вывода, если она не существует
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    logger.info(f"Начало обработки HTML-файлов из директории: {args.input_dir}")
    logger.info(f"Результат будет сохранен в: {args.output_path}")
    
    try:
        # Обрабатываем HTML-файлы и извлекаем пары вопрос-ответ
        process_html_files(args.input_dir, args.output_path)
        logger.info("Обработка завершена успешно!")
        return 0
    except Exception as e:
        logger.error(f"Произошла ошибка при обработке файлов: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

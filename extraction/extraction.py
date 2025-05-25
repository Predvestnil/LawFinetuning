import os
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Any, Optional

def extract_qa_pairs(html_files_dir: str) -> List[Dict[str, Any]]:
    """
    Извлекает пары вопрос-ответ из HTML-файлов, где рейтинг автора ответа выше 4.5.
    
    Args:
        html_files_dir: Путь к директории с HTML-файлами
        
    Returns:
        List[Dict[str, Any]]: Список пар вопрос-ответ в формате для дообучения Llama-2-7b-chat-hf
    """
    qa_pairs = []
    
    # Перебираем все HTML-файлы в указанной директории
    for filename in os.listdir(html_files_dir):
        if not filename.endswith('.html'):
            continue
            
        file_path = os.path.join(html_files_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            
        # Парсим HTML с помощью BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Извлекаем вопрос
        question_title = soup.find('h1', class_='question9111ru__h1')
        if not question_title:
            continue
        
        question_text_div = soup.find('div', class_='question__text')
        if not question_text_div:
            continue
        
        question_content = question_text_div.find(id=lambda x: x and 'question-content' in x)
        if not question_content:
            continue
        
        # Получаем текст вопроса
        question_title_text = question_title.get_text(strip=True)
        question_body_text = ""
        
        # Ищем текст вопроса внутри div с border-left
        question_body_div = question_content.find('div', style=lambda x: x and 'border-left' in x)
        if question_body_div and question_body_div.find_all('p'):
            for p in question_body_div.find_all('p'):
                question_body_text += p.get_text(strip=True) + "\n"
        
        question_text = question_title_text + "\n\n" + question_body_text.strip()
        
        # Извлекаем все ответы
        answers = soup.find_all('div', class_='answer9111ru')
        
        for answer in answers:
            # Извлекаем рейтинг автора
            rating_element = answer.find('a', class_='ratingBlock__value')
            if not rating_element:
                continue
                
            # Извлекаем числовое значение рейтинга
            rating_text = rating_element.get_text(strip=True)
            try:
                rating = float(rating_text)
            except ValueError:
                continue
                
            # Проверяем, что рейтинг больше 4.5
            if rating <= 4.5:
                continue
                
            # Извлекаем информацию об авторе
            author_element = answer.find('a', class_='answer9111ru__user__name')
            status_element = answer.find('div', class_='answer9111ru__user__status')
            
            author_name = author_element.get_text(strip=True) if author_element else "Неизвестный автор"
            author_status = status_element.get_text(strip=True) if status_element else "Неизвестный статус"
                
            # Извлекаем текст ответа
            answer_text_div = answer.find('div', class_='answer9111ru__text')
            if not answer_text_div:
                continue
                
            answer_text = ""
            for p in answer_text_div.find_all('p'):
                answer_text += p.get_text(strip=True) + "\n"
                
            answer_text = answer_text.strip()
            if not answer_text:
                continue
                
            # Формируем пару вопрос-ответ в формате для дообучения Llama-2-7b-chat-hf
            qa_pair = format_for_llama2(
                question=question_text,
                answer=answer_text,
                author_name=author_name,
                author_status=author_status,
                rating=rating
            )
            
            qa_pairs.append(qa_pair)
    
    return qa_pairs

def format_for_llama2(question: str, answer: str, author_name: str, author_status: str, rating: float) -> Dict[str, Any]:
    """
    Форматирует пару вопрос-ответ в соответствии с требованиями для дообучения Llama-2-7b-chat-hf.
    
    Формат Llama-2-7b-chat-hf:
    {
        "conversations": [
            {"from": "human", "value": "ВОПРОС"},
            {"from": "gpt", "value": "ОТВЕТ"}
        ]
    }
    
    Args:
        question: Текст вопроса
        answer: Текст ответа
        author_name: Имя автора ответа
        author_status: Статус автора ответа
        rating: Рейтинг автора
        
    Returns:
        Dict[str, Any]: Пара вопрос-ответ в формате для Llama-2-7b-chat-hf
    """
    # Добавляем информацию об авторе в начало ответа
    formatted_answer = f"[{author_status}, рейтинг: {rating}]\n\n{answer}"
    
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": formatted_answer}
        ]
    }

def save_qa_pairs(qa_pairs: List[Dict[str, Any]], output_path: str) -> None:
    """
    Сохраняет пары вопрос-ответ в формате JSONL для дообучения.
    
    Args:
        qa_pairs: Список пар вопрос-ответ
        output_path: Путь к файлу для сохранения
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"Сохранено {len(qa_pairs)} пар вопрос-ответ в {output_path}")

def process_html_files(input_dir: str, output_path: str) -> None:
    """
    Обрабатывает все HTML-файлы в директории и сохраняет пары вопрос-ответ.
    
    Args:
        input_dir: Директория с HTML-файлами
        output_path: Путь к файлу для сохранения
    """
    qa_pairs = extract_qa_pairs(input_dir)
    save_qa_pairs(qa_pairs, output_path)
    
    print(f"Всего извлечено: {len(qa_pairs)} пар вопрос-ответ")

# Пример использования
if __name__ == "__main__":
    input_directory = "./html_files"  # Директория с HTML-файлами
    output_file = "./qa_pairs_for_llama.jsonl"  # Файл для сохранения пар вопрос-ответ
    
    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    process_html_files(input_directory, output_file)
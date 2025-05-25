# LegalLlama Finetune

Проект для извлечения пар вопрос-ответ из юридических форумов и дообучения модели Llama 2 для ответов на юридические вопросы.

## Описание проекта

LegalLlama Finetune - это инструмент для сбора и обработки данных из юридических форумов, а также для дообучения языковой модели Llama 2 на этих данных. Проект позволяет:

1. Извлекать пары вопрос-ответ из HTML-файлов юридических форумов
2. Фильтровать ответы на основе рейтинга авторов
3. Форматировать данные для дообучения модели
4. Выполнять дообучение модели Llama 2 с использованием LoRA адаптеров
5. Тестировать полученную модель

## Структура проекта

```
LegalLlama-Finetune/
├── extraction/
│   ├── __init__.py
│   └── extraction.py          # Код для извлечения пар вопрос-ответ
├── finetuning/
│   ├── __init__.py
│   └── finetuning.py          # Код для дообучения модели
├── scripts/
│   ├── extract_data.py        # Скрипт для извлечения данных
│   └── train_model.py         # Скрипт для запуска дообучения
├── data/
│   ├── html_files/            # Директория для HTML-файлов
│   └── qa_pairs/              # Директория для извлеченных пар вопрос-ответ
├── models/                    # Директория для сохранения дообученной модели
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

## Требования

- Python 3.8 или выше
- CUDA-совместимая видеокарта с памятью не менее 8 ГБ (рекомендуется 16+ ГБ)
- Hugging Face токен для доступа к модели Llama 2

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/LegalLlama-Finetune.git
cd LegalLlama-Finetune
```

2. Создайте и активируйте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Войдите в свой аккаунт Hugging Face:
```bash
huggingface-cli login
```

## Использование

### Извлечение данных

1. Поместите HTML-файлы форумов в директорию `data/html_files/`
2. Запустите скрипт извлечения данных:
```bash
python scripts/extract_data.py --input_dir data/html_files --output_path data/qa_pairs/legal_qa_pairs.jsonl
```

### Дообучение модели

```bash
python scripts/train_model.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --data_path data/qa_pairs/legal_qa_pairs.jsonl \
    --output_dir models/llama2_legal \
    --num_train_epochs 3 
```

### Дополнительные параметры дообучения

- `--per_device_train_batch_size`: Размер батча для обучения (по умолчанию 4)
- `--gradient_accumulation_steps`: Шаги накопления градиента (по умолчанию 2)
- `--learning_rate`: Скорость обучения (по умолчанию 2e-4)
- `--lora_r`: Ранг для LoRA адаптеров (по умолчанию 16)
- `--lora_alpha`: Альфа для LoRA (по умолчанию 32)
- `--lora_dropout`: Dropout для LoRA (по умолчанию 0.05)

## Пример использования дообученной модели

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

# Загрузка модели и токенизатора
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, "models/llama2_legal")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Создание пайплайна для генерации текста
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15,
)

# Пример запроса
question = "Нужно ли обращаться к нотариусу при дарении доли квартиры?"
result = pipe(f"<s>[INST] {question} [/INST]")
print(result[0]['generated_text'])
```

## Лицензия

MIT
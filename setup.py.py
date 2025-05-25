from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="legal-llama-finetune",
    version="0.1.0",
    author="Ваше имя",
    author_email="ваш.email@example.com",
    description="Инструмент для извлечения юридических вопросов и ответов и дообучения Llama 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/LegalLlama-Finetune",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)

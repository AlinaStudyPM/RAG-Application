import re
import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict

import pytesseract
from PIL import Image
import torch
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModel

import chromadb
from chromadb.config import Settings

import requests
import json

#-------------------------------------------
#              Parameters
#-------------------------------------------

# Задаём параметры
EMBEDDINGS_MODEL = 'intfloat/multilingual-e5-small' # Название модели, которая превращает текст в векторное представление
CHROMA_DB_PATH = './embeddings_chroma_db'           # Название создаваемой базы данных
CHROMA_COLLECTION_NAME = 'geological_embedddings'   # Название коллекции в базе данных
OLLAMA_MODEL = 'gemma3:4b'                          # Название модели нейросети ('deepseek-r1:latest', 'gemma3:4b')
TXT_FOLDER_PATH = 'OutputFiles'                     # Папка с промежуточными файлами (для отладки)

os.environ["PATH"] += os.pathsep + r"C:\poppler\poppler-25.07.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#-------------------------------------------
#            PDF-Recognizer
#-------------------------------------------

# Извлекает текст из PDF-файла, разделяя его на абзацы.
def extract_text_from_pdf(pdf_file_path: str) -> List[str]:
    paragraphs = []
    current_text = ""

    images = convert_from_path(pdf_file_path)
    for image in images:
        new_text = pytesseract.image_to_string(image, lang='eng+rus')
        new_text = (new_text.replace("-\n ", "")
                   .replace("-\n", "")
                   .strip())
        current_text += " " + new_text

        split_text = re.split(r'\.\n|\n\n', current_text)
        split_text = [p.strip() for p in split_text if p.strip()]
        if len(split_text) > 0:
            current_text = split_text[-1].strip()
        else:
            current_text = ""

        for p in split_text[:-1]:
            if p.strip() != "":
                paragraphs.append(p.replace("\n", " ").strip() + ".")
        if current_text.endswith("."):
            paragraphs.append(current_text.replace("\n", " ").strip() + ".")
        current_text = ""

    if current_text != "":
        paragraphs.append(current_text)
    return paragraphs

# Получает на вход путь к папке и возвращает список файлов заданного типа.
def get_files_names_from_folder(folder_url:str, type:str)  -> List[str]:
    paths_to_files = []

    if not os.path.exists(folder_url):
        raise FileNotFoundError(f"Папка {folder_url} не найдена.")

    for filename in os.listdir(folder_url):
        if filename.endswith(type):
            paths_to_files.append(filename)
    return paths_to_files

# Генерирует векторные представления (эмбеддинги) для списка текстов с пакетной обработкой.
def generate_embeddings(model:str, texts: List[str], batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model).to(device)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    torch.cuda.empty_cache()
    return embeddings

# Добавляет тексты и их эмбеддинги в базу данных Chroma.
def add_record_to_chroma(db_path:str,
                        collection_name:str,
                        book_name:str,
                        texts: List[str],
                        text_embeddings):
    db_client = chromadb.PersistentClient(db_path)
    collection = db_client.get_or_create_collection(name=collection_name)

    existing_ids = collection.get()["ids"]
    max_id = max(int(id_) for id_ in existing_ids) if existing_ids else -1
    ids = [str(max_id + 1 + i) for i in range(len(texts))]

    metadatas = [{"source": book_name, "hnsw:space": "cosine"} for _ in range(len(texts))]

    collection.upsert(
        ids=ids, # настроить нумерацию
        embeddings=text_embeddings,
        metadatas=metadatas,
        documents=texts
    )

# Проверяет наличие записей из данного файла в базе данных
def is_in_chroma(db_path: str, collection_name: str, book_name: str) -> bool:
    client = chromadb.PersistentClient(db_path)
    collection = client.get_collection(name=collection_name)
    hits = collection.get(where={"source": book_name}, limit=1)
    return len(hits["ids"]) > 0

# Удаляет из базы данных все записи, соответсвующие данному файлу
def delete_from_chroma(db_path: str, collection_name: str, book_name: str) -> None:
    client = chromadb.PersistentClient(db_path)
    collection = client.get_collection(name=collection_name)
    collection.delete(where={"source": book_name})

# Выводит список уникальных названий файлов из базы данных
def list_files_in_chroma(db_path: str, collection_name: str):
    client = chromadb.PersistentClient(db_path)
    collection = client.get_collection(name=collection_name)
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    file_names = {m.get("source") for m in metadatas if m.get("source")}
    for name in sorted(file_names):
        print(name)
        

#  Форматирует результаты поиска в удобочитаемую строку.
def format_results(data: dict, n: int) -> str:
    documents = data['documents'][0]
    metadatas = data['metadatas'][0]
    result = [f"Топ {n} подходящих абзацев:", ""]
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        source = meta.get('source', 'Неизвестный источник')
        text = doc
        result.append(f"{i}) Источник: {source}")
        result.append(f"    Текст: {text}")
        result.append("")
    return "\n".join(result)

# Загружает файл
def upload_file(file_path: str):
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return
    
    file_name = os.path.basename(file_path)
    print(f'\nОбрабатывается файл {file_name}')
    if is_in_chroma(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, file_name):
        print('В базе данных уже есть файл с таким же названием. Хотите его обновить? Y/N')
        choice = input().strip().upper()
        if choice == 'Y':
            delete_from_chroma(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, file_name)
            print('Старые данные удалены. Продолжаем обработку...')
        else:
            print('Обработка отменена.')
            return
        
    texts = extract_text_from_pdf(file_path)
    embeddings = generate_embeddings(EMBEDDINGS_MODEL, texts)
    add_record_to_chroma(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, file_name, texts, embeddings)
    print(f'Файл {file_name} успешно добавлен\n')

# Загружает все файлы из папки
def upload_folder(folder_path: str):
    for file_name in get_files_names_from_folder(folder_path, ".pdf"):
        file_path = os.path.join(folder_path, file_name)
        upload_file(file_path)

#-------------------------------------------
#            Answer Generator
#-------------------------------------------

# Выполняет поиск похожих текстов в базе данных Chroma.
def chroma_query(db_path:str,
                 collection_name:str,
                 model:str,
                 searched_text:str,
                 top_k: int = 10):
    query_embedding = generate_embeddings(model, [searched_text])[0]
    db_client = chromadb.PersistentClient(db_path)
    collection = db_client.get_or_create_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results # Dict[str, List[Union[List[Any], List[Dict[str, Any]]]]]

# Формирует запрос для нейросети на основе контекста
def ollama_query(model: str,
                 input: str, context: str) -> str:
    #context_str = "\n\n".join([f"Контекст {i+1}: {text}" for i, text in enumerate(context)])
    full_prompt = f"""Ты — профессор геологии, академик Российской академии наук с 40-летним стажем.
        Отвечай строго на русском языке, сохраняя научную точность. Пиши в обезличенной форме без вступительных слов.
        Используй специальные термины на английском только когда это необходимо (например, "black shale"),
        а также химические формулы (например, Au(CN)₂⁻).

        Инструкции:
        1. Твоя главная задача дать ответ на вопрос.
        2. Используй академический стиль изложения.
        3. Для дат используй формат: "в 1920-х годах"
        4. Цитируй источники когда возможно, например "Согласно исследованиям Петрова (2015)..."
        5. Если информации недостаточно, отвечай: "В доступных источниках нет достаточных данных"
        6. После ответа отдельно напиши "Источники:" и перечисли в виде списка дословно названия книг-источников из контекста, 
        которые использовались в контексте.

        Контекст для ответа:
        {context}

        Вопрос: {input}
        """

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return json.loads(response.text)["response"]
    except Exception as e:
        return f"Ошибка при запросе к Ollama: {str(e)}"
    
# Формирует контекст для запроса пользователя и отправляет запрос нейросети
def user_query(ollama_model: str,
               embeddings_model: str,
               db_path: str,
               db_collection: str,
               question: str,
               top_k: int = 5) -> str:
    search_results = chroma_query(db_path, db_collection, embeddings_model, question, top_k)
    context = format_results(search_results, top_k)
    answer = ollama_query(ollama_model, question, context)
    return answer

#------------------------------------------
#                Testing
#------------------------------------------

# (Для тестирования) Читает текст из TXT-файла и разделяет его на абзацы.
def read_paragraphs_from_txt(file_path:str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        paragraphs = file.read().split("\n\n")
        paragraphs = [p for p in paragraphs if p.strip()]
    return paragraphs

# (Для тестирования) По тексту выдаёт похожие записи из базы данных
def test_context():
    print("Искомый текст: ")
    searched_text = input()
    number_of_responses = 20
    response = chroma_query(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, EMBEDDINGS_MODEL, searched_text, number_of_responses)
    print("Результат:")
    result = format_results(response, number_of_responses)
    print(result)

#------------------------------------------
#                Launching
#------------------------------------------

# Запускает интерактивный цикл и ожидает команд
def start():
    print("Список всех команд доступ через команду \\help.")
    while True:
        command = input("\nВведите команду: ").strip()
        if command == "\\upload":
            file_path = input("Введите путь к файлу: ")
            upload_file(file_path)
        elif command == "\\upload_folder":
            folder_path = input("Введите путь к папке: ")
            upload_folder(folder_path)
        elif command == "\\list_files":
            list_files_in_chroma(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)
        elif command == "\\chat":
            question = input("Введите ваш запрос: ").strip()
            answer = user_query(OLLAMA_MODEL, EMBEDDINGS_MODEL, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, question)
            print("\nОтвет:")
            print(answer)
        elif command == "\\exit":
            print("Выход из программы.")
            break
        elif command.strip() == "\\help":
            print("Доступные команды:")
            print(" \\upload - загрузить файл pdf")
            print(" \\upload_folder - загрузить файлы pdf из папки")
            print(" \\list_files - посмотреть загруженные файлы")
            print(" \\chat - отправить запрос чат-боту")
            print(" \\exit - выход из программы")
            print(" \\help - показать эту справку")
        else:
            print("Неправильная команда. Используйте \\help для справки.")

if __name__=="__main__":
    start()
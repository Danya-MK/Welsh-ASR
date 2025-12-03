import torch
import pandas as pd
import librosa
import os
import re
import evaluate
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor
)
from datasets import Dataset

# ================= КОНФИГУРАЦИЯ =================

MODEL_ID = "facebook/wav2vec2-large-xlsr-53" 
CSV_FILE_PATH = "C:/Proekt/s_nulya/test/transcripts_output_wav.csv"
AUDIO_FOLDER_PATH = "C:/Proekt/s_nulya/test/welsh_wav_test"

# ❗️ ИСПРАВЛЕНИЕ: Путь к вашему локальному vocab.json
VOCAB_FILE_PATH = "C:/Proekt/s_nulya/vocab.json" 

# Фактические имена столбцов
COL_FILENAME = 'Название файла'
COL_TRANSCRIPT = 'Расшифровка'

# ================= ГЛАВНАЯ ФУНКЦИЯ =================

def main():
    
    ## 1. Загрузка и подготовка данных
    print("Загрузка датасета...")
    
    try:
        df = pd.read_csv(CSV_FILE_PATH, sep=',')
    except Exception as e:
        print(f"❌ Ошибка при чтении CSV. Проверьте путь или разделитель. Ошибка: {e}")
        return

    if COL_FILENAME not in df.columns or COL_TRANSCRIPT not in df.columns:
        print(f"❌ Ошибка: Не найдены ожидаемые столбцы '{COL_FILENAME}' и '{COL_TRANSCRIPT}'.")
        print(f"Фактические столбцы: {list(df.columns)}")
        return

    df = df[[COL_FILENAME, COL_TRANSCRIPT]].dropna()
    df = df.rename(columns={COL_FILENAME: 'path', COL_TRANSCRIPT: 'sentence'})
    df['path'] = [os.path.join(AUDIO_FOLDER_PATH, x.replace('.mp3', '.wav')) for x in df['path']]
    
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\\(\)]'
    def clean_text(text):
        return re.sub(chars_to_ignore_regex, '', str(text)).lower() + " " 

    df['sentence'] = df['sentence'].apply(clean_text)

    initial_rows = len(df)
    df = df[df['path'].apply(os.path.exists)]
    
    if len(df) == 0:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: Аудиофайлы не найдены. Проверьте путь: '{AUDIO_FOLDER_PATH}'.")
        return

    if len(df) < initial_rows:
        print(f"⚠️ Предупреждение: Пропущено {initial_rows - len(df)} строк (аудиофайлы не найдены).")

    # ❗ ИСПРАВЛЕНИЕ: Создаем один полный датасет для оценки
    full_dataset = Dataset.from_pandas(df)
    # test_dataset теперь указывает на весь датасет
    test_dataset = full_dataset 

    ## 2. Загрузка модели (Использование локального vocab.json)
    
    print(f"Загрузка предобученной модели {MODEL_ID}...")
    
    if not os.path.exists(VOCAB_FILE_PATH):
        print(f"❌ Ошибка: Локальный словарь не найден по пути: {VOCAB_FILE_PATH}")
        return
        
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        
        # Токенизатор загружается напрямую из локального файла
        tokenizer = Wav2Vec2CTCTokenizer(
            VOCAB_FILE_PATH,
            unk_token="[UNK]", 
            pad_token="[PAD]", 
            word_delimiter_token="|",
        )
        
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, 
            tokenizer=tokenizer
        )
        
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        print("✅ Модель и процессор успешно загружены с использованием локального словаря!")

    except Exception as e:
        print(f"❌ Ошибка загрузки Wav2Vec2. Ошибка: {e}")
        return

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    wer_metric = evaluate.load("wer")
    predictions = []
    references = []

    ## 3. Цикл оценки
    
    print(f"Начало оценки Wav2Vec2-XLS-R-53 на {len(test_dataset)} примерах...")
    
    # ❗ ИСПРАВЛЕНИЕ: Итерируем по всему датасету
    for i, item in enumerate(test_dataset): 
        audio_path = item["path"]
        reference = item["sentence"]
        
        speech, sr = librosa.load(audio_path, sr=16000)

        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits

        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]

        predictions.append(pred_str)
        references.append(reference)

        # Выводим только первые 5 примеров, чтобы не спамить в консоль
        if i < 5:
            print(f"\n--- Пример {i+1} ---")
            print(f"Ref:  {reference}")
            print(f"Pred: {pred_str}")

    ## 4. Расчет финального WER
    
    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"\n=============================")
    print(f"Результат Wav2Vec2-XLS-R-53 WER ({len(predictions)} примеров): {wer:.2%}")
    print(f"=============================")

if __name__ == "__main__":
    main()
import torch
import pandas as pd
import librosa
import os
import re
import evaluate
from transformers import HubertForCTC, Wav2Vec2Processor
from datasets import Dataset

# ================= КОНФИГУРАЦИЯ =================
MODEL_ID = "facebook/hubert-large-ls960-ft"
CSV_FILE_PATH = "C:/Proekt/s_nulya/test/transcripts_output_wav.csv"
AUDIO_FOLDER_PATH = "C:/Proekt/s_nulya/test/welsh_wav_test"

# Имена столбцов из вашего CSV-файла
COL_FILENAME = 'Название файла'
COL_TRANSCRIPT = 'Расшифровка'

def main():
    
    ## 1. Загрузка и подготовка данных
    print("Загрузка датасета...")
    df = pd.read_csv(CSV_FILE_PATH, sep=',')
    
    # Проверка столбцов и переименование
    if COL_FILENAME not in df.columns or COL_TRANSCRIPT not in df.columns:
        print(f"❌ Ошибка: Не найдены ожидаемые столбцы '{COL_FILENAME}' и '{COL_TRANSCRIPT}'.")
        return
        
    df = df[[COL_FILENAME, COL_TRANSCRIPT]].dropna()
    df = df.rename(columns={COL_FILENAME: 'path', COL_TRANSCRIPT: 'sentence'})
    
    # Формируем полный путь и чистим текст
    df['path'] = [os.path.join(AUDIO_FOLDER_PATH, x.replace('.mp3', '.wav')) for x in df['path']]
    
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\\(\)]'
    def clean_text(text):
        # Добавляем пробел в конце для CTC декодирования
        return re.sub(chars_to_ignore_regex, '', str(text)).lower() + " "

    df['sentence'] = df['sentence'].apply(clean_text)
    
    # Проверка наличия аудиофайлов
    initial_rows = len(df)
    df = df[df['path'].apply(os.path.exists)]

    if len(df) == 0:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: Аудиофайлы не найдены. Проверьте путь: '{AUDIO_FOLDER_PATH}'.")
        return

    if len(df) < initial_rows:
        print(f"⚠️ Предупреждение: Пропущено {initial_rows - len(df)} строк (аудиофайлы не найдены).")

    # ❗ ИСПРАВЛЕНИЕ: Создаем один полный датасет для оценки
    full_dataset = Dataset.from_pandas(df)
    test_dataset = full_dataset 
    
    ## 2. Загрузка предобученной модели и процессора
    print(f"Загрузка предобученной модели {MODEL_ID}...")
    try:
        # HuBERT использует Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        model = HubertForCTC.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"❌ Ошибка загрузки HuBERT: {e}")
        return

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    wer_metric = evaluate.load("wer")
    predictions = []
    references = []

    print(f"Начало оценки HuBERT на {len(test_dataset)} примерах...")
    
    ## 3. Цикл оценки (для всех примеров)
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

        if i < 5:
            print(f"\n--- Пример {i+1} ---")
            print(f"Ref:  {reference}")
            print(f"Pred: {pred_str}")

    ## 4. Расчет WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"\n=============================")
    print(f"Результат HuBERT WER ({len(predictions)} примеров): {wer:.2%}")
    print(f"=============================")

if __name__ == "__main__":
    main()
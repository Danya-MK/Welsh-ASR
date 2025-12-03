import torch
import pandas as pd
import librosa
import os
import re
import evaluate
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset

# ================= КОНФИГУРАЦИЯ =================

MODEL_ID = "openai/whisper-small"

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
CSV_FILE_PATH = os.path.join(BASE_DIR, "test", "transcripts_output_wav.csv")
AUDIO_FOLDER_PATH = os.path.join(BASE_DIR, "test", "welsh_wav_test")
# ------------------------------------------------

LANGUAGE = "welsh"
TASK = "transcribe"

# Имена столбцов из вашего CSV-файла
COL_FILENAME = 'Название файла'
COL_TRANSCRIPT = 'Расшифровка'

def main():
    
    ## 1. Загрузка и подготовка данных
    print("Загрузка датасета...")
    try:
        df = pd.read_csv(CSV_FILE_PATH, sep=',')
    except Exception as e:
        print(f"❌ Ошибка при чтении CSV. Проверьте путь или разделитель. Ошибка: {e}")
        return
    
    # Используем фактические имена столбцов и переименовываем
    if COL_FILENAME not in df.columns or COL_TRANSCRIPT not in df.columns:
        print(f"❌ Ошибка: Не найдены ожидаемые столбцы '{COL_FILENAME}' и '{COL_TRANSCRIPT}'.")
        return
        
    df = df[[COL_FILENAME, COL_TRANSCRIPT]].dropna()
    df = df.rename(columns={COL_FILENAME: 'path', COL_TRANSCRIPT: 'sentence'})
    
    # Формируем полный путь и чистим текст
    df['path'] = [os.path.join(AUDIO_FOLDER_PATH, x.replace('.mp3', '.wav')) for x in df['path']]
    
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\\(\)]'
    def clean_text(text):
        return re.sub(chars_to_ignore_regex, '', str(text)).lower()

    df['sentence'] = df['sentence'].apply(clean_text)
    
    # Проверка наличия аудиофайлов
    initial_rows = len(df)
    df = df[df['path'].apply(os.path.exists)]
    
    if len(df) == 0:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: Аудиофайлы не найдены. Проверьте путь: '{AUDIO_FOLDER_PATH}'.")
        return

    if len(df) < initial_rows:
        print(f"⚠️ Предупреждение: Пропущено {initial_rows - len(df)} строк (аудиофайлы не найдены).")

    full_dataset = Dataset.from_pandas(df)
    test_dataset = full_dataset 
    
    
    ## 2. Загрузка предобученной модели и процессора
    print(f"Загрузка предобученной модели {MODEL_ID}...")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"❌ Ошибка загрузки Whisper: {e}")
        return

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    wer_metric = evaluate.load("wer")
    predictions = []
    references = []

    print(f"Начало оценки Whisper Small на {len(test_dataset)} примерах...")
    
    
    ## 3. Цикл оценки (для всех примеров)
    for i, item in enumerate(test_dataset): 
        audio_path = item["path"]
        reference = item["sentence"]

        speech, sr = librosa.load(audio_path, sr=16000)

        # Инференс Whisper
        input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features
        
        with torch.no_grad():
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
            generated_ids = model.generate(input_features.to(device), forced_decoder_ids=forced_decoder_ids)
        
        pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Нормализация предсказания (убираем знаки препинания, приводим к нижнему регистру)
        pred_str = re.sub(chars_to_ignore_regex, '', pred_str).lower()

        predictions.append(pred_str)
        references.append(reference)

        if i < 5:
            print(f"\n--- Пример {i+1} ---")
            print(f"Ref:  {reference}")
            print(f"Pred: {pred_str}")

    ## 4. Расчет WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"\n=============================")
    print(f"Результат Whisper Small WER ({len(predictions)} примеров): {wer:.2%}")
    print(f"=============================")

if __name__ == "__main__":
    main()
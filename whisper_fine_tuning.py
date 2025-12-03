import pandas as pd
import torch
import re
import os
import evaluate
import numpy as np
import librosa
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Dict, List, Union

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
PROJECT_ROOT = os.getcwd() 

TSV_FILE_PATH = os.path.join(PROJECT_ROOT, "train", "train.tsv") # Динамический путь к TSV
AUDIO_FOLDER_PATH = os.path.join(PROJECT_ROOT, "train", "welsh_wav") # Динамический путь к WAV
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "whisper-welsh-small") # Динамический путь для сохранения

# Параметры
BATCH_SIZE = 16
EPOCHS = 30
WHISPER_MODEL_ID = "openai/whisper-small" 
LANGUAGE = "welsh" 
TASK = "transcribe" 

# ==========================================
# 2. DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch["labels"] = labels
        return batch

# ==========================================
# 3. ОСНОВНОЙ КОД
# ==========================================
def main():
    # Отключаем параллелизм токенизатора
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # --- ПОДГОТОВКА ДАННЫХ ---
    print(f"Загрузка данных из {TSV_FILE_PATH}...")
    df = pd.read_csv(TSV_FILE_PATH, sep='\t', quoting=3, on_bad_lines='skip')
    df = df[['path', 'sentence']].dropna()

    # Обработка путей
    df['path'] = df['path'].apply(lambda x: x.replace('.mp3', '.wav'))
    # List comprehension чтобы избежать ошибки лямбды и os
    df['path'] = [os.path.join(AUDIO_FOLDER_PATH, x) for x in df['path']]
    
    print("Проверка валидности файлов...")
    df = df[df['path'].apply(os.path.exists)]
    print(f"Валидных файлов: {len(df)}")

    if len(df) == 0:
        print("ОШИБКА: Файлы не найдены.")
        return

    dataset = Dataset.from_pandas(df)
    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns(["__index_level_0__"])

    dataset = dataset.train_test_split(test_size=0.1)

    # --- ОЧИСТКА ТЕКСТА ---
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\\(\)]'

    def remove_special_characters(batch):
        if batch["sentence"] is None:
            batch["sentence"] = ""
        batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
        return batch

    dataset = dataset.map(remove_special_characters)

    # --- ПРОЦЕССОР ---
    print("Загрузка процессора Whisper...")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID, language=LANGUAGE, task=TASK)
    
    # Переименовываем колонку для удобства
    dataset = dataset.rename_column("path", "audio_path") 

    def prepare_dataset(batch):
        
        audio_paths = batch["audio_path"]
        sentences = batch["sentence"]
        
        batch_input_features = []
        batch_labels = []
        
        for audio_path, sentence in zip(audio_paths, sentences):
            try:
                # 1. Загрузка аудио (librosa)
                audio_data, sr = librosa.load(audio_path, sr=16000)
                
                # 2. Спектрограмма
                input_features = processor.feature_extractor(
                    audio_data, 
                    sampling_rate=sr
                ).input_features[0]
                
                # 3. Токенизация текста
                labels = processor.tokenizer(sentence).input_ids
                
                batch_input_features.append(input_features)
                batch_labels.append(labels)
                
            except Exception as e:
                print(f"Ошибка при чтении файла {audio_path}: {e}")
                # В случае ошибки добавляем пустышку, чтобы не ломать батч
                batch_input_features.append(np.zeros((80, 3000))) 
                batch_labels.append(processor.tokenizer("").input_ids)

        # Возвращаем в формате, который ждет модель
        return {"input_features": batch_input_features, "labels": batch_labels}

    print("Установка трансформации данных (Lazy Loading)...")
    # ВМЕСТО .map() используем .set_transform()
    # Это не запускает обработку прямо сейчас, а только регистрирует функцию.
    dataset.set_transform(prepare_dataset)
    
    # --- МОДЕЛЬ ---
    print("Загрузка модели Whisper...")
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)
    
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    model.config.suppress_tokens = []

    # --- МЕТРИКИ ---
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # --- ОБУЧЕНИЕ ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        
        eval_strategy="steps",
        num_train_epochs=EPOCHS,
        fp16=torch.cuda.is_available(), 
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=1e-5,
        warmup_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        remove_unused_columns=False, 
        dataloader_num_workers=0, 
        predict_with_generate=True, 
        generation_max_length=225,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=processor,
    )

    print("Начало обучения Whisper...")
    # Обучение начнется сразу, аудио будет грузиться батчами
    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✅ Готово! Модель Whisper сохранена в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
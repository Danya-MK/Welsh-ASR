import pandas as pd
import torch
import json
import re
import os
import librosa
import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC, 
    TrainingArguments, 
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List, Union

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
# Определяем корневую директорию для построения относительных путей
PROJECT_ROOT = os.getcwd() 

TSV_FILE_PATH = os.path.join(PROJECT_ROOT, "train", "train.tsv") # Динамический путь к TSV
AUDIO_FOLDER_PATH = os.path.join(PROJECT_ROOT, "train", "welsh_wav") # Динамический путь к WAV
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "wav2vec2-welsh-model") # Динамический путь для сохранения

# Параметры
BATCH_SIZE = 16
EPOCHS = 30

# ==========================================
# ОСНОВНОЙ КОД
# ==========================================

def main():
    # 2. ПОДГОТОВКА ДАННЫХ
    print(f"Загрузка данных из {TSV_FILE_PATH}...")
    df = pd.read_csv(TSV_FILE_PATH, sep='\t', quoting=3, on_bad_lines='skip')

    df = df[['path', 'sentence']]
    df = df.dropna()

    # Меняем расширение .mp3 на .wav
    df['path'] = df['path'].apply(lambda x: x.replace('.mp3', '.wav'))
    
    # Формируем полные пути
    df['path'] = df['path'].apply(lambda x: os.path.join(AUDIO_FOLDER_PATH, x))

    # Проверка существования файлов
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

    # 3. ОЧИСТКА ТЕКСТА
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\\(\)]'

    def remove_special_characters(batch):
        if batch["sentence"] is None:
            batch["sentence"] = ""
        batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
        return batch

    dataset = dataset.map(remove_special_characters)

    # 4. СОЗДАНИЕ СЛОВАРЯ
    def extract_all_chars(batch):
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names["train"])

    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab.json', 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    print("Словарь создан.")

    # 5. ПРОЦЕССОР
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 6. ОБРАБОТКА АУДИО
    def prepare_dataset(batch):
        audio_arrays = []
        valid_sentences = []
        
        for path, sentence in zip(batch["path"], batch["sentence"]):
            try:
                # Загружаем
                speech_array, sr = librosa.load(path, sr=16000)
                audio_arrays.append(speech_array)
                valid_sentences.append(sentence)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Заглушка (тишина 1 сек), чтобы не ломать батч
                audio_arrays.append(np.zeros(16000)) 
                valid_sentences.append("")

        # Процессинг аудио
        inputs = processor(audio_arrays, sampling_rate=16000, padding=True)
        batch["input_values"] = inputs.input_values
        
        # Процессинг текста
        with processor.as_target_processor():
             batch["labels"] = processor(valid_sentences).input_ids
             
        return batch

    print("Установка трансформации данных (Lazy Loading)...")
    dataset.set_transform(prepare_dataset)

    # 7. DATA COLLATOR
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.feature_extractor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )

            with self.processor.as_target_processor():
                labels_batch = self.processor.tokenizer.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # 8. МЕТРИКИ
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 9. МОДЕЛЬ И ОБУЧЕНИЕ
    print("Загрузка модели...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        
        eval_strategy="steps",
        num_train_epochs=EPOCHS,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False, 
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=1e-4,
        warmup_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=0, 
        remove_unused_columns=False, # Запрещаем удалять колонки path/sentence
        load_best_model_at_end=True,  # Это будет загружать модель с лучшими метриками в конце
        metric_for_best_model="wer",  # Оценка метрики для лучшей модели (WER)
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=processor.feature_extractor, 
    )

    print("Начало обучения...")
    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Готово! Модель сохранена в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
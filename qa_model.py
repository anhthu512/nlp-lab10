"""Extractive QA components, training, evaluation, and checkpoint utilities."""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import re
import string
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "by",
        "with",
        "and",
        "or",
        "as",
        "from",
        "that",
        "this",
        "it",
        "its",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
    ]
)


def seed_everything(seed=42):
    """Seed Python, NumPy, PyTorch, and deterministic CUDA operations."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def normalize_answer(text):
    """Apply the SQuAD English normalization rules."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, gold):
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction, gold):
    predicted = normalize_answer(prediction).split()
    reference = normalize_answer(gold).split()
    if not predicted or not reference:
        return float(predicted == reference)
    overlap = sum((Counter(predicted) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction, gold_answers):
    references = list(gold_answers) or [""]
    return {
        "EM": 100 * max(exact_match(prediction, gold) for gold in references),
        "F1": 100 * max(token_f1(prediction, gold) for gold in references),
    }


def lexical_overlap(question, context):
    """Fraction of unique question content words occurring in the context."""
    question_words = set(normalize_answer(question).split()) - STOP_WORDS
    context_words = set(normalize_answer(context).split()) - STOP_WORDS
    return len(question_words & context_words) / max(len(question_words), 1)


@dataclass
class AnswerSpan:
    answer: str = ""
    start: int = -1
    end: int = -1
    score: float = -math.inf
    confidence: float = 0.0
    start_token: int = -1
    end_token: int = -1
    window_index: int = -1


class KeywordBaseline:
    """Choose the sentence with the largest question keyword overlap."""

    def predict(self, question, context):
        question_words = set(normalize_answer(question).split()) - STOP_WORDS
        best = AnswerSpan()
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", context):
            text = match.group().strip()
            if not text:
                continue
            sentence_words = set(normalize_answer(text).split()) - STOP_WORDS
            score = len(question_words & sentence_words) / max(
                len(question_words), 1
            )
            if score > best.score:
                start = (
                    match.start()
                    + len(match.group())
                    - len(match.group().lstrip())
                )
                best = AnswerSpan(text, start, start + len(text), score)
        return best


def align_answer_span(
    offsets, sequence_ids, answer_start, answer_end, cls_index
):
    """Map [character_start, character_end) to inclusive token boundaries.

    A partially visible or missing answer maps to CLS for both targets.
    Offsets belonging to question, special, and padding tokens are excluded.
    """
    if len(offsets) != len(sequence_ids):
        raise ValueError("offsets and sequence_ids must have equal lengths")
    if answer_start < 0 or answer_end <= answer_start:
        return cls_index, cls_index
    context_tokens = [
        index
        for index, (offset, sequence_id) in enumerate(
            zip(offsets, sequence_ids)
        )
        if sequence_id == 1 and offset is not None and offset[1] > offset[0]
    ]
    if not context_tokens:
        return cls_index, cls_index
    if (
        answer_start < offsets[context_tokens[0]][0]
        or answer_end > offsets[context_tokens[-1]][1]
    ):
        return cls_index, cls_index
    start_token = next(
        (
            i
            for i in context_tokens
            if offsets[i][0] <= answer_start < offsets[i][1]
        ),
        None,
    )
    end_token = next(
        (
            i
            for i in reversed(context_tokens)
            if offsets[i][0] < answer_end <= offsets[i][1]
        ),
        None,
    )
    if start_token is None or end_token is None or start_token > end_token:
        return cls_index, cls_index
    return start_token, end_token


def shorten_question(question, tokenizer, max_question_tokens=48):
    encoded = tokenizer(
        question.lstrip(),
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if len(encoded["input_ids"]) <= max_question_tokens:
        return question.lstrip()
    end = encoded["offset_mapping"][max_question_tokens - 1][1]
    return question.lstrip()[:end]


def prepare_features(
    examples, tokenizer, max_length=384, stride=128, max_question_tokens=48
):
    """Build windows with original character offsets and per-window labels."""
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for offset mapping")
    if tokenizer.cls_token_id is None:
        raise ValueError("This implementation requires a CLS token")
    minimum_context = (
        max_length
        - max_question_tokens
        - tokenizer.num_special_tokens_to_add(True)
    )
    if max_question_tokens < 1 or not 0 <= stride < minimum_context:
        raise ValueError(
            "Stride must be smaller than the available context window"
        )
    questions = [
        shorten_question(question, tokenizer, max_question_tokens)
        for question in examples["question"]
    ]
    encoded = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    encoded["example_id"] = []
    encoded["start_positions"] = []
    encoded["end_positions"] = []
    encoded["answer_in_window"] = []
    for window, sample_index in enumerate(
        encoded["overflow_to_sample_mapping"]
    ):
        sequence_ids = encoded.sequence_ids(window)
        offsets = encoded["offset_mapping"][window]
        cls_index = encoded["input_ids"][window].index(tokenizer.cls_token_id)
        labels = (cls_index, cls_index)
        answers = examples.get("answers", [None] * len(questions))[
            sample_index
        ]
        if answers:
            for start, text in zip(answers["answer_start"], answers["text"]):
                candidate = align_answer_span(
                    offsets, sequence_ids, start, start + len(text), cls_index
                )
                if candidate != (cls_index, cls_index):
                    labels = candidate
                    break
        encoded["example_id"].append(str(examples["id"][sample_index]))
        encoded["start_positions"].append(labels[0])
        encoded["end_positions"].append(labels[1])
        encoded["answer_in_window"].append(labels != (cls_index, cls_index))
        encoded["offset_mapping"][window] = [
            offset if sequence_ids[index] == 1 else None
            for index, offset in enumerate(offsets)
        ]
    return dict(encoded)


def make_features(examples, tokenizer, max_length=384, stride=128):
    return examples.map(
        lambda batch: prepare_features(batch, tokenizer, max_length, stride),
        batched=True,
        remove_columns=examples.column_names,
        load_from_cache_file=False,
        desc=f"QA windows: max_length={max_length}, stride={stride}",
    )


class ExtractiveQAModel(nn.Module):
    """A pretrained encoder and a two-output linear span prediction head."""

    def __init__(self, model_name, from_config=False):
        super().__init__()
        if from_config:
            self.backbone = AutoModel.from_config(
                AutoConfig.from_pretrained(model_name)
            )
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
        self.qa_head = nn.Linear(self.backbone.config.hidden_size, 2)
        self.accepted_inputs = set(
            inspect.signature(self.backbone.forward).parameters
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
    ):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if (
            token_type_ids is not None
            and "token_type_ids" in self.accepted_inputs
        ):
            inputs["token_type_ids"] = token_type_ids
        hidden = self.backbone(**inputs).last_hidden_state
        start_logits, end_logits = self.qa_head(hidden).unbind(dim=-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            criterion = nn.CrossEntropyLoss()
            loss_start = criterion(start_logits, start_positions)
            loss_end = criterion(end_logits, end_positions)
            loss = (loss_start + loss_end) / 2
        return {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

    def save_checkpoint(self, directory, tokenizer, metadata=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(
            directory / "backbone", safe_serialization=True
        )
        tokenizer.save_pretrained(directory / "tokenizer")
        torch.save(self.qa_head.state_dict(), directory / "qa_head.pt")
        (directory / "metadata.json").write_text(
            json.dumps(metadata or {}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(cls, directory, device="cpu"):
        directory = Path(directory)
        if not (directory / "qa_head.pt").is_file():
            raise FileNotFoundError(
                f"No trained QA checkpoint at {directory}. "
                "Run Experiment 2 in qa_finetuning.ipynb first."
            )
        model = cls(directory / "backbone")
        model.qa_head.load_state_dict(
            torch.load(
                directory / "qa_head.pt", map_location="cpu", weights_only=True
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(
            directory / "tokenizer", use_fast=True
        )
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        return model.to(device).eval(), tokenizer, metadata


def decode_span(
    start_logits,
    end_logits,
    offsets,
    context,
    sequence_ids=None,
    max_answer_length=64,
    window_index=-1,
):
    """Maximize start + end score with i <= j using only context tokens.

    end is an exclusive character offset. Confidence is the product of
    within-window context-token probabilities, not a calibrated correctness score.
    """
    starts = np.asarray(start_logits, dtype=np.float64)
    ends = np.asarray(end_logits, dtype=np.float64)
    if len(starts) != len(ends) or len(starts) != len(offsets):
        raise ValueError("Logits and offsets must have equal lengths")
    if sequence_ids is not None and len(sequence_ids) != len(offsets):
        raise ValueError("sequence_ids and offsets must have equal lengths")
    if max_answer_length < 1:
        raise ValueError("max_answer_length must be positive")
    valid = np.array(
        [
            offset is not None
            and 0 <= offset[0] < offset[1] <= len(context)
            and (sequence_ids is None or sequence_ids[index] == 1)
            for index, offset in enumerate(offsets)
        ],
        dtype=bool,
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return AnswerSpan(window_index=window_index)
    start_prob = np.zeros_like(starts)
    end_prob = np.zeros_like(ends)
    for logits, probabilities in ((starts, start_prob), (ends, end_prob)):
        values = np.exp(logits[valid] - np.max(logits[valid]))
        probabilities[valid] = values / values.sum()
    best = AnswerSpan(window_index=window_index)
    invalid_prefix = np.concatenate(([0], np.cumsum(~valid)))
    for i in indices:
        candidates = indices[
            (indices >= i) & (indices < i + max_answer_length)
        ]
        candidates = candidates[
            invalid_prefix[candidates + 1] == invalid_prefix[i]
        ]
        if not len(candidates):
            continue
        j = int(candidates[np.argmax(ends[candidates])])
        score = float(starts[i] + ends[j])
        if score > best.score:
            char_start, char_end = int(offsets[i][0]), int(offsets[j][1])
            best = AnswerSpan(
                context[char_start:char_end],
                char_start,
                char_end,
                score,
                float(start_prob[i] * end_prob[j]),
                int(i),
                j,
                window_index,
            )
    return best


class TensorFeatures(Dataset):
    def __init__(self, features, include_labels=True):
        self.features = features
        self.keys = [
            key
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in features.column_names
        ]
        if include_labels:
            self.keys += ["start_positions", "end_positions"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        row = self.features[index]
        return {
            key: torch.tensor(row[key], dtype=torch.long) for key in self.keys
        }


def predict_features(
    model, features, examples, device, batch_size=8, max_answer_length=64
):
    """Aggregate every overflow window back to its original example ID."""
    model.eval()
    loader = DataLoader(
        TensorFeatures(features), batch_size=batch_size, shuffle=False
    )
    contexts = {str(example["id"]): example["context"] for example in examples}
    predictions = {}
    total_loss, count, position = 0.0, 0, 0
    with torch.inference_mode():
        for batch in loader:
            output = model(
                **{key: value.to(device) for key, value in batch.items()}
            )
            size = len(batch["input_ids"])
            total_loss += float(output["loss"].item()) * size
            count += size
            starts = output["start_logits"].cpu().numpy()
            ends = output["end_logits"].cpu().numpy()
            for local_index in range(size):
                row = features[position + local_index]
                example_id = str(row["example_id"])
                candidate = decode_span(
                    starts[local_index],
                    ends[local_index],
                    row["offset_mapping"],
                    contexts[example_id],
                    max_answer_length=max_answer_length,
                    window_index=position + local_index,
                )
                if (
                    example_id not in predictions
                    or candidate.score > predictions[example_id].score
                ):
                    predictions[example_id] = candidate
            position += size
    return predictions, total_loss / max(count, 1)


def boundary_category(prediction, answers):
    """Compare with all annotated spans; choose the closest valid annotation."""
    gold_spans = [
        (start, start + len(text), text)
        for start, text in zip(answers["answer_start"], answers["text"])
    ]
    if not gold_spans:
        return "no_gold", -1, -1
    start, end, _ = max(
        gold_spans,
        key=lambda gold: (
            int(prediction.start == gold[0]) + int(prediction.end == gold[1]),
            token_f1(prediction.answer, gold[2]),
            -abs(prediction.start - gold[0]) - abs(prediction.end - gold[1]),
        ),
    )
    start_correct = prediction.start == start
    end_correct = prediction.end == end
    if start_correct and end_correct:
        category = "correct"
    elif not start_correct and end_correct:
        category = "wrong_start"
    elif start_correct and not end_correct:
        category = "wrong_end"
    else:
        category = "wrong_span"
    return category, start, end


def evaluation_rows(
    examples, predictions, tokenizer=None, model_name="Transformer"
):
    rows = []
    for example in examples:
        example_id = str(example["id"])
        if example_id not in predictions:
            raise ValueError(f"Missing prediction for example {example_id}")
        prediction = predictions[example_id]
        answers = example["answers"]
        scores = score_answer(prediction.answer, answers["text"])
        category, start, end = boundary_category(prediction, answers)
        first_answer = next(iter(answers["text"]), "")
        length = (
            len(tokenizer(first_answer, add_special_tokens=False)["input_ids"])
            if tokenizer is not None
            else len(normalize_answer(first_answer).split())
        )
        rows.append(
            {
                "id": example_id,
                "model": model_name,
                "question": example["question"],
                "gold": json.dumps(answers["text"], ensure_ascii=False),
                "prediction": prediction.answer,
                **scores,
                "gold_start": start,
                "gold_end": end,
                "pred_start": prediction.start,
                "pred_end": prediction.end,
                "confidence": prediction.confidence,
                "error_type": category,
                "lexical_overlap": lexical_overlap(
                    example["question"], example["context"]
                ),
                "answer_tokens": length,
            }
        )
    return rows


def aggregate_metrics(rows):
    if not rows:
        return {"n": 0, "EM": float("nan"), "F1": float("nan")}
    return {
        "n": len(rows),
        "EM": sum(row["EM"] for row in rows) / len(rows),
        "F1": sum(row["F1"] for row in rows) / len(rows),
    }


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def train_experiment(
    model_name,
    tokenizer,
    train_features,
    val_features,
    val_examples,
    device,
    epochs=2,
    batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    seed=42,
    max_answer_length=64,
    checkpoint_dir=None,
    run_config=None,
):
    """Start from identical initialization for each independent experiment."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    seed_everything(seed)
    model = ExtractiveQAModel(model_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorFeatures(train_features),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history, step_history = [], []
    best_f1, best_predictions, best_epoch = -1.0, None, None
    training_seconds, global_step = 0.0, 0
    _synchronize(device)
    run_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        _synchronize(device)
        training_start = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            size = len(batch["input_ids"])
            total_loss += loss.item() * size
            count += size
            global_step += 1
            step_history.append(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "training_loss": loss.item(),
                }
            )
            if batch_index % 50 == 0 or batch_index == len(loader):
                print(
                    f"Epoch {epoch}/{epochs}, batch {batch_index}/{len(loader)}, "
                    f"mean training loss={total_loss / count:.4f}",
                    flush=True,
                )
        _synchronize(device)
        epoch_seconds = time.perf_counter() - training_start
        training_seconds += epoch_seconds
        predictions, validation_loss = predict_features(
            model,
            val_features,
            val_examples,
            device,
            batch_size,
            max_answer_length,
        )
        metrics = aggregate_metrics(evaluation_rows(val_examples, predictions))
        history.append(
            {
                "epoch": epoch,
                "step": global_step,
                "training_loss": total_loss / max(count, 1),
                "validation_loss": validation_loss,
                **metrics,
                "epoch_training_seconds": epoch_seconds,
                "training_seconds": training_seconds,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if metrics["F1"] > best_f1:
            best_f1, best_predictions, best_epoch = (
                metrics["F1"],
                predictions,
                epoch,
            )
            if checkpoint_dir is not None:
                model.save_checkpoint(
                    checkpoint_dir,
                    tokenizer,
                    {
                        **(run_config or {}),
                        "best_epoch": epoch,
                        "validation_EM": metrics["EM"],
                        "validation_F1": metrics["F1"],
                    },
                )
    _synchronize(device)
    wall_seconds = time.perf_counter() - run_start
    del optimizer, model
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return {
        "history": history,
        "step_history": step_history,
        "predictions": best_predictions,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "wall_seconds": wall_seconds,
    }


def save_prediction_log(path, examples, predictions):
    with Path(path).open("w", encoding="utf-8") as stream:
        for example in examples:
            prediction = predictions[str(example["id"])]
            row = {
                "id": str(example["id"]),
                "question": example["question"],
                "gold": example.get("answers", {}),
                **asdict(prediction),
            }
            stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )

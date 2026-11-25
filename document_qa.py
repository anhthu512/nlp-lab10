"""Direct question/context QA using the checkpoint from qa_finetuning.ipynb."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import Dataset
from qa_model import ExtractiveQAModel, make_features, predict_features


class DocumentQA:
    def __init__(self, checkpoint_dir=None, device=None, batch_size=8):
        checkpoint_dir = checkpoint_dir or Path(__file__).parent / "checkpoint"
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.tokenizer, self.metadata = (
            ExtractiveQAModel.load_checkpoint(checkpoint_dir, self.device)
        )
        self.batch_size = batch_size

    def answer(self, question, context):
        """Return a span with [start, end) positions and uncalibrated confidence.

        The SQuAD 1.1 model always attempts a span; it has no calibrated
        answerability threshold for questions without an answer in the context.
        """
        if not question.strip() or not context.strip():
            raise ValueError("question and context must both be nonempty")
        examples = Dataset.from_list(
            [
                {
                    "id": "document",
                    "question": question,
                    "context": context,
                    "answers": {"text": [], "answer_start": []},
                }
            ]
        )
        features = make_features(
            examples,
            self.tokenizer,
            max_length=self.metadata.get("max_length", 384),
            stride=self.metadata.get("stride", 128),
        )
        predictions, _ = predict_features(
            self.model,
            features,
            examples,
            self.device,
            self.batch_size,
            self.metadata.get("max_answer_length", 64),
        )
        result = asdict(predictions["document"])
        result["confidence_kind"] = "uncalibrated_window_probability_product"
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Direct extractive Document QA"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(__file__).parent / "checkpoint"
    )
    parser.add_argument("--context-file", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--no-interactive", action="store_true")
    args = parser.parse_args()
    context = (
        args.context_file.read_text(encoding="utf-8")
        if args.context_file
        else "Extractive question answering selects an answer span from a supplied "
        "document. The Stanford Question Answering Dataset was introduced in "
        "2016. Its examples contain a context paragraph, a question, and an "
        "annotated answer. A span prediction model estimates the start and end "
        "positions of an answer. This demo receives its context directly."
    )
    system = DocumentQA(args.checkpoint, args.device)
    print("Context:", context)
    questions = (
        [args.question]
        if args.question
        else [
            "When was the Stanford Question Answering Dataset introduced?",
            "What does extractive question answering select?",
            "What positions does a span prediction model estimate?",
        ]
    )
    for question in questions:
        print(
            json.dumps(
                {"question": question, **system.answer(question, context)},
                ensure_ascii=False,
                indent=2,
            )
        )
    if args.no_interactive:
        return
    while True:
        try:
            question = input("Question (empty line to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        print(
            json.dumps(
                system.answer(question, context), ensure_ascii=False, indent=2
            )
        )


if __name__ == "__main__":
    main()

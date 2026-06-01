import warnings
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer*", category=DeprecationWarning)

from dataclasses import asdict
import wandb
import os
import torch

from models import build_model_and_tokenizer, parse_args
from data import build_eval_dataset_dict, get_data_collator, get_compute_metrics_dict
from engine import TrainerStreamGenerator, StreamGenerator
from evaluate_metrics import calculate_metrics_for_streaming_narrations


def generate():
    args = parse_args()

    model, tokenizer = build_model_and_tokenizer(is_training=False, **asdict(args))
    data_collator = get_data_collator(stream_generate=True, tokenizer=tokenizer, model_config=model.config, **asdict(args))
    generate_dataset_dict = build_eval_dataset_dict(tokenizer=tokenizer, model_config=model.config, **asdict(args))

    stream_generator = StreamGenerator(model, tokenizer, args)
    trainer = TrainerStreamGenerator(
        stream_generator=stream_generator,
        model=model, tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
    )

    assert len(generate_dataset_dict) == 1
    res = trainer.predict(next(iter(generate_dataset_dict.values())))

    torch.distributed.barrier()

    if trainer.is_world_process_zero():
        pred_path = args.output_dir.replace('outputs/', '')
        # Now we need to evaluate the generated narrations on disk
        metrics = calculate_metrics_for_streaming_narrations(pred_path, data_root=args.data_root)

        wandb.init(
            project=os.getenv("WANDB_PROJECT", "Stream_Generate"),
            name=pred_path,
        )

        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    generate()
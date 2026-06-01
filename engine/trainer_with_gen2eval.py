import torch
from transformers import Trainer
import os

class TrainerWithGenToEval(Trainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] = None,
    ):
        with torch.no_grad(), self.compute_loss_context_manager():
            inputs = self._prepare_inputs(inputs)
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)
            sample_idxs = inputs.pop('sample_idxs')
            evaluation_kwargs = inputs.pop('evaluation_kwargs')
            evaluator = evaluation_kwargs.pop('evaluator')
            output_ids, _ = getattr(model, evaluator)(
                **inputs, **evaluation_kwargs, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_attentions=self.args.output_attentions)
            return (None, output_ids.reshape(1, -1), sample_idxs)


class TrainerWithGenToEvalSaveAttention(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prediction_step(
            self,
            model: torch.nn.Module,
            inputs: dict,
            prediction_loss_only: bool,
            ignore_keys: list[str] = None,
    ):
        with torch.no_grad(), self.compute_loss_context_manager():
            inputs = self._prepare_inputs(inputs)
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)

            sample_idxs = inputs.pop('sample_idxs')
            evaluation_kwargs = inputs.pop('evaluation_kwargs')
            evaluator = evaluation_kwargs.pop('evaluator')

            # Assuming inputs contains 'input_ids' you want to save
            input_ids = inputs.get('input_ids', None)

            output_ids, attentions = getattr(model, evaluator)(
                **inputs, **evaluation_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_attentions=self.args.output_attentions)

            # Save the outputs to disk
            sample_idx = int(sample_idxs[0])
            output_file = os.path.join(self.args.output_dir, f'attentions_{sample_idx:05}.pt')
            data = {
                "ids": input_ids[0].cpu(),
                "attention": attentions.cpu()
            }
            torch.save(data, output_file)

            del data

            return (None, output_ids.reshape(1, -1), sample_idxs)

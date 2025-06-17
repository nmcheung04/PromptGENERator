import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)

class PromptProjector(nn.Module):
    # project prompt to embeddings
    def __init__(self, num_rna_features, embedding_dim, num_prompt_tokens):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_prompt_tokens = num_prompt_tokens
        output_size = num_prompt_tokens * embedding_dim

        self.projector = nn.Linear(num_rna_features, output_size)

        # self.projector = nn.Sequential(
        #     nn.Linear(num_rna_features, (num_rna_features + output_size) // 2),
        #     nn.GELU(),
        #     nn.Linear((num_rna_features + output_size) // 2, output_size)
        # )
    
    def forward(self, rna_prompt_vector):
        projected_prompt = self.projector(rna_prompt_vector)
        prompt_embeddings = projected_prompt.view(-1, self.num_prompt_tokens, self.embedding_dim)
        return prompt_embeddings

class PromptedGenerator(PreTrainedModel):
    # prompt wrapper for generator
    def __init__(self, config, model_name, prompt_tuning_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.prompt_config = prompt_tuning_config
        self.num_prompt_tokens = self.prompt_config["num_prompt_tokens"]

        self.generator = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
        )

        self.prompt_projector = PromptProjector(
            num_rna_features = prompt_tuning_config['num_rna_features'],
            embedding_dim = self.generator.config.hidden_size,
            num_prompt_tokens=prompt_tuning_config['num_prompt_tokens']
        )

        self.freeze()
    
    def freeze(self):
        # freeze backbone of generator
        for param in self.generator.parameters():
            param.requires_grad = False

        # unfreeze prompt param and head
        for param in self.prompt_projector.parameters():
            param.requires_grad = True

        for param in self.generator.score.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 Model size: {total_params / 1e6:.1f}M parameters")
        print(f"   Trainable params: {trainable_params / 1e6:.1f}M parameters")

    def get_input_embeddings(self):
        return self.generator.model.embed_tokens

    def forward(
        self,
        input_ids,
        attention_mask,
        prompt,
        labels,
        **kwargs
    ):
        
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'num_items_in_batch'} # was getting errors when included

        dna_word_embeddings = self.get_input_embeddings()(input_ids)
        prompt_embeddings = self.prompt_projector(prompt)

        bos_embedding = dna_word_embeddings[:, :1, :]
        dna_embeddings = dna_word_embeddings[:, 1:, :]
        
        combined_embeddings = torch.cat(
            [bos_embedding, prompt_embeddings, dna_embeddings], dim=1
        )
        
        batch_size = input_ids.shape[0]
        prompt_mask = torch.ones(
            batch_size, self.num_prompt_tokens, device=attention_mask.device
        )

        combined_attention_mask = torch.cat(
            [attention_mask[:, :1], prompt_mask, attention_mask[:, 1:]], dim=1
        )
        
        outputs = self.generator(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=labels,
            **filtered_kwargs,
        )

        return outputs
        

class DataCollatorForPromptTuning:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        rna_prompts = [torch.tensor(f.pop("prompt")) for f in features]
        collated_features = self.data_collator(features)
        collated_features["prompt"] = torch.stack(rna_prompts)
        
        return collated_features


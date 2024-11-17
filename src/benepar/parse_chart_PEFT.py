import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification  # added for pretrained model

from . import char_lstm
from . import decode_chart
from . import nkutil
from .partitioned_transformer import (
    ConcatPositionalEncoding,
    FeatureDropout,
    PartitionedTransformerEncoder,
    PartitionedTransformerEncoderLayer,
)
from . import parse_base
from . import retokenization
from . import subbatching

# Added for adapter ----------------------------------------------------------------
from peft import get_peft_model, AdaLoraModel, LoHaModel, LoHaConfig, LoKrModel, LoKrConfig

# Import spaCy for modifier and conjunction extraction
import spacy

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')


class ChartParser(nn.Module, parse_base.BaseParser):
    def __init__(
        self,
        tag_vocab,
        label_vocab,
        char_vocab,
        hparams,
        pretrained_model_path=None,
        adapter_name=None,
        adapter_config=None,
        mod_vocab_size=128,
        conj_vocab_size=128,
        mod_embedding_dim=128,
        conj_embedding_dim=128,
        mod_to_idx=None,
        conj_to_idx=None,
    ):
        super().__init__()
        self.config = locals()
        self.config.pop("self")
        self.config.pop("__class__")
        self.config.pop("pretrained_model_path")
        self.config["hparams"] = hparams.to_dict()

        self.tag_vocab = tag_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab

        self.d_model = hparams.d_model

        self.char_encoder = None
        self.pretrained_model = None
        self.adapter_name = adapter_name
        self.adapter_config = adapter_config

        # Modifier embedding and projection
        self.mod_embedding = nn.Embedding(mod_vocab_size, mod_embedding_dim)
        self.W_mod = nn.Linear(mod_embedding_dim, self.d_model)

        # Conjunction embedding and projection
        self.conj_embedding = nn.Embedding(conj_vocab_size, conj_embedding_dim)
        self.W_conj = nn.Linear(conj_embedding_dim, self.d_model)

        # Store modifier and conjunction mappings
        self.mod_to_idx = mod_to_idx
        self.conj_to_idx = conj_to_idx

        if hparams.use_chars_lstm:
            assert (
                not hparams.use_pretrained
            ), "use_chars_lstm and use_pretrained are mutually exclusive"
            self.retokenizer = char_lstm.RetokenizerForCharLSTM(self.char_vocab)
            self.char_encoder = char_lstm.CharacterLSTM(
                max(self.char_vocab.values()) + 1,
                hparams.d_char_emb,
                hparams.d_model // 2,  # Half-size to leave room for
                # partitioned positional encoding
                char_dropout=hparams.char_lstm_input_dropout,
            )
        elif hparams.use_pretrained:
            if pretrained_model_path is None:  # when using pretrained model go in here
                self.retokenizer = retokenization.Retokenizer(
                    hparams.pretrained_model, retain_start_stop=True
                )

                self.pretrained_model = AutoModel.from_pretrained(
                    hparams.pretrained_model
                )
                print("hparams.pretrained_model: ", hparams.pretrained_model)

                if self.adapter_name in ["LoRA", "LoHa", "LoKr", "IA3", "VeRa", "BOFT", "PrefixTuning", "P_Tuning", "PromptTuning"]:
                    self.pretrained_model = get_peft_model(self.pretrained_model, adapter_config)
                    print(f"Training model with using adapter: {self.adapter_name}")
                elif self.adapter_name == "AdaLoRA":
                    self.pretrained_model = AdaLoraModel(self.pretrained_model, adapter_config, "default")
                    print(f"Training model with using adapter: {self.adapter_name}")

                if hasattr(self.pretrained_model, 'print_trainable_parameters'):
                    self.pretrained_model.print_trainable_parameters()

            else:
                self.retokenizer = retokenization.Retokenizer(
                    pretrained_model_path, retain_start_stop=True
                )
                self.pretrained_model = AutoModel.from_config(
                    AutoConfig.from_pretrained(pretrained_model_path)
                )
            d_pretrained = self.pretrained_model.config.hidden_size

            if hparams.use_encoder:
                # original:
                self.project_pretrained = nn.Linear(
                    d_pretrained, hparams.d_model // 2, bias=False
                )

            else:
                # original:
                self.project_pretrained = nn.Linear(
                    d_pretrained, hparams.d_model, bias=False
                )

        if hparams.use_encoder:
            self.morpho_emb_dropout = FeatureDropout(hparams.morpho_emb_dropout)
            self.add_timing = ConcatPositionalEncoding(
                d_model=hparams.d_model,
                max_len=hparams.encoder_max_len,
            )
            encoder_layer = PartitionedTransformerEncoderLayer(
                hparams.d_model,
                n_head=hparams.num_heads,
                d_qkv=hparams.d_kv,
                d_ff=hparams.d_ff,
                ff_dropout=hparams.relu_dropout,
                residual_dropout=hparams.residual_dropout,
                attention_dropout=hparams.attention_dropout,
            )
            self.encoder = PartitionedTransformerEncoder(
                encoder_layer, hparams.num_layers
            )
        else:
            self.morpho_emb_dropout = None
            self.add_timing = None
            self.encoder = None

        self.f_label = nn.Sequential(
            nn.Linear(hparams.d_model, hparams.d_label_hidden),
            nn.LayerNorm(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, max(label_vocab.values())),
        )

        if hparams.predict_tags:
            # Original:
            self.f_tag = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_tag_hidden),
                nn.LayerNorm(hparams.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_tag_hidden, max(tag_vocab.values()) + 1),
            )

            self.tag_loss_scale = hparams.tag_loss_scale
            self.tag_from_index = {i: label for label, i in tag_vocab.items()}
        else:
            self.f_tag = None
            self.tag_from_index = None

        self.decoder = decode_chart.ChartDecoder(
            label_vocab=self.label_vocab,
            force_root_constituent=hparams.force_root_constituent,
        )
        self.criterion = decode_chart.SpanClassificationMarginLoss(
            reduction="sum", force_root_constituent=hparams.force_root_constituent
        )

        self.parallelized_devices = None

    @property
    def device(self):
        if self.parallelized_devices is not None:
            return self.parallelized_devices[0]
        else:
            return next(self.f_label.parameters()).device

    @property
    def output_device(self):
        if self.parallelized_devices is not None:
            return self.parallelized_devices[1]
        else:
            return next(self.f_label.parameters()).device

    def parallelize(self, *args, **kwargs):
        self.parallelized_devices = (torch.device("cuda", 0), torch.device("cuda", 1))
        for child in self.children():
            if child != self.pretrained_model:
                child.to(self.output_device)
        self.pretrained_model.parallelize(*args, **kwargs)

    @classmethod
    def from_trained(cls, model_path):
        if os.path.isdir(model_path):
            # Multi-file format used when exporting models for release.
            # Unlike the checkpoints saved during training, these files include
            # all tokenizer parameters and a copy of the pre-trained model
            # config (rather than downloading these on-demand).
            config = AutoConfig.from_pretrained(model_path).benepar
            state_dict = torch.load(
                os.path.join(model_path, "benepar_model.bin"), map_location="cpu"
            )
            config["pretrained_model_path"] = model_path
        else:
            # Single-file format used for saving checkpoints during training.
            data = torch.load(model_path, map_location="cpu")
            config = data["config"]
            state_dict = data["state_dict"]

        hparams = config["hparams"]

        if "force_root_constituent" not in hparams:
            hparams["force_root_constituent"] = True

        config["hparams"] = nkutil.HParams(**hparams)
        parser = cls(**config)
        parser.load_state_dict(state_dict)
        return parser

    def encode(self, example):
        if self.char_encoder is not None:
            encoded = self.retokenizer(example.words, return_tensors="np")
        else:
            encoded = self.retokenizer(example.words, example.space_after)

        # Extract the sentence from the example
        sentence = ' '.join(example.words)
        doc = nlp(sentence)

        # Extract modifier indices
        mod_indices = []
        for token in doc:
            if token.pos_ in ['ADJ', 'ADV']:
                mod_word = token.text.lower()
                mod_idx = self.mod_to_idx.get(mod_word, self.mod_to_idx['<no_mod>'])
            else:
                mod_idx = self.mod_to_idx['<no_mod>']
            mod_indices.append(mod_idx)
        # Store in the encoded example
        encoded["mod_indices"] = mod_indices

        # Extract conjunction indices
        seq_len = len(doc)
        conj_indices = [[self.conj_to_idx['<no_conj>']] * seq_len for _ in range(seq_len)]
        for token in doc:
            if token.pos_ == 'CCONJ':
                left = token.i - 1 if token.i > 0 else token.i
                right = token.i + 1 if token.i + 1 < seq_len else token.i
                conj_word = token.text.lower()
                conj_idx = self.conj_to_idx.get(conj_word, self.conj_to_idx['<no_conj>'])
                conj_indices[left][right] = conj_idx
                conj_indices[right][left] = conj_idx  # If symmetric
        # Store in the encoded example
        encoded["conj_indices"] = conj_indices

        if example.tree is not None:
            encoded["span_labels"] = torch.tensor(
                self.decoder.chart_from_tree(example.tree)
            )
            if self.f_tag is not None:
                encoded["tag_labels"] = torch.tensor(
                    [-100] + [self.tag_vocab[tag] for _, tag in example.pos()] + [-100]
                )
        return encoded

    def pad_encoded(self, encoded_batch):
        # Exclude 'mod_indices' and 'conj_indices' when calling self.retokenizer.pad
        batch = self.retokenizer.pad(
            [
                {
                    k: v
                    for k, v in example.items()
                    if k not in ["span_labels", "tag_labels", "mod_indices", "conj_indices"]
                }
                for example in encoded_batch
            ],
            return_tensors="pt",
        )

        if encoded_batch and "span_labels" in encoded_batch[0]:
            batch["span_labels"] = decode_chart.pad_charts(
                [example["span_labels"] for example in encoded_batch]
            )
        if encoded_batch and "tag_labels" in encoded_batch[0]:
            batch["tag_labels"] = nn.utils.rnn.pad_sequence(
                [example["tag_labels"] for example in encoded_batch],
                batch_first=True,
                padding_value=-100,
            )

        # Pad 'mod_indices'
        max_len = batch['input_ids'].size(1)
        batch['mod_indices'] = torch.tensor([
            ex['mod_indices'] + [self.mod_to_idx['<no_mod>']] * (max_len - len(ex['mod_indices']))
            for ex in encoded_batch
        ], dtype=torch.long)

        # Pad 'conj_indices'
        batch_conj_indices = []
        for ex in encoded_batch:
            seq_len = len(ex['conj_indices'])
            # Pad each row
            padded_rows = [
                row + [self.conj_to_idx['<no_conj>']] * (max_len - seq_len)
                for row in ex['conj_indices']
            ]
            # Pad the columns (add rows if necessary)
            padded_rows += [[self.conj_to_idx['<no_conj>']] * max_len] * (max_len - seq_len)
            batch_conj_indices.append(padded_rows)
        batch['conj_indices'] = torch.tensor(batch_conj_indices, dtype=torch.long)

        return batch


    def _get_lens(self, encoded_batch):
        if self.pretrained_model is not None:
            return [len(encoded["input_ids"]) for encoded in encoded_batch]
        return [len(encoded["valid_token_mask"]) for encoded in encoded_batch]

    def encode_and_collate_subbatches(self, examples, subbatch_max_tokens):
        batch_size = len(examples)
        batch_num_tokens = sum(len(x.words) for x in examples)
        encoded = [self.encode(example) for example in examples]

        res = []
        for ids, subbatch_encoded in subbatching.split(
            encoded, costs=self._get_lens(encoded), max_cost=subbatch_max_tokens
        ):
            subbatch = self.pad_encoded(subbatch_encoded)
            subbatch["batch_size"] = batch_size
            subbatch["batch_num_tokens"] = batch_num_tokens
            res.append((len(ids), subbatch))
        return res

    def forward(self, batch):
        valid_token_mask = batch["valid_token_mask"].to(self.output_device)

        if (
            self.encoder is not None
            and valid_token_mask.shape[1] > self.add_timing.timing_table.shape[0]
        ):
            raise ValueError(
                "Sentence of length {} exceeds the maximum supported length of "
                "{}".format(
                    valid_token_mask.shape[1] - 2,
                    self.add_timing.timing_table.shape[0] - 2,
                )
            )

        # Get modifier and conjunction indices from batch
        mod_indices = batch["mod_indices"].to(self.device)  # Shape: (batch_size, seq_len)
        conj_indices = batch["conj_indices"].to(self.device)  # Shape: (batch_size, seq_len, seq_len)

        # Debugging statements
        print("mod_indices shape:", batch['mod_indices'].shape)
        print("input_ids shape:", batch['input_ids'].shape)
        print("valid_token_mask shape:", batch['valid_token_mask'].shape)

        if self.char_encoder is not None:
            assert isinstance(self.char_encoder, char_lstm.CharacterLSTM)
            char_ids = batch["char_ids"].to(self.device)
            extra_content_annotations = self.char_encoder(char_ids, valid_token_mask)
        elif self.pretrained_model is not None:
            input_ids = batch["input_ids"].to(self.device)
            words_from_tokens = batch["words_from_tokens"].to(self.output_device)
            pretrained_attention_mask = batch["attention_mask"].to(self.device)

            extra_kwargs = {}
            if "token_type_ids" in batch:
                extra_kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)
            if "decoder_input_ids" in batch:
                extra_kwargs["decoder_input_ids"] = batch["decoder_input_ids"].to(
                    self.device
                )
                extra_kwargs["decoder_attention_mask"] = batch[
                    "decoder_attention_mask"
                ].to(self.device)

            pretrained_out = self.pretrained_model(
                input_ids, attention_mask=pretrained_attention_mask, **extra_kwargs
            )

            features = pretrained_out.last_hidden_state.to(self.output_device)

            features = features[
                torch.arange(features.shape[0])[:, None],
                F.relu(words_from_tokens),
            ]

            features.masked_fill_(~valid_token_mask[:, :, None], 0)
            if self.encoder is not None:
                extra_content_annotations = self.project_pretrained(features)

        if self.encoder is not None:
            encoder_in = self.add_timing(
                self.morpho_emb_dropout(extra_content_annotations)
            )

            annotations = self.encoder(encoder_in, valid_token_mask)
            # Rearrange the annotations to ensure that the transition to
            # fenceposts captures an even split between position and content.
            # TODO(nikita): try alternatives, such as omitting position entirely
            annotations = torch.cat(
                [
                    annotations[..., 0::2],
                    annotations[..., 1::2],
                ],
                -1,
            )
        else:
            assert self.pretrained_model is not None
            annotations = self.project_pretrained(features)

        if self.f_tag is not None:
            tag_scores = self.f_tag(annotations)
        else:
            tag_scores = None

        fencepost_annotations = torch.cat(
            [
                annotations[:, :-1, : self.d_model // 2],
                annotations[:, 1:, self.d_model // 2 :],
            ],
            -1,
        )

        # Compute h_j - h_i (delta_h)
        delta_h = (
            torch.unsqueeze(fencepost_annotations, 1)
            - torch.unsqueeze(fencepost_annotations, 2)
        )[:, :-1, 1:]  # Shape: (batch_size, seq_len, seq_len, hidden_size)

        # Get modifier embeddings and project
        mod_embeddings = self.mod_embedding(mod_indices)  # Shape: (batch_size, seq_len, mod_embedding_dim)
        mod_embeddings_proj = self.W_mod(mod_embeddings)  # Shape: (batch_size, seq_len, hidden_size)
        mod_embeddings_proj_expanded = mod_embeddings_proj.unsqueeze(2)  # Shape: (batch_size, seq_len, 1, hidden_size)

        # Get conjunction embeddings and project
        conj_embeddings = self.conj_embedding(conj_indices)  # Shape: (batch_size, seq_len, seq_len, conj_embedding_dim)
        E_conj_ij_transformed = self.W_conj(conj_embeddings)  # Shape: (batch_size, seq_len, seq_len, hidden_size)

        # Compute element-wise multiplication
        delta_h_weighted = E_conj_ij_transformed * delta_h  # Element-wise multiplication

        # Add modifier embeddings
        span_features = delta_h_weighted + mod_embeddings_proj_expanded  # Shape: (batch_size, seq_len, seq_len, hidden_size)

        # Compute span scores
        span_scores = self.f_label(span_features)
        span_scores = torch.cat(
            [span_scores.new_zeros(span_scores.shape[:-1] + (1,)), span_scores], -1
        )

        return span_scores, tag_scores

    def compute_loss(self, batch):
        span_scores, tag_scores = self.forward(batch)
        span_labels = batch["span_labels"].to(span_scores.device)
        span_loss = self.criterion(span_scores, span_labels)
        # Divide by the total batch size, not by the subbatch size
        span_loss = span_loss / batch["batch_size"]
        if tag_scores is None:
            return span_loss
        else:
            tag_labels = batch["tag_labels"].to(tag_scores.device)
            tag_loss = self.tag_loss_scale * F.cross_entropy(
                tag_scores.reshape((-1, tag_scores.shape[-1])),
                tag_labels.reshape((-1,)),
                reduction="sum",
                ignore_index=-100,
            )
            tag_loss = tag_loss / batch["batch_num_tokens"]
            return span_loss + tag_loss

    def _parse_encoded(
        self, examples, encoded, return_compressed=False, return_scores=False
    ):
        with torch.no_grad():
            batch = self.pad_encoded(encoded)
            span_scores, tag_scores = self.forward(batch)
            if return_scores:
                span_scores_np = span_scores.cpu().numpy()
            else:
                # Start/stop tokens don't count, so subtract 2
                lengths = batch["valid_token_mask"].sum(-1) - 2
                charts_np = self.decoder.charts_from_pytorch_scores_batched(
                    span_scores, lengths.to(span_scores.device)
                )
            if tag_scores is not None:
                tag_ids_np = tag_scores.argmax(-1).cpu().numpy()
            else:
                tag_ids_np = None

        for i in range(len(encoded)):
            example_len = len(examples[i].words)
            if return_scores:
                yield span_scores_np[i, :example_len, :example_len]
            elif return_compressed:
                output = self.decoder.compressed_output_from_chart(charts_np[i])
                if tag_ids_np is not None:
                    output = output.with_tags(tag_ids_np[i, 1 : example_len + 1])
                yield output
            else:
                if tag_scores is None:
                    leaves = examples[i].pos()
                else:
                    predicted_tags = [
                        self.tag_from_index[i]
                        for i in tag_ids_np[i, 1 : example_len + 1]
                    ]
                    leaves = [
                        (word, predicted_tag)
                        for predicted_tag, (word, gold_tag) in zip(
                            predicted_tags, examples[i].pos()
                        )
                    ]
                yield self.decoder.tree_from_chart(charts_np[i], leaves=leaves)

    def parse(
        self,
        examples,
        return_compressed=False,
        return_scores=False,
        subbatch_max_tokens=None,
    ):
        training = self.training
        self.eval()
        encoded = [self.encode(example) for example in examples]
        if subbatch_max_tokens is not None:
            res = subbatching.map(
                self._parse_encoded,
                examples,
                encoded,
                costs=self._get_lens(encoded),
                max_cost=subbatch_max_tokens,
                return_compressed=return_compressed,
                return_scores=return_scores,
            )
        else:
            res = self._parse_encoded(
                examples,
                encoded,
                return_compressed=return_compressed,
                return_scores=return_scores,
            )
            res = list(res)
        self.train(training)
        return res

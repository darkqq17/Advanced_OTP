# Added for LoRA ----------------------------------------------------------------
from peft import LoraConfig, TaskType, AdaLoraConfig, LoHaConfig, LoKrConfig, IA3Config, VeraConfig, BOFTConfig, PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig

adapter_configurations = {
    "LoRA": LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.1),
    "AdaLoRA": AdaLoraConfig(peft_type="ADALORA", task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1),
    "LoHa": LoHaConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, alpha=32, target_modules=["k", "q", "v", "proj_out", "to_out.0", "proj_in", "ff.net.0.proj", "ff.net.2", "fc1", "fc2"], rank_dropout=0.1, module_dropout=0.1, init_weights=True),
    "LoKr": LoKrConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, alpha=32, target_modules=["k", "q", "v", "proj_out", "fc1", "fc2", "ff.net.0.proj", "ff.net.2", "to_out.0"], rank_dropout=0.1, module_dropout=0.1, init_weights=True,),
    "IA3": IA3Config(task_type=TaskType.SEQ_2_SEQ_LM, peft_type="IA3", target_modules=["k", "v", "w0"], feedforward_modules=["w0"]),
    "VeRa": VeraConfig(task_type=TaskType.SEQ_2_SEQ_LM, peft_type="VERA", r=256),
    "BOFT": BOFTConfig(task_type=TaskType.SEQ_2_SEQ_LM, peft_type="BOFT", boft_block_size=8, boft_n_butterfly_factor=1, target_modules=["k", "q", "v", "output.dense", "mlp.fc1", "mlp.fc2"], boft_dropout=0.1, bias="boft_only", modules_to_save=["classifier"]),
    "P_Tuning": PromptEncoderConfig(task_type=TaskType.SEQ_2_SEQ_LM, peft_type="P_TUNING", num_virtual_tokens=20, token_dim=768, num_transformer_submodules=1, num_attention_heads=12, num_layers=12, encoder_reparameterization_type="MLP", encoder_hidden_size=768),
    "PromptTuning": PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, peft_type="PROMPT_TUNING", num_virtual_tokens=20, token_dim=768, num_transformer_submodules=1, num_attention_heads=12, num_layers=12, prompt_tuning_init="TEXT", prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral", tokenizer_name_or_path="t5-base"),
    # "PrefixTuning": PrefixTuningConfig(task_type=TaskType.SEQ_CLS, peft_type="PREFIX_TUNING", num_virtual_tokens=20, token_dim=768, num_transformer_submodules=1, num_attention_heads=12, num_layers=12, encoder_hidden_size=768),
}
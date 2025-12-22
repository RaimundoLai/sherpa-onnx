import onnx
import onnxruntime as ort
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

onnx_model_path = "g2p_t5_model.onnx"
model_name = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval() 
text_input = "charsiu is delicious"
encoded_input = tokenizer(text_input, return_tensors="pt") # "pt" for PyTorch tensors

dummy_input_ids = encoded_input['input_ids']
dummy_attention_mask = encoded_input['attention_mask']
decoder_start_token_id = model.config.decoder_start_token_id
if decoder_start_token_id is None:
    decoder_start_token_id = tokenizer.pad_token_id
    if decoder_start_token_id is None:
        print("警告: 無法找到 decoder_start_token_id 或 pad_token_id。請手動確認。")
        decoder_start_token_id = 0 # 假設為 0

dummy_decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)


try:
    input_names = ["input_ids", "attention_mask", "decoder_input_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'decoder_input_ids': {0: 'batch_size', 1: 'decoder_sequence_length'},
        'logits': {0: 'batch_size', 1: 'decoder_sequence_length', 2: 'vocab_size'}
    }

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_decoder_input_ids),
        onnx_model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"PyTorch 模型已成功轉換為 ONNX 並保存至 {onnx_model_path}")

except Exception as e:
    print(f"ONNX 轉換失敗: {e}")
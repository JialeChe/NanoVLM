import sys, os
sys.path.insert(0, '.')
import torch

print('=== 1. Import Test ===')
from configs.model_config import NanoVLMConfig
from configs.training_config import TrainingConfig
from src.nanovlm.utils.utils import set_seed
from src.nanovlm.model.nanovlm import NanoVLM
from src.nanovlm.data.dataset import LLaVADataset
print('All imports OK')

print('\n=== 2. Data Config ===')
data_path = './data/llava_pretrain/blip_laion_cc_sbu_558k.json'
image_base_dir = './data/llava_pretrain/images'
print(f'Data: {data_path}')
print(f'Images: {image_base_dir}')

print('\n=== 3. Model Loading ===')
torch.manual_seed(42)
device = torch.device('cuda')
model_config = NanoVLMConfig()
model = NanoVLM(model_config)
model.to(device)
print(f'Model on {device}')

print('\n=== 4. Stage1 Setup ===')
model.set_stage('stage1')
model.get_trainable_parameters()
print('Stage1 OK')

print('\n=== 5. Dataset ===')
dataset = LLaVADataset(
    data_path=data_path,
    tokenizer=model.language_model.tokenizer,
    image_processor=model.vision_encoder.processor,
    image_token_id=model.image_token_id,
    num_image_tokens=model.vision_encoder.get_num_patches(),
    max_seq_length=2048,
    image_base_dir=image_base_dir,
)
print(f'Dataset size: {len(dataset)}')

print('\n=== 6. Forward + Backward ===')
sample = dataset[0]
input_ids = sample['input_ids'].unsqueeze(0).to(device)
attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
labels = sample['labels'].unsqueeze(0).to(device)
pixel_values = sample['pixel_values'].unsqueeze(0).to(device)

model.train()
with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
    outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs['loss']
print(f'Forward OK, loss: {loss.item():.4f}')
loss.backward()
print('Backward OK')

print('\n=== 7. Gradient Check ===')
mlp_ok = any(p.grad is not None for p in model.connector.parameters())
vis_ok = not any(p.grad is not None for p in model.vision_encoder.parameters())
lm_ok = not any(p.grad is not None for p in model.language_model.model.parameters())
print(f'  MLP has grad: {mlp_ok} (should be True)')
print(f'  Vision has grad: {not vis_ok} (should be False)')
print(f'  LM has grad: {not lm_ok} (should be False)')
print(f'  All checks pass: {mlp_ok and vis_ok and lm_ok}')
print(f'  GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
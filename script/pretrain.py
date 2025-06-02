import os
import argparse
import time
import math
import warnings
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer

from .model import MiniMindForCausalLM, MiniMindConfig
from dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    print(content)
    
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    
    for step, (X, Y, loss_mask) in pbar:
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        with torch.amp.autocast():
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps  # 梯度累积
            
        scaler.scale(loss).backward()
        
        if (step+1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 恢复被混合精度放大的梯度至原始尺度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，防止梯度爆炸
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iter_per_epoch}) Loss:{loss.item() * args.accumulation_steps:.3f} \
                epoch_time:{spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60}'
            )
        
        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f"{args.save_dir}/pretrain_{lm_config.hidden_size}.pth"
            
            state_dict = model.state_dict()
            
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()
        
def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/minimind_tokenizer')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    print(f"LLM 总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learing_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()
    
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    args.save_dir = os.path.join(args.out_dir)
    os.mkdir(args.save_dir, exist_ok=Ture)
    os.mkdir(args.out_dir, exist_ok=True)
    model, tokenizer = init_model(lm_config)
    
    train_dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    print(f"模型位于设备：{model.device}, 词表长度：{tokenizer.vocab_size}, DataLoader：{train_loader}")
    
    scaler = torch.amp.GradScaler(device=args.device, enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch=epoch)
import os
import argparse
import time
import math
import warnings
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer

from .model import MiniMindForCausalLM, MiniMindConfig
from dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    print(content)
    
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    
    for step, (X, Y, loss_mask) in pbar:
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        with torch.amp.autocast():
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps  # 梯度累积
            
        scaler.scale(loss).backward()
        
        if (step+1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 恢复被混合精度放大的梯度至原始尺度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，防止梯度爆炸
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/minimind_tokenizer')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    print(f"LLM 总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learing_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()
    
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    args.save_dir = os.path.join(args.out_dir)
    os.mkdir(args.save_dir, exist_ok=Ture)
    os.mkdir(args.out_dir, exist_ok=True)
    model, tokenizer = init_model(lm_config)
    
    train_dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    print(f"模型位于设备：{model.device}, 词表长度：{tokenizer.vocab_size}, DataLoader：{train_loader}")
    
    scaler = torch.amp.GradScaler(device=args.device, enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch=epoch)
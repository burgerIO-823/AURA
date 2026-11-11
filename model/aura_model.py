#!/usr/bin/env python3
"""AURA model: frozen X-Codec RVQ codes -> L/R/S audio tokens -> MLP projector -> Qwen3-8B + LoRA.

Per step (the codec is run offline, see data_preparation/):

    codes_L, codes_R (8, T)  --frozen codebook lookup-->  per-layer 1024-d codewords
      content band  = sum of RVQ layers [0, content_cut)        (default L0-L2)
      sound band    = sum of RVQ layers [content_cut, 8)        (default L3-L7)
      side channel  S = (L - R) / 2, computed per band before pooling
      time pooling  x4 (content bands: max, sound bands: mean), padding-aware
      per frame     [L-content, L-sound, R-content, R-sound, S-content, S-sound]
    -> 2-layer GELU MLP projector (1024 -> 4096 -> 4096)
    -> prepended to the prompt embeddings of Qwen3-8B (frozen backbone + LoRA)

Trainable: projector + LoRA.  Frozen: RVQ codebooks, Qwen3 backbone.

A checkpoint directory holds four files:
    aura_model.safetensors   LoRA weights
    adapter_config.json      LoRA configuration (PEFT)
    proj.pt                  projector state dict
    aura_config.json         architecture (see DEFAULT_CONFIG)
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

# peft + an old torchao raises inside is_torchao_available() instead of returning False.
# AURA has no torchao-quantized layers, so the probe is disabled before importing LoRA.
try:
    import peft.import_utils as _peft_import_utils
    import peft.tuners.lora.torchao as _peft_lora_torchao
    _peft_import_utils.is_torchao_available = lambda: False
    _peft_lora_torchao.is_torchao_available = lambda: False
except Exception:
    pass
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

CONFIG_NAME = "aura_config.json"
PROJECTOR_NAME = "proj.pt"
ADAPTER_NAME = "aura_model.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
CHECKPOINT_FILES = (CONFIG_NAME, PROJECTOR_NAME, ADAPTER_NAME, ADAPTER_CONFIG_NAME)
CODEBOOK_DIM = 1024
NUM_RVQ_LAYERS = 8
SLOTS = ("left_content", "left_sound", "right_content", "right_sound",
         "side_content", "side_sound")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

DEFAULT_CONFIG = {
    "model_type": "aura",
    "base_model": "Qwen/Qwen3-8B",
    "codec": "X-Codec xcodec_hubert_general_audio_v2 (frozen), 8 RVQ layers, 16 kHz, 50 Hz",
    "slots": list(SLOTS),
    "side_definition": "S=(L-R)/2 after codebook lookup, before pooling",
    "content_cut": 3,
    "time_pool": 4,
    "pool_content": "max",
    "pool_sound": "mean",
    "audio_frames": 500,
    "projector": {"type": "mlp_gelu", "in_dim": CODEBOOK_DIM},
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": list(LORA_TARGETS)},
}
# Fields that define the computation; a warm start must agree on all of them.
ARCHITECTURE_KEYS = ("slots", "content_cut", "time_pool", "pool_content", "pool_sound",
                     "audio_frames", "projector", "lora")


def load_config(ckpt_dir):
    with open(os.path.join(ckpt_dir, CONFIG_NAME), encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("model_type") != "aura":
        raise ValueError(f"{ckpt_dir} is not an AURA checkpoint")
    return config


def load_codebooks(path):
    """RVQ codebooks (8, 1024, 1024), written by data_preparation/extract_codebooks.py."""
    codebooks = torch.load(path, map_location="cpu")
    if tuple(codebooks.shape) != (NUM_RVQ_LAYERS, CODEBOOK_DIM, CODEBOOK_DIM):
        raise ValueError(f"unexpected codebook shape {tuple(codebooks.shape)} in {path}")
    return codebooks


def pool_time(values, mask, factor, mode):
    """Padding-aware temporal pooling.

    values (B, T, D), mask (B, T) in {0, 1} -> (B, ceil(T/factor), D) and the pooled mask.
    "mean" ignores padded frames; "max" excludes them; an all-padding window becomes 0.
    """
    if factor <= 1:
        return values, mask
    batch, frames, dim = values.shape
    windows = (frames + factor - 1) // factor
    pad = windows * factor - frames
    if pad:
        values = F.pad(values, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad))
    grouped = values.view(batch, windows, factor, dim)
    window_mask = mask.view(batch, windows, factor)
    valid = window_mask.sum(2)
    if mode == "mean":
        weight = window_mask.unsqueeze(-1).to(grouped.dtype)
        pooled = (grouped * weight).sum(2) / weight.sum(2).clamp(min=1.0)
    elif mode == "max":
        lowest = torch.finfo(grouped.dtype).min
        pooled = grouped.masked_fill(window_mask.unsqueeze(-1) == 0, lowest).max(2).values
        pooled = pooled.masked_fill((valid == 0).unsqueeze(-1), 0.0)
    else:
        raise ValueError(f"unknown pooling mode: {mode}")
    return pooled, (valid > 0).to(mask.dtype)


class AuraQwen(nn.Module):
    roles_per_frame = len(SLOTS)

    def __init__(self, qwen_path, codebooks_path, config=None, dtype=torch.bfloat16,
                 attn_implementation=None):
        super().__init__()
        self.config = json.loads(json.dumps(config or DEFAULT_CONFIG))
        cfg = self.config
        self.dtype = dtype
        self.time_pool = int(cfg["time_pool"])
        self.pool_content = cfg["pool_content"]
        self.pool_sound = cfg["pool_sound"]
        cut = int(cfg["content_cut"])
        if not 1 <= cut < NUM_RVQ_LAYERS:
            raise ValueError(f"content_cut must be in [1, 7], got {cut}")
        self.content_layers = tuple(range(cut))
        self.sound_layers = tuple(range(cut, NUM_RVQ_LAYERS))

        # Frozen Qwen3 backbone with LoRA adapters.  attn_implementation="eager" is only
        # needed when attention weights are inspected.
        llm = AutoModelForCausalLM.from_pretrained(
            qwen_path, torch_dtype=dtype, attn_implementation=attn_implementation)
        llm.config.use_cache = False
        lora = cfg["lora"]
        self.llm = get_peft_model(llm, LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
            task_type="CAUSAL_LM", target_modules=list(lora["target_modules"])))
        self.hidden = self.llm.config.hidden_size

        # Frozen RVQ codebooks: a non-persistent buffer, never optimised.
        self.register_buffer("codebooks", load_codebooks(codebooks_path).to(dtype), persistent=False)

        self.proj = nn.Sequential(
            nn.Linear(CODEBOOK_DIM, self.hidden, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden, dtype=dtype))
        cfg["projector"].update({"hidden_dim": self.hidden, "out_dim": self.hidden})

    # ---------------------------------------------------------------- audio path
    def _band(self, codes, layers):
        value = 0
        for i in layers:
            value = value + F.embedding(codes[:, i].long(), self.codebooks[i])
        return value

    def encode_audio(self, codes_l, codes_r, frame_mask):
        """codes_l/codes_r (B, 8, T), frame_mask (B, T) ->
        audio embeddings (B, 6*T', hidden) and pooled frame mask (B, T')."""
        content_l = self._band(codes_l, self.content_layers)
        sound_l = self._band(codes_l, self.sound_layers)
        content_r = self._band(codes_r, self.content_layers)
        sound_r = self._band(codes_r, self.sound_layers)
        content_s = (content_l - content_r) * 0.5
        sound_s = (sound_l - sound_r) * 0.5

        pooled_mask = frame_mask
        if self.time_pool > 1:
            content_l, pooled_mask = pool_time(content_l, frame_mask, self.time_pool, self.pool_content)
            content_r, _ = pool_time(content_r, frame_mask, self.time_pool, self.pool_content)
            content_s, _ = pool_time(content_s, frame_mask, self.time_pool, self.pool_content)
            sound_l, _ = pool_time(sound_l, frame_mask, self.time_pool, self.pool_sound)
            sound_r, _ = pool_time(sound_r, frame_mask, self.time_pool, self.pool_sound)
            sound_s, _ = pool_time(sound_s, frame_mask, self.time_pool, self.pool_sound)

        batch, frames, dim = content_l.shape
        slots = torch.stack([content_l, sound_l, content_r, sound_r, content_s, sound_s], dim=2)
        return self.proj(slots.reshape(batch, frames * self.roles_per_frame, dim)), pooled_mask

    def audio_attention(self, pooled_mask):
        return pooled_mask.repeat_interleave(self.roles_per_frame, dim=1)

    def forward(self, input_ids, attention_mask, labels, codes_l, codes_r, frame_mask):
        """Audio tokens are prepended to the text; their labels are -100 (no loss)."""
        audio_embeds, pooled_mask = self.encode_audio(codes_l, codes_r, frame_mask)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        audio_attn = self.audio_attention(pooled_mask)
        audio_labels = torch.full(audio_attn.shape, -100, dtype=labels.dtype, device=labels.device)
        return self.llm(inputs_embeds=torch.cat([audio_embeds, text_embeds], dim=1),
                        attention_mask=torch.cat([audio_attn, attention_mask], dim=1),
                        labels=torch.cat([audio_labels, labels], dim=1))

    # ------------------------------------------------------------- persistence
    def save_pretrained(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.llm.peft_config["default"].save_pretrained(out_dir)
        save_file({k: v.detach().contiguous().cpu()
                   for k, v in get_peft_model_state_dict(self.llm).items()},
                  os.path.join(out_dir, ADAPTER_NAME), metadata={"format": "pt"})
        torch.save(self.proj.state_dict(), os.path.join(out_dir, PROJECTOR_NAME))
        with open(os.path.join(out_dir, CONFIG_NAME), "w", encoding="utf-8") as stream:
            json.dump(self.config, stream, indent=2)
            stream.write("\n")

    def load_weights(self, ckpt_dir):
        """Load projector + LoRA from a checkpoint directory (frozen parts are untouched)."""
        theirs = load_config(ckpt_dir)
        for key in ARCHITECTURE_KEYS:
            mine, other = self.config[key], theirs.get(key)
            if key == "projector":
                mine = {k: v for k, v in mine.items() if k in ("type", "in_dim")}
                other = {k: v for k, v in (other or {}).items() if k in ("type", "in_dim")}
            if mine != other:
                raise ValueError(f"{ckpt_dir}: config[{key!r}]={other!r} does not match {mine!r}")
        self.proj.load_state_dict(torch.load(os.path.join(ckpt_dir, PROJECTOR_NAME), map_location="cpu"))
        result = set_peft_model_state_dict(self.llm, load_file(os.path.join(ckpt_dir, ADAPTER_NAME)))
        unexpected = getattr(result, "unexpected_keys", [])
        if unexpected:
            raise ValueError(f"{ckpt_dir}: unexpected LoRA keys, e.g. {unexpected[:3]}")

    @classmethod
    def from_pretrained(cls, ckpt_dir, qwen_path, codebooks_path, **kwargs):
        model = cls(qwen_path, codebooks_path, config=load_config(ckpt_dir), **kwargs)
        model.load_weights(ckpt_dir)
        return model

    def enable_grad_checkpointing(self):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.llm.enable_input_require_grads()

    def prepare_for_generation(self):
        """Training disables the KV cache; generation needs it (set on both PEFT wrappers)."""
        self.llm.config.use_cache = True
        self.llm.base_model.model.config.use_cache = True
        return self

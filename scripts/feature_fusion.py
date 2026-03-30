import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import scipy.special
import os
import pickle
import time

# ---------------------------------------------------------------------------
# LLM embedding model configuration
# Supported models in the paper: CodeBERT, DistilCodeBERT, OpenAI text-emb-3-large, DeepSeek-Coder-33B
# Default: microsoft/codebert-base (CodeBERT, 12-layer, 768-dim)
# Override via environment variable: SE4SC_LLM_MODEL=distilbert/distilcodebert-base
# ---------------------------------------------------------------------------
DEFAULT_LLM_MODEL = "microsoft/codebert-base"
LLM_MODEL_NAME = os.environ.get("SE4SC_LLM_MODEL", DEFAULT_LLM_MODEL)

try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm_model = AutoModel.from_pretrained(LLM_MODEL_NAME)
except Exception as e:
    print(f"Error loading LLM model '{LLM_MODEL_NAME}': {e}")
    print("Set SE4SC_LLM_MODEL environment variable to a valid HuggingFace model name.")
    print("Supported: microsoft/codebert-base, microsoft/codebert-base-mlm, distilbert/distilcodebert-base")
    raise


def get_embeddings(text):
    """Get mean-pooled last hidden state embedding from LLM, with caching."""
    if text in _embedding_cache:
        return _embedding_cache[text]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = llm_model(**inputs)
    result = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    _embedding_cache[text] = result
    return result


# Embedding cache: identical control flow fragments produce identical CFEF vectors
_embedding_cache = {}


# ---------------------------------------------------------------------------
# PCA fitting: supports both mock bootstrap and real-data fitting
# ---------------------------------------------------------------------------
PCA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pca_model.pkl")
PCA_N_COMPONENTS = 3

def _build_snippet_text(jumpSeq, constraint, pc_hex):
    return f"<jumpSeq>{jumpSeq}</jumpSeq> <constraint>{constraint}</constraint> <pc>{pc_hex}</pc>"


def fit_pca_from_real_data(snippets):
    """
    Fit PCA on real control-flow fragments collected during symbolic execution.
    
    Args:
        snippets: list of snippet strings (pre-built via _build_snippet_text).
    
    Returns:
        Fitted PCA object.
    """
    embeddings_list = []
    for text in snippets:
        emb = get_embeddings(text)
        embeddings_list.append(emb)
    
    pca = PCA(n_components=PCA_N_COMPONENTS)
    pca.fit(embeddings_list)
    
    # Save for reuse
    with open(PCA_MODEL_PATH, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA model fitted on {len(snippets)} real snippets and saved to {PCA_MODEL_PATH}")
    return pca


def _load_or_bootstrap_pca():
    """Load saved PCA model, or bootstrap with representative snippets."""
    if os.path.exists(PCA_MODEL_PATH):
        with open(PCA_MODEL_PATH, 'rb') as f:
            pca = pickle.load(f)
        print(f"PCA model loaded from {PCA_MODEL_PATH}")
        return pca
    
    # Bootstrap with representative EVM control-flow snippets
    bootstrap_snippets = [
        {'jumpSeq': 'PUSH1 0x40 JUMPI', 'constraint': 'msg.sender == owner', 'pc': '0x12'},
        {'jumpSeq': 'PUSH1 0x10 JUMP', 'constraint': 'balance > 0', 'pc': '0x1a'},
        {'jumpSeq': 'PUSH2 0x100 JUMPI', 'constraint': 'block.timestamp >= deadline', 'pc': '0x20'},
        {'jumpSeq': 'DUP1 SWAP1 JUMP', 'constraint': 'amount <= allowance', 'pc': '0x30'},
        {'jumpSeq': 'PUSH1 0x60 JUMPI', 'constraint': 'value != 0', 'pc': '0x44'},
        {'jumpSeq': 'PUSH2 0x200 JUMP', 'constraint': '', 'pc': '0x58'},
        {'jumpSeq': 'PUSH1 0x80 JUMPI', 'constraint': 'approved == True', 'pc': '0x6c'},
        {'jumpSeq': 'SWAP1 PUSH1 0x50 JUMP', 'constraint': 'tokenId < totalSupply', 'pc': '0x80'},
    ]
    embeddings_list = [
        get_embeddings(_build_snippet_text(s['jumpSeq'], s['constraint'], s['pc']))
        for s in bootstrap_snippets
    ]
    pca = PCA(n_components=PCA_N_COMPONENTS)
    pca.fit(embeddings_list)
    print("PCA model bootstrapped with representative snippets (will be replaced by real data during training)")
    return pca


pca = _load_or_bootstrap_pca()


# ---------------------------------------------------------------------------
# Trainable Feature Fusion Module (PyTorch)
# ---------------------------------------------------------------------------

class FeatureFusionModule(nn.Module):
    """
    Coverage-driven interactive fusion module (Section 3.3.2).
    Fuses 3D CFEF and 10D SEF via SSI and SSF attention units
    with learnable projection matrices.
    """
    def __init__(self, cfef_dim=3, sef_dim=10, d_intermediate=8, output_dim=13):
        super().__init__()
        self.d_intermediate = d_intermediate
        self.output_dim = output_dim
        
        # SSI projection matrices (learnable)
        self.W_q_ssi = nn.Linear(cfef_dim, d_intermediate, bias=False)
        self.W_k_ssi = nn.Linear(sef_dim, d_intermediate, bias=False)
        self.W_v_ssi = nn.Linear(sef_dim, d_intermediate, bias=False)
        
        # SSF projection matrices (learnable)
        self.W_q_ssf = nn.Linear(d_intermediate, d_intermediate, bias=False)
        self.W_k_ssf = nn.Linear(cfef_dim, d_intermediate, bias=False)
        self.W_v_ssf = nn.Linear(cfef_dim, d_intermediate, bias=False)
        
        # Output projection
        self.W_out = nn.Linear(d_intermediate, output_dim, bias=False)
    
    def forward(self, cfef, sef, coverage_branch, coverage_path):
        """
        Args:
            cfef: (..., cfef_dim) tensor - CFEF vector(s)
            sef: (..., sef_dim) tensor - SEF vector(s)
            coverage_branch: float or (...,) tensor
            coverage_path: float or (...,) tensor
        Returns:
            unified: (..., output_dim) tensor - fused feature vector(s)
        """
        # Handle both single and batched inputs
        single = cfef.dim() == 1
        if single:
            cfef = cfef.unsqueeze(0)
            sef = sef.unsqueeze(0)
        batch_size = cfef.size(0)
        
        # Preprocessing: coverage weight
        if isinstance(coverage_branch, (int, float)):
            w_cov = torch.sigmoid(torch.tensor(coverage_branch + coverage_path, dtype=torch.float32, device=cfef.device))
            w_cov = w_cov.expand(batch_size)
        else:
            w_cov = torch.sigmoid(coverage_branch + coverage_path)  # (batch,)
        
        # Preprocess SEF with coverage weighting
        sef_scaled = sef / (torch.norm(sef, dim=-1, keepdim=True) + 1e-8)  # (batch, sef_dim)
        cov_weights = torch.ones_like(sef_scaled)
        cov_weights[:, 3] = w_cov
        cov_weights[:, 4] = w_cov
        sef_weighted = sef_scaled * cov_weights
        
        # Preprocess CFEF (L2 normalize)
        cfef_scaled = cfef / (torch.norm(cfef, dim=-1, keepdim=True) + 1e-8)  # (batch, cfef_dim)
        
        # SSI Unit: CFEF queries SEF
        Q_ssi = self.W_q_ssi(cfef_scaled)       # (batch, d)
        K_ssi = self.W_k_ssi(sef_weighted)       # (batch, d)
        V_ssi = self.W_v_ssi(sef_weighted)       # (batch, d)
        
        # Outer-product attention per sample
        scores_ssi = (Q_ssi.unsqueeze(2) * K_ssi.unsqueeze(1)) / (self.d_intermediate ** 0.5)  # (batch, d, d)
        attn_ssi = torch.softmax(scores_ssi, dim=2)  # (batch, d, d)
        Z1 = (attn_ssi @ V_ssi.unsqueeze(2)).squeeze(2)  # (batch, d)
        
        # SSF Unit: Z1 queries CFEF
        Q_ssf = self.W_q_ssf(Z1)                # (batch, d)
        K_ssf = self.W_k_ssf(cfef_scaled)        # (batch, d)
        V_ssf = self.W_v_ssf(cfef_scaled)        # (batch, d)
        
        scores_ssf = (Q_ssf.unsqueeze(2) * K_ssf.unsqueeze(1)) / (self.d_intermediate ** 0.5)
        attn_ssf = torch.softmax(scores_ssf, dim=2)
        Z2 = (attn_ssf @ V_ssf.unsqueeze(2)).squeeze(2)  # (batch, d)
        
        # Fusion coefficient
        alpha = w_cov.unsqueeze(1)  # (batch, 1)
        
        # Fused feature
        F_f = Z1 + alpha * Z2  # (batch, d)
        
        # Project to output dimension
        F_f_proj = self.W_out(F_f)  # (batch, output_dim)
        
        # Residual connection
        pad_sef = self.output_dim - sef_weighted.size(1)
        pad_cfef = self.output_dim - cfef_scaled.size(1)
        F_sef_padded = torch.cat([sef_weighted, torch.zeros(batch_size, pad_sef, device=sef.device)], dim=1)
        F_cfef_padded = torch.cat([torch.zeros(batch_size, pad_cfef, device=cfef.device), cfef_scaled], dim=1)
        residual = F_sef_padded + F_cfef_padded
        residual = residual / (torch.norm(residual, dim=-1, keepdim=True) + 1e-8)
        
        unified = F_f_proj + residual
        unified = unified / (torch.norm(unified, dim=-1, keepdim=True) + 1e-8)
        
        if single:
            unified = unified.squeeze(0)
        
        return unified


# Global fusion module instance (will be trained alongside regression model)
fusion_module = FeatureFusionModule()


# ---------------------------------------------------------------------------
# Main fusion function (called during symbolic execution)
# ---------------------------------------------------------------------------

def symflow_feature_fusion(jumpSeq, pc, sef, constraint="", coverage_branch=0.5, coverage_path=0.5):
    """
    Generate unified 13D feature vector for state prioritization (Sections 3.3-3.4).
    
    Args:
        jumpSeq (str): EVM instruction sequence (e.g., 'PUSH1 0x80 PUSH1 0x40 JUMPI').
        pc (int): Program counter (e.g., 0x12).
        sef (list): List of 10 floats [stackSize, successor, ..., subpath], normalized [0, 1].
        constraint (str): Symbolic expression of the current path condition.
        coverage_branch (float): SEF coverage_branch (index 3), normalized [0, 1].
        coverage_path (float): SEF coverage_path (index 4), normalized [0, 1].
    
    Returns:
        np.ndarray: Unified 13D feature vector, L2-normalized.
    """
    # Input validation
    if not isinstance(jumpSeq, str):
        raise ValueError(f"jumpSeq must be a string, got {type(jumpSeq)}")
    if not isinstance(pc, int):
        raise ValueError(f"pc must be an integer, got {type(pc)}")
    if not isinstance(constraint, str):
        constraint = str(constraint)
    
    sef = np.array(sef, dtype=np.float32)
    if len(sef) != 10 or not (sef >= 0).all() or not (sef <= 1).all():
        raise ValueError("SEF must be a list of 10 floats with values in [0, 1]")
    if not (0 <= coverage_branch <= 1) or not (0 <= coverage_path <= 1):
        raise ValueError("coverage_branch and coverage_path must be in [0, 1]")

    # Step 1: Build control flow fragment text (Section 3.3.1)
    input_text = _build_snippet_text(jumpSeq, constraint, hex(pc))

    # Step 2: Generate LLM embedding (Section 3.3.1)
    try:
        _t_emb = time.time()
        _was_cached = input_text in _embedding_cache
        embedding = get_embeddings(input_text)
        _emb_elapsed = time.time() - _t_emb
        # Store for benchmark access
        symflow_feature_fusion._last_embedding_time = _emb_elapsed
        symflow_feature_fusion._last_was_cached = _was_cached
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

    # Step 3: Reduce to 3D CFEF via PCA (Section 3.3.1)
    try:
        cfef = pca.transform(embedding.reshape(1, -1))[0]
        norm = np.linalg.norm(cfef) + 1e-8
        cfef = cfef / norm
    except Exception as e:
        print(f"Error in PCA reduction: {e}")
        raise

    # Step 4: Feature fusion via trainable module (Section 3.3.2)
    cfef_tensor = torch.tensor(cfef, dtype=torch.float32)
    sef_tensor = torch.tensor(sef, dtype=torch.float32)
    
    fusion_module.eval()
    with torch.no_grad():
        unified = fusion_module(cfef_tensor, sef_tensor, coverage_branch, coverage_path)
    
    return unified.numpy()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock SEF data
    sef = [0.3745, 0.9507, 0.7320, 0.5987, 0.1560, 0.1560, 0.0581, 0.8662, 0.6011, 0.7081]
    unified_feature = symflow_feature_fusion(
        jumpSeq="PUSH1 0x80 PUSH1 0x40 JUMPI",
        pc=0x12,
        constraint="msg.sender == owner",
        sef=sef,
        coverage_branch=sef[3],
        coverage_path=sef[4]
    )
    print(f"Unified Feature Vector: {unified_feature.tolist()}")
    print(f"Vector dimension: {len(unified_feature)}")
    print(f"L2 norm: {np.linalg.norm(unified_feature):.4f}")

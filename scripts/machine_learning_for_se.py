import time
import z3
from z3 import Solver, BitVec, is_bv, simplify
import sys
from bse_version2 import convert_to_symbolic_bytecode, OpcodeHandlers, SymbolicVariableGenerator, SymExec, BytecodeExecutor
from testSolc import func_solc, bytecode_to_opcodes
from collections import deque
import os
import logging
from constants import STACK_MAX, SUCCESSOR_MAX, TEST_CASE_NUMBER_MAX, DEPTH_MAX, ICNT_MAX, SUBPATH_MAX, REWARD_MAX

from feature_fusion import symflow_feature_fusion, fusion_module

# iterLearn
# genData # trainStrategy
# SymExec dataFromTests
# update
# extractFeature predict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import re

from pathlib import Path

all_single_part_execution_time = []
all_single_all_execution_time = []

# Refactored
class SYMFLOWModel:
    def __init__(self, input_dim=None, hidden_dims=[128, 64], learning_rate=0.001, feature_keys=None):
        """
        Initialize LEARCH model
        
        Args:
            input_dim (int): Feature dimension
            hidden_dims (list): Hidden layer dimensions
            learning_rate (float): Learning rate
            feature_keys (list): Feature key list
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_keys = feature_keys
        self.input_dim = input_dim if input_dim is not None else len(feature_keys) if feature_keys else None
        
        if self.input_dim is None:
            raise ValueError("input_dim or feature_keys must be provided")
        
        # Define neural network
        layers = []
        prev_dim = self.input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Sigmoid activation
        self.model = nn.Sequential(*layers).to(self.device)
        
        self.criterion = nn.MSELoss()
        # Joint optimizer: regression model + fusion module (Section 3.3.2)
        self.fusion_module = fusion_module.to(self.device)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.fusion_module.parameters()),
            lr=learning_rate
        )

    def prepare_data(self, dataset, test_size=0.2, random_state=42, batch_size=32):
        """
        Prepare dataset, convert to tensors, and split into training/validation sets.

        Args:
            dataset: List of tuples [(sef_10d, cfef_3d, cov_branch, cov_path, reward), ...].
            test_size: Proportion of dataset for validation (default: 0.2).
            random_state: Random seed for reproducibility (default: 42).
            batch_size: Batch size for DataLoader (default: 32).

        Returns:
            train_loader: DataLoader for training data.
            X_val: Dict with 'sef', 'cfef', 'cov_branch', 'cov_path' tensors.
            y_val: Validation reward tensor.
        """
        if not dataset:
            raise ValueError("Dataset is empty")

        # Extract components
        sefs = np.array([d[0] for d in dataset], dtype=np.float32)       # (n, 10)
        cfefs = np.array([d[1] for d in dataset], dtype=np.float32)      # (n, 3)
        cov_branches = np.array([d[2] for d in dataset], dtype=np.float32)  # (n,)
        cov_paths = np.array([d[3] for d in dataset], dtype=np.float32)     # (n,)
        rewards = np.array([d[4] for d in dataset], dtype=np.float32)       # (n,)

        # Convert to tensors
        sefs_t = torch.tensor(sefs, dtype=torch.float32)
        cfefs_t = torch.tensor(cfefs, dtype=torch.float32)
        cov_branches_t = torch.tensor(cov_branches, dtype=torch.float32)
        cov_paths_t = torch.tensor(cov_paths, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        # Split into training/validation sets
        n = len(dataset)
        if random_state is not None:
            torch.manual_seed(random_state)
        indices = torch.randperm(n)
        test_size_int = int(test_size * n)
        val_idx = indices[:test_size_int]
        train_idx = indices[test_size_int:]

        # Validation set
        X_val = {
            'sef': sefs_t[val_idx].to(self.device),
            'cfef': cfefs_t[val_idx].to(self.device),
            'cov_branch': cov_branches_t[val_idx].to(self.device),
            'cov_path': cov_paths_t[val_idx].to(self.device),
        }
        y_val = rewards_t[val_idx].to(self.device)

        # Training DataLoader
        train_dataset = TensorDataset(
            sefs_t[train_idx], cfefs_t[train_idx],
            cov_branches_t[train_idx], cov_paths_t[train_idx],
            rewards_t[train_idx]
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, X_val, y_val

    def train(self, train_loader, X_val, y_val, epochs=100, patience=3, min_delta=0.5e-4):
        """
        Train model with early stopping. Fusion module is jointly trained
        via backpropagation through the full forward pass.
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        if len(train_loader) == 0:
            raise ValueError("Empty training data")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training mode for both regression model and fusion module
            self.model.train()
            self.fusion_module.train()
            train_loss = 0.0
            total_samples = 0

            for batch in train_loader:
                batch_sef, batch_cfef, batch_cov_br, batch_cov_pa, batch_y = batch
                batch_sef = batch_sef.to(self.device)
                batch_cfef = batch_cfef.to(self.device)
                batch_cov_br = batch_cov_br.to(self.device)
                batch_cov_pa = batch_cov_pa.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass through fusion module (WITH gradients)
                fused = self.fusion_module(batch_cfef, batch_sef, batch_cov_br, batch_cov_pa)

                # Forward pass through regression model
                outputs = self.model(fused).squeeze()
                loss = self.criterion(outputs, batch_y)

                # Backward pass updates BOTH fusion module and regression model
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * batch_sef.size(0)
                total_samples += batch_sef.size(0)

            train_loss /= total_samples

            # Validation
            self.model.eval()
            self.fusion_module.eval()
            with torch.no_grad():
                val_fused = self.fusion_module(
                    X_val['cfef'], X_val['sef'],
                    X_val['cov_branch'], X_val['cov_path']
                )
                val_outputs = self.model(val_fused).squeeze()
                val_loss = self.criterion(val_outputs, y_val).item()

            logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.save("best_model.pth")
                logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at Epoch {epoch+1}, Best Val Loss: {best_val_loss:.4f}")
                    break

        self.load("best_model.pth")
        logger.info("Loaded best model")
        return self.model


    def predict(self, features_1, features_2):
        """
        Predict a single reward for a state using unified features from SEF and CFEF.

        Args:
            features_1: 10D SEF feature list, e.g., [0.0, 1.0, ..., 0.0].
            features_2: CFEF triple [opcode_string, pc, constraint], e.g., ["PUSH 0x56 JUMPI", 80, "x == 0x1234"].

        Returns:
            reward: Float, predicted reward value.
        """
        self.model.eval()
        self.fusion_module.eval()
        
        # Extract constraint from features_2 (index 2), default to empty string
        constraint = features_2[2] if len(features_2) > 2 else ""
        
        # Extract CFEF via frozen LLM + PCA (no gradient needed for inference)
        from feature_fusion import get_embeddings, pca, _build_snippet_text
        if not isinstance(constraint, str):
            constraint = str(constraint)
        input_text = _build_snippet_text(features_2[0], constraint, hex(features_2[1]))

        import time as _time
        _t_emb = _time.time()
        embedding = get_embeddings(input_text)
        self._last_embedding_time = _time.time() - _t_emb

        cfef = pca.transform(embedding.reshape(1, -1))[0]
        cfef = cfef / (np.linalg.norm(cfef) + 1e-8)

        # Fuse via trained fusion module (no gradient for inference)
        cfef_tensor = torch.tensor(cfef, dtype=torch.float32).to(self.device)
        sef_tensor = torch.tensor(np.array(features_1, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            fused = self.fusion_module(cfef_tensor, sef_tensor, features_1[3], features_1[4])
            reward = self.model(fused.unsqueeze(0)).squeeze().cpu().item()
        
        return reward

    def save(self, path):
        """Save model weights and fusion module weights."""
        torch.save({
            'model': self.model.state_dict(),
            'fusion_module': self.fusion_module.state_dict()
        }, path)

    def load(self, path):
        """Load model weights and fusion module weights."""
        checkpoint = torch.load(path)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            if 'fusion_module' in checkpoint:
                self.fusion_module.load_state_dict(checkpoint['fusion_module'])
        else:
            # Backward compatibility: old format only has model state_dict
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.fusion_module.to(self.device)


class LEARCHModel:
    def __init__(self, input_dim=None, hidden_dims=[128, 64], learning_rate=0.001, feature_keys=None):
        """
        Initialize LEARCH model
        
        Args:
            input_dim (int): Feature dimension
            hidden_dims (list): Hidden layer dimensions
            learning_rate (float): Learning rate
            feature_keys (list): Feature key list
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_keys = feature_keys
        self.input_dim = input_dim if input_dim is not None else len(feature_keys) if feature_keys else None
        
        if self.input_dim is None:
            raise ValueError("input_dim or feature_keys must be provided")
        
        # Define neural network
        layers = []
        prev_dim = self.input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Sigmoid activation
        self.model = nn.Sequential(*layers).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def prepare_data(self, dataset, test_size=0.2, random_state=42, batch_size=32):
        """
        Prepare dataset: convert to tensors and split train/val
        
        Args:
            dataset: List of [(features_dict, reward), ...]
            test_size: Validation set ratio
            random_state: Random seed
            batch_size: Batch size
        
        Returns:
            train_loader: Training data loader
            X_val, y_val: Validation set tensors
        """
        if not dataset:
            raise ValueError("Dataset is empty")
        
        # Extract features and rewards
        self.feature_keys = list(dataset[0][0].keys())
        features = np.array([[d[key] for key in self.feature_keys] for d, _ in dataset])
        rewards = np.array([r for _, r in dataset])
        # print(features)
        # print(rewards)
        # Validate data range
        if not (features >= 0).all() or not (features <= 1).all() or not (rewards >= 0).all() or not (rewards <= 1).all():
            raise ValueError("Features and rewards must be in [0, 1]")
        
        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Split train/val using random_split
        if random_state is not None:
            torch.manual_seed(random_state)
        full_dataset = TensorDataset(features, rewards)
        test_size_int = int(test_size * len(full_dataset))
        train_size = len(full_dataset) - test_size_int
        train_dataset, val_dataset = random_split(full_dataset, [train_size, test_size_int])
        

        X_val = torch.stack([x for x, _ in val_dataset]).to(self.device)
        y_val = torch.stack([y for _, y in val_dataset]).to(self.device)
        

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        return train_loader, X_val, y_val

    def train(self, train_loader, X_val, y_val, epochs=100, patience=3, min_delta=0.5e-4):
        """
        Train model with early stopping
        
        Args:
            train_loader: Training data loader
            X_val, y_val: Validation set tensors
            epochs: Max epochs (default 15, fast convergence)
            patience: Early stopping patience (default 3)
            min_delta: Min improvement threshold (default 1e-6)
        
        Returns:
            model: Trained model
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Validate input data
        if len(train_loader) == 0 or X_val.numel() == 0:
            raise ValueError("Empty training or validation data")
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training mode
            self.model.train()
            train_loss = 0.0
            total_samples = 0
            
            # Batch training
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)  # weighted by batch size
                total_samples += batch_X.size(0)
            
            # Compute average training loss
            train_loss /= total_samples
            
            # Validation mode
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = self.criterion(val_outputs, y_val).item()
            
            # Log losses
            logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping: save only on significant improvement
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.save("best_model.pth")
                logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at Epoch {epoch+1}, Best Val Loss: {best_val_loss:.4f}")
                    break
        
        # Load best model
        self.load("best_model.pth")
        logger.info("Loaded best model")
        return self.model

    def predict(self, features):
        """
        Predict state rewards
        
        Args:
            features: List of [{features_dict}, ...]
        
        Returns:
            rewards: Predicted reward array
        """
        self.model.eval()
        if not features:
            raise ValueError("Features list is empty")
        for f in features:
            if list(f.keys()) != self.feature_keys:
                raise ValueError(f"Feature keys mismatch: {f.keys()} vs {self.feature_keys}")
        with torch.no_grad():
            inputs = torch.tensor(
                [[f[key] for key in self.feature_keys] for f in features],
                dtype=torch.float32
            ).to(self.device)
            rewards = self.model(inputs).squeeze().cpu().numpy()
            if rewards.ndim == 0:
                rewards = np.array([rewards.item()])  # Convert 0D to 1D
            # No np.clip needed: Sigmoid guarantees output in (0, 1)
        return rewards

    def save(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

def collect_sol_files(folder_path):
    sol_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.sol'):
            sol_files.append(os.path.join(folder_path, file))
    return sol_files

# Refactored
def iterLearn(way, progs, strategies, N=3):
    dataset = []
    learned = []
    for i in range(N):
        print("...")
        newData = genData(progs, strategies, way)
        dataset.extend(newData)


        if way == "learch":
            model = LEARCHModel(input_dim=10)
        else:
            model = SYMFLOWModel(input_dim=13)

        train_loader, X_val, y_val = model.prepare_data(dataset)

        model.train(train_loader, X_val, y_val)
        if way == "learch":
            model.save(f"best_learch_model_round_{i+1}.pth")  # Round-specific
        else:
            model.save(f"best_symflow_model_round_{i+1}.pth")  # Round-specific
        learned.append(model)
        if way == "learch":
            strategies = [["learch", model]]
        else:
            strategies = [["symflow", model]]
    return learned

def genData(progs, strategies, way):
    dataset = []
    for strategy in strategies:
        print("...")
        i = 1
        for prog in progs:
            print("...")
            # Compile smart contract
            symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(prog)
            # print(len(symbolic_and_real))
            # print(symbolic_and_real)
            for item in symbolic_and_real:
                if len(item[0]) > 0 and len(item[1]) > 0:
                    executor = SymExec(item[0], item[1], strategy, way)
                    success = executor.execute()

                    if success == False:
                        print("Contract lacks STOP/RETURN, skipping")
                        continue
                    else:
                        if way == "learch":
                            executor.prue_tree(executor.origin_node)
                            executor.count_node_reward(executor.origin_node)
                            part_dataset = build_dataset(executor.origin_node)
                            normalized_dataset = normalize(part_dataset, executor)
                            print(f"normalized_dataset:{normalized_dataset}")
                            dataset.extend(normalized_dataset)
                        else:

                            executor.prue_tree(executor.origin_node)
                            executor.count_node_reward(executor.origin_node)

                            # Collect real snippets for PCA fitting on first pass
                            from feature_fusion import fit_pca_from_real_data, _build_snippet_text, pca as _current_pca
                            import os as _os
                            _pca_path = _os.path.join(_os.path.dirname(__file__), "pca_model.pkl")
                            if not _os.path.exists(_pca_path):
                                snippets = []
                                def _collect_snippets(node):
                                    if node is None:
                                        return
                                    constraint = node.constraint if hasattr(node, 'constraint') else ""
                                    if not isinstance(constraint, str):
                                        constraint = str(constraint)
                                    snippets.append(_build_snippet_text(node.jumpSeq, constraint, hex(node.bytecode_list_index)))
                                    for child in node.children_node:
                                        _collect_snippets(child)
                                _collect_snippets(executor.origin_node)
                                if len(snippets) >= 10:
                                    fit_pca_from_real_data(snippets)
                                    # Reload the module-level pca
                                    import feature_fusion
                                    feature_fusion.pca = feature_fusion._load_or_bootstrap_pca()

                            fusioned_dataset = build_dataset_symflow(executor.origin_node, executor)
                            print(f"fusioned_dataset:{fusioned_dataset}")
                            dataset.extend(fusioned_dataset)
                else:
                    print("Compilation empty, skipping contract")
                    pass
            i += 1
    return dataset

def convert_runtime_opcode_to_symbolic_and_real(path):
    symbolic_and_real = []
    # symbolic_and_real = []
    with open(path,"r",) as file:
        Automata_contract = file.read()

    contracts_bytecode = func_solc(Automata_contract)
    
    for contract_id, (full_bytecode, runtime_bytecode) in contracts_bytecode.items():
        full_opcode = bytecode_to_opcodes(bytes.fromhex(full_bytecode))
        runtime_opcode = bytecode_to_opcodes(bytes.fromhex(runtime_bytecode))
        runtime_opcode_without_metadatahash = runtime_bytecode[:-88]
        runtime_opcode = bytecode_to_opcodes(
            bytes.fromhex(runtime_opcode_without_metadatahash)
        )
        symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)
        symbolic_and_real.append([symbolic_bytecode, runtime_opcode])



    # current_full_bytecode, current_runtime_bytecode = contracts_bytecode[contract_id]

    # runtime_opcode = bytecode_to_opcodes(
    #     bytes.fromhex(runtime_opcode_without_metadatahash)
    # )
    # symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)
    # symbolic_and_real.append([symbolic_bytecode, runtime_opcode])
    return symbolic_and_real
    
def bfs_read_test_tree(node, depth=0):
    indent = "    " * depth
    # stack_number={len(node.stack)} successor_number={node.successor_number} test_case_number={node.test_case_number}  depth={node.depth} cpicnt={node.cpicnt} icnt={node.icnt} covNew={node.covNew} subpath={node.subpath} branch_new_instruction={node.branch_new_instruction} path_new_instruction={node.path_new_instruction}
    print(f"{indent}- Node(index={node.bytecode_list_index} branch_new_block={node.branch_new_block} path_new_block={node.path_new_block} reward={node.reward}") # parent_node={node.parent_node} children_num={len(node.children_node)} branch_new_instruction_pc_range={node.branch_new_instruction_pc_range} executed={node.executed}
    for child in node.children_node:
        bfs_read_test_tree(child, depth + 1)
    
def count_leaf_nodes(node):

    if not node.children_node:
        return 1
    

    leaf_count = 0
    for child in node.children_node:
        leaf_count += count_leaf_nodes(child)
    
    return leaf_count

def build_dataset_symflow(head, se):
    dataset = []

    def traverse(current_node):
        if current_node is None:
            return
        # Extract feature dict
        features_1 = [current_node.stack_size, current_node.successor_number, current_node.test_case_number, current_node.branch_new_instruction, current_node.path_new_instruction, current_node.depth, current_node.cpicnt, current_node.icnt, current_node.covNew, current_node.subpath]
        features_1 = normalize_symflow(features_1, se)
        features_2 = [current_node.jumpSeq, current_node.bytecode_list_index, current_node.constraint]
        reward = current_node.reward

        # Extract CFEF (3D) from LLM embedding + PCA, without fusion
        from feature_fusion import get_embeddings, pca, _build_snippet_text
        constraint = features_2[2] if len(features_2) > 2 else ""
        if not isinstance(constraint, str):
            constraint = str(constraint)
        input_text = _build_snippet_text(features_2[0], constraint, hex(features_2[1]))
        embedding = get_embeddings(input_text)
        cfef = pca.transform(embedding.reshape(1, -1))[0]
        norm = np.linalg.norm(cfef) + 1e-8
        cfef = (cfef / norm).tolist()

        # Store raw components: (sef_10d, cfef_3d, coverage_branch, coverage_path, reward)
        dataset.append((features_1, cfef, features_1[3], features_1[4], min(reward / REWARD_MAX, 1)))
        
        # Recursively visit child nodes
        for child in current_node.children_node:
            traverse(child)

    traverse(head)
    return dataset

def normalize_symflow(features_1, se):
    # Normalize
    features_1[0] = min(features_1[0] / STACK_MAX, 1)
    features_1[1] = min(features_1[1] / SUCCESSOR_MAX, 1)
    features_1[2] = min(features_1[2] / TEST_CASE_NUMBER_MAX, 1)
    features_1[3] = min(features_1[3] / len(se.real_bytecode), 1)
    features_1[4] = min(features_1[4] / len(se.real_bytecode), 1)
    features_1[5] = min(features_1[5] / DEPTH_MAX, 1)
    features_1[6] = min(features_1[6] / len(se.real_bytecode), 1)
    features_1[7] = min(features_1[7] / ICNT_MAX, 1)
    features_1[8] = min(features_1[8] / len(se.real_bytecode), 1)
    features_1[9] = min(features_1[9] / SUBPATH_MAX, 1)

    return features_1

def build_dataset(head):
    dataset = []

    def traverse(current_node):
        if current_node is None:
            return
        # Extract feature dict
        features = {
            "stack_size": current_node.stack_size,
            "successor_number": current_node.successor_number,
            "test_case_number": current_node.test_case_number,
            "branch_new_instruction": current_node.branch_new_instruction,
            "path_new_instruction": current_node.path_new_instruction,
            "depth": current_node.depth,
            "cpicnt": current_node.cpicnt,
            "icnt": current_node.icnt,
            "covNew": current_node.covNew,
            "subpath": current_node.subpath,
        }
        reward = current_node.reward
        dataset.append((features, reward))
        
        # Recursively visit child nodes
        for child in current_node.children_node:
            traverse(child)

    traverse(head)
    return dataset

def normalize(dataset, se):
    """
    Normalize features in dataset
    
    Args:
        dataset: List of [(features_dict, reward), ...]
    
    Returns:
        normalized_dataset: Normalized dataset
    """
    normalized_dataset = []
    for features, reward in dataset:
        # Copy feature dict
        norm_features = features.copy()
        # Normalize
        norm_features["stack_size"] = min(features["stack_size"] / STACK_MAX, 1)
        norm_features["successor_number"] = min(features["successor_number"] / SUCCESSOR_MAX, 1)
        norm_features["test_case_number"] = min(features["test_case_number"] / TEST_CASE_NUMBER_MAX, 1)
        norm_features["branch_new_instruction"] = min(features["branch_new_instruction"] / len(se.real_bytecode), 1)
        norm_features["path_new_instruction"] = min(features["path_new_instruction"] / len(se.real_bytecode), 1)
        norm_features["depth"] = min(features["depth"] / DEPTH_MAX, 1)
        norm_features["cpicnt"] = min(features["cpicnt"] / len(se.real_bytecode), 1)
        norm_features["icnt"] = min(features["icnt"] / ICNT_MAX, 1)
        norm_features["covNew"] = min(features["covNew"] / len(se.real_bytecode), 1)
        norm_features["subpath"] = min(features["subpath"] / SUBPATH_MAX, 1)
        # Normalize
        normalized_dataset.append((norm_features, min(reward / REWARD_MAX, 1)))
    return normalized_dataset


def trained_symflow_model_use_for_se(model_path, folder_path, way):
    trained_model = SYMFLOWModel(input_dim=13)
    trained_model.load(model_path)

    from vulnerability_detector import VulnerabilityDetector

    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return {}

    results = {"coverage":[], "select_state_accuracy":[], "arrive_assigned_coverage_time":[], "vulnerabilities":[]}
    i = 1


    for sol_file in folder.glob("*.sol"):
        print("...")
        smart_contract_path = str(sol_file)
        print(f"smart_contract_path:{smart_contract_path}")
        
        symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(smart_contract_path)
        # runtime_opcode = convert_func(smart_contract_path)
        # symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)

        for item in symbolic_and_real:
            if len(item[0]) > 0 and len(item[1]) > 0:
                detector = VulnerabilityDetector()
                executor = SymExec(item[0], item[1], ["symflow",trained_model], way, detector=detector)
                success = executor.execute()
                if success == False:
                    print("Contract lacks STOP/RETURN, skipping")
                else:
                    results["coverage"].append(executor.coverage)
                    results["select_state_accuracy"].append(executor.select_state_accuracy)
                    if (executor.arrive_assigned_coverage_time != []):
                        results["arrive_assigned_coverage_time"].append(executor.arrive_assigned_coverage_time[0])
                    else:
                        results["arrive_assigned_coverage_time"].append(None)
                    results["vulnerabilities"].append(detector.get_results())
            else:
                print("Compilation empty, skipping contract")
                pass
        i += 1
    
    print(results)


def trained_learch_model_use_for_se(model_path, folder_path, way):
    from vulnerability_detector import VulnerabilityDetector

    feature_keys = ["stack_size", "successor_number", "test_case_number", "branch_new_instruction", "path_new_instruction", "depth", "cpicnt", "icnt", "covNew", "subpath"]
    trained_model = LEARCHModel(input_dim=10, feature_keys=feature_keys)
    trained_model.load(model_path)

    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return {}

    results = {"coverage":[], "select_state_accuracy":[], "arrive_assigned_coverage_time":[], "vulnerabilities":[]}
    i = 1

    for sol_file in folder.glob("*.sol"):
        print("...")
        smart_contract_path = str(sol_file)
        print(f"smart_contract_path:{smart_contract_path}")
        
        symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(smart_contract_path)

        for item in symbolic_and_real:
            if len(item[0]) > 0 and len(item[1]) > 0:
                detector = VulnerabilityDetector()
                executor = SymExec(item[0], item[1], ["learch",trained_model], way, detector=detector)
                success = executor.execute()
                if success == False:
                    print("Contract lacks STOP/RETURN, skipping")
                else:
                    results["coverage"].append(executor.coverage)
                    results["select_state_accuracy"].append(executor.select_state_accuracy)
                    if (executor.arrive_assigned_coverage_time != []):
                        results["arrive_assigned_coverage_time"].append(executor.arrive_assigned_coverage_time[0])
                    else:
                        results["arrive_assigned_coverage_time"].append(None)
                    results["vulnerabilities"].append(detector.get_results())
            else:
                print("Compilation empty, skipping contract")
                pass
        i += 1
    
    print(results)


def rss_use_for_se(folder_path, way):
    from vulnerability_detector import VulnerabilityDetector

    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return {}

    results = {"coverage":[], "select_state_accuracy":[], "arrive_assigned_coverage_time":[], "vulnerabilities":[]}
    i = 1
    for sol_file in folder.glob("*.sol"):
        print("...")
        smart_contract_path = str(sol_file)
        print(f"smart_contract_path:{smart_contract_path}")

        symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(smart_contract_path)

        for item in symbolic_and_real:
            if len(item[0]) > 0 and len(item[1]) > 0:
                detector = VulnerabilityDetector()
                executor = SymExec(item[0], item[1], "rss", way, detector=detector)
                success = executor.execute()
                if success == False:
                    print("Contract lacks STOP/RETURN, skipping")
                else:
                    results["coverage"].append(executor.coverage)
                    results["select_state_accuracy"].append(executor.select_state_accuracy)
                    if (executor.arrive_assigned_coverage_time != []):
                        results["arrive_assigned_coverage_time"].append(executor.arrive_assigned_coverage_time[0])
                    else:
                        results["arrive_assigned_coverage_time"].append(None)
                    results["vulnerabilities"].append(detector.get_results())

            else:
                print("Compilation empty, skipping contract")
                pass
        i += 1
    print(results)


def baseline_strategy_use_for_se(strategy, folder_path, way="learch"):
    """Run a pluggable baseline strategy on all contracts in folder_path.

    Args:
        strategy: a BaseStrategy instance (e.g. MythrilBFS())
        folder_path: path to folder containing .sol files
        way: node type to use ("learch" for TestTreeNode)
    """
    from vulnerability_detector import VulnerabilityDetector

    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return {}

    results = {"coverage": [], "select_state_accuracy": [], "arrive_assigned_coverage_time": [], "vulnerabilities": []}
    i = 1
    for sol_file in folder.glob("*.sol"):
        print("...")
        smart_contract_path = str(sol_file)
        print(f"smart_contract_path:{smart_contract_path}")

        symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(smart_contract_path)

        for item in symbolic_and_real:
            if len(item[0]) > 0 and len(item[1]) > 0:
                detector = VulnerabilityDetector()
                executor = SymExec(item[0], item[1], strategy, way, detector=detector)
                success = executor.execute()
                if success == False:
                    print("Contract lacks STOP/RETURN, skipping")
                else:
                    results["coverage"].append(executor.coverage)
                    results["select_state_accuracy"].append(executor.select_state_accuracy)
                    if executor.arrive_assigned_coverage_time:
                        results["arrive_assigned_coverage_time"].append(executor.arrive_assigned_coverage_time[0])
                    else:
                        results["arrive_assigned_coverage_time"].append(None)
                    results["vulnerabilities"].append(detector.get_results())
            else:
                print("Compilation empty, skipping contract")
                pass
        i += 1

    print(results)
    return results


def count_sm_bytecode_len(folder_path, way):

    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return {}

    results = {"bytecode_len":[]}
    i = 1

    for sol_file in folder.glob("*.sol"):
        print("...")
        smart_contract_path = str(sol_file)
        print(f"smart_contract_path:{smart_contract_path}")
        
        symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(smart_contract_path)
        # runtime_opcode = convert_func(smart_contract_path)
        # symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)

        for item in symbolic_and_real:
            if len(item[0]) > 0 and len(item[1]) > 0:
                executor = SymExec(item[0], item[1], "rss", way)
                success = executor.execute()
                if success == False:
                    print("Contract lacks STOP/RETURN, skipping")
                else:
                    results["bytecode_len"].append(len(executor.real_bytecode))
                    print(f"bytecode_len:{len(executor.real_bytecode)}")
            else:
                print("Compilation empty, skipping contract")
                pass
        i += 1
    
    print(results)



def trained_model_use_for_predict(model_path, feature_demo):
    feature_keys = ["stack_size", "successor_number", "test_case_number", "branch_new_instruction", "path_new_instruction", "depth", "cpicnt", "icnt", "covNew", "subpath"]
    trained_model = LEARCHModel(input_dim=10, feature_keys=feature_keys)
    trained_model.load(model_path)
    predicted_rewards = trained_model.predict(feature_demo)
    print(f"Predicted Rewards: {predicted_rewards}")


def convert_func(path):
    # symbolic_and_real = []
    with open(path,"r",) as file:
        Automata_contract = file.read()

    contracts_bytecode = func_solc(Automata_contract)


    contract_id = list(contracts_bytecode.keys())[0]

    current_full_bytecode, current_runtime_bytecode = contracts_bytecode[contract_id]
    runtime_opcode_without_metadatahash = current_runtime_bytecode[:-88]
    runtime_opcode = bytecode_to_opcodes(
        bytes.fromhex(runtime_opcode_without_metadatahash)
    )
    # symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)
    # symbolic_and_real.append([symbolic_bytecode, runtime_opcode])
    # return symbolic_and_real
    return runtime_opcode

def main():
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_smartcontract_dataset", "dataset_for_train")
    sol_files = collect_sol_files(dataset_dir)
    # iterLearn("learch", sol_files, ["rss"])
    iterLearn("symflow", sol_files, ["rss"])
    

# def main():

#     progs = convert_runtime_opcode_to_symbolic_and_real()
#     se = SymExec(progs[0][0], progs[0][1], "rss")
#     all_stacks = se.execute()
    
#     print(f"se.control_flow_graph:{se.control_flow_graph}")



#     print(f"se.passed_program_paths:{se.passed_program_paths}")
#     for item in se.passed_program_paths:
#         print(se.real_bytecode[item[-1]])

#     print(f"se.passed_program_paths_to_passed_number:{se.passed_program_paths_to_passed_number}")

#     print(f"se.subpath_k4_to_number:{se.subpath_k4_to_number}")










#     dataset = build_dataset(se.origin_node)
#     print(f"dataset:{dataset}")
#     print(f"len(dataset):{len(dataset)}")

#     normalized_dataset = normalize(dataset, se)
#     print(f"normalized_dataset:{normalized_dataset}")

#     # se = SymExec(progs[0][0], progs[0][1], "learned")




# def demo_train():

#     dataset = 

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt

class ExpertDemonstrationDataset(Dataset):
    """Dataset that preserves expert demonstration patterns without reward engineering."""
    
    def __init__(self, csv_file: str, context_len: int = 3, min_sequence_len: int = 3):
        self.data = pd.read_csv(csv_file)
        self.context_len = context_len
        self.min_sequence_len = min_sequence_len
        
        # State features
        self.state_columns = [
            'cat_id', 'confidence', 'box_area', 'dx', 'dy', 'darea',
            'elapsed_time', 'last_action_id', 'cat_offset', 'aspect_ratio',
            'visibility_score', 'velocity_mag', 'sensor_distance', 'last_known_offset'
        ]
        
        # Create longer sequences to capture behavioral patterns
        self.sequences = self._create_behavioral_sequences()
        
        print(f"Created {len(self.sequences)} behavioral sequences")
        self._analyze_sequence_diversity()
        
    def _create_behavioral_sequences(self):
        """Create sequences that capture behavioral patterns from expert demonstrations."""
        sequences = []
        
        # Use sliding window but with longer context to capture patterns
        for i in range(len(self.data) - self.context_len + 1):
            sequence_data = self.data.iloc[i:i + self.context_len]
            
            # Skip sequences where there's too much time gap (different sessions)
            time_diffs = sequence_data['timestamp'].diff().iloc[1:]
            if any(time_diffs > 10):  # Skip if >10 second gaps
                continue
            
            states = sequence_data[self.state_columns].values.astype(np.float32)
            actions = sequence_data['selected_action'].values.astype(np.int64)
            
            # Use constant returns for pure behavioral cloning
            returns = np.ones(len(actions), dtype=np.float32)
            returns_to_go = np.cumsum(returns[::-1])[::-1].copy()
            
            # Add sequence diversity metrics
            action_diversity = len(set(actions))
            state_changes = np.sum(np.abs(np.diff(states, axis=0)))
            
            sequences.append({
                'states': states,
                'actions': actions,
                'returns_to_go': returns_to_go.reshape(-1, 1),
                'timesteps': np.arange(len(states)),
                'action_diversity': action_diversity,
                'state_changes': state_changes,
                'sequence_id': i
            })
        
        return sequences
    
    def _analyze_sequence_diversity(self):
        """Analyze the diversity in our sequences."""
        print("\n📊 SEQUENCE ANALYSIS:")
        print("-" * 50)
        
        # Action pattern analysis
        action_patterns = []
        for seq in self.sequences:
            pattern = tuple(seq['actions'])
            action_patterns.append(pattern)
        
        pattern_counts = Counter(action_patterns)
        unique_patterns = len(pattern_counts)
        total_sequences = len(action_patterns)
        
        print(f"Unique action patterns: {unique_patterns}/{total_sequences} ({unique_patterns/total_sequences*100:.1f}%)")
        
        # Most common patterns
        print("\nMost common action patterns:")
        for pattern, count in pattern_counts.most_common(10):
            action_names = ["Stop", "Right", "Left", "Back", "Forward"]
            pattern_str = " → ".join([action_names[a] for a in pattern])
            print(f"  {pattern_str}: {count} times")
        
        # Action diversity per sequence
        diversities = [seq['action_diversity'] for seq in self.sequences]
        print(f"\nAction diversity per sequence:")
        print(f"  Mean: {np.mean(diversities):.2f}")
        print(f"  Min: {min(diversities)}, Max: {max(diversities)}")
        
        # State change analysis
        state_changes = [seq['state_changes'] for seq in self.sequences]
        print(f"\nState variability:")
        print(f"  Mean change: {np.mean(state_changes):.2f}")
        print(f"  High variability sequences: {sum(1 for x in state_changes if x > np.mean(state_changes))}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return {
            'states': torch.tensor(sequence['states'], dtype=torch.float32),
            'actions': torch.tensor(sequence['actions'], dtype=torch.long),
            'returns_to_go': torch.tensor(sequence['returns_to_go'], dtype=torch.float32),
            'timesteps': torch.tensor(sequence['timesteps'], dtype=torch.long)
        }

class ImprovedDecisionTransformer(nn.Module):
    """Improved Decision Transformer with better capacity for learning diverse behaviors."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_len: int = 3,
        hidden_size: int = 64,  # Larger for better capacity
        n_layers: int = 3,      # More layers
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.hidden_size = hidden_size
        
        # Better embeddings
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.action_embedding = nn.Embedding(action_dim, hidden_size)
        self.return_embedding = nn.Linear(1, hidden_size)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, 3 * context_len, hidden_size))
        
        # Transformer with residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=2 * hidden_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer norm
        self.ln_f = nn.LayerNorm(hidden_size)
        
        # Better action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_dim)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, states, actions, returns_to_go):
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Embed tokens
        state_embeddings = self.state_embedding(states)
        action_embeddings = self.action_embedding(actions)
        return_embeddings = self.return_embedding(returns_to_go)
        
        # Stack embeddings: (return, state, action) for each timestep
        token_embeddings = torch.stack(
            [return_embeddings, state_embeddings, action_embeddings], dim=2
        ).reshape(batch_size, 3 * seq_len, self.hidden_size)
        
        # Add positional embeddings
        if token_embeddings.shape[1] <= self.pos_embedding.shape[1]:
            pos_embeddings = self.pos_embedding[:, :token_embeddings.shape[1], :]
        else:
            pos_embeddings = self.pos_embedding[:, -token_embeddings.shape[1]:, :]
        
        x = token_embeddings + pos_embeddings
        
        # Causal mask
        seq_len_tokens = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len_tokens, seq_len_tokens), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)
        
        # Transformer
        x = self.transformer(x, mask=causal_mask)
        x = self.ln_f(x)
        
        # Extract action predictions (every 3rd token starting from index 1)
        action_indices = torch.arange(1, seq_len_tokens, 3, device=x.device)
        action_tokens = x[:, action_indices, :]
        
        # Predict actions
        action_logits = self.action_head(action_tokens)
        
        return action_logits

def train_expert_demonstration_model(
    csv_file: str,
    model_save_path: str = "expert_demo_transformer.pt",
    context_len: int = 3,
    state_dim: int = 14,
    action_dim: int = 5,
    hidden_size: int = 64,
    n_layers: int = 3,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    num_epochs: int = 200,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train model purely on expert demonstrations without reward engineering."""
    
    print(f"Training Expert Demonstration Model on {device}")
    print(f"Context length: {context_len}")
    print(f"Model size: {hidden_size} hidden, {n_layers} layers")
    
    # Create dataset
    dataset = ExpertDemonstrationDataset(csv_file, context_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = ImprovedDecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        context_len=context_len,
        hidden_size=hidden_size,
        n_layers=n_layers
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader)
    )
    
    # Loss function - simple cross entropy for behavioral cloning
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        action_correct = {i: 0 for i in range(action_dim)}
        action_total = {i: 0 for i in range(action_dim)}
        
        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            returns_to_go = batch['returns_to_go'].to(device)
            
            # Forward pass
            action_logits = model(states, actions, returns_to_go)
            
            # Calculate loss
            loss = criterion(action_logits.reshape(-1, action_dim), actions.reshape(-1))
            
            # Calculate accuracy
            predictions = torch.argmax(action_logits.reshape(-1, action_dim), dim=1)
            targets = actions.reshape(-1)
            
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.shape[0]
            
            # Per-action accuracy
            for i in range(action_dim):
                mask = targets == i
                if mask.sum() > 0:
                    action_correct[i] += (predictions[mask] == i).sum().item()
                    action_total[i] += mask.sum().item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        overall_accuracy = correct_predictions / total_predictions
        
        # Per-action accuracy
        action_names = ["Stop", "Right", "Left", "Back", "Forward"]
        action_accs = []
        for i in range(action_dim):
            if action_total[i] > 0:
                acc = action_correct[i] / action_total[i]
                action_accs.append(acc)
            else:
                action_accs.append(0.0)
        
        print(f"Epoch {epoch+1:3d}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {overall_accuracy:.3f}")
        print(f"    Per-action: " + " ".join([f"{name[:4]}:{acc:.2f}" for name, acc in zip(action_names, action_accs)]))
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"    ✓ Saved best model")
    
    print("Training completed!")
    
    # Model size info
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"\nFinal model stats:")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Expected Pi 5 inference: ~{total_params/10000:.1f}ms")
    
    return model

if __name__ == "__main__":
    print("="*80)
    print("EXPERT DEMONSTRATION TRAINING")
    print("="*80)
    
    # Train model
    model = train_expert_demonstration_model(
        csv_file="Flattened_State_Vectors2.csv",
        model_save_path="expert_demo_transformer.pt",
        context_len=3,  # Longer context for better patterns
        hidden_size=64, # More capacity
        n_layers=3,     # Deeper network
        batch_size=24,
        learning_rate=3e-4,
        num_epochs=150
    )
    
    print("\nExpert demonstration model ready!")
    print("Update your Pi 5 code to load 'expert_demo_transformer.pt'")
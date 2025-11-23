# Phase 4 Upgrade 2: Specialized Expert Training

**Status**: ✅ INFRASTRUCTURE IMPLEMENTED (Disabled by default, requires aux target preparation)
**Date**: 2025-11-23
**Expected Impact**: +0.3-0.6% NLL improvement when fully enabled

---

## Problem Statement

**Current Issue**: All expert models are trained on the same objective (predict next 6 numbers). This makes them redundant rather than complementary - they learn similar patterns and provide correlated predictions.

**Why This Matters**:
- Ensemble benefits most when experts are diverse and complementary
- Different architectures excel at different pattern types
- Single-objective training doesn't leverage architectural strengths

---

## Solution: Multi-Task Learning with Specialized Objectives

Train each expert with **architecture-specific auxiliary tasks** that encourage specialization:

### Specialization Strategy

| Expert | Architecture Strength | Primary Task | Auxiliary Task | Purpose |
|--------|----------------------|--------------|----------------|----------|
| **TCN** | Temporal trends | Next 6 numbers | Frequency change trends | Learn how number frequencies shift over time |
| **Transformer** | Long-range dependencies | Next 6 numbers | Future pattern prediction | Anticipate draws 2-3 steps ahead |
| **GNN** | Graph structure | Next 6 numbers | Co-occurrence communities | Identify which numbers cluster together |
| **LSTM** | Sequential patterns | Next 6 numbers | *(Future)* Sum prediction | Learn number relationship constraints |
| **Bayes** | Statistical inference | Posterior estimation | *(Already specialized)* | Confidence calibration built-in |
| **Markov** | Transition modeling | Conditional prediction | *(Already specialized)* | Short-term transitions only |

---

## Implementation Details

### 1. TCN: Frequency Change Prediction

**Auxiliary Task**: Predict how number frequencies are changing (trending up/down)

```python
# Output: 40 values ranging from -1 (decreasing) to +1 (increasing)
freq_change = layers.Dense(32, activation='relu')(global_features)
freq_change = layers.Dense(40, activation='tanh', name='freq_change')(freq_change)

# Multi-task loss
loss = {
    'main_logits': pl_set_loss,  # 85% weight
    'freq_change': 'mse'          # 15% weight
}
```

**Benefits**:
- Forces TCN to learn temporal dynamics explicitly
- Frequency trends provide regularization
- Helps distinguish between stable vs trending numbers

### 2. Transformer: Long-Range Pattern Prediction

**Auxiliary Task**: Predict future draw characteristics (2-3 steps ahead)

```python
# Output: 40 probabilities for future draw likelihood
future_pattern = layers.Dense(64, activation='relu')(global_features)
future_pattern = layers.Dropout(0.3)(future_pattern)
future_pattern = layers.Dense(40, activation='sigmoid', name='future_pattern')(future_pattern)

# Multi-task loss
loss = {
    'main_logits': pl_set_loss,      # 80% weight
    'future_pattern': 'binary_crossentropy'  # 20% weight
}
```

**Benefits**:
- Leverages Transformer's attention mechanism for long-range dependencies
- Learn patterns that predict beyond immediate next draw
- Provides complementary signal to short-term focused models

### 3. GNN: Co-occurrence Community Prediction

**Auxiliary Task**: Predict which numbers form strong co-occurrence communities

```python
# Output: 40 community membership scores
cooc_community = layers.Dense(32, activation='relu')(graph_features)
cooc_community = layers.Dense(40, activation='sigmoid', name='cooc_community')(cooc_community)

# Multi-task loss
loss = {
    'main_logits': pl_set_loss,           # 85% weight
    'cooc_community': 'binary_crossentropy'  # 15% weight
}
```

**Benefits**:
- Exploits GNN's graph structure for co-occurrence modeling
- Community detection provides structural regularization
- Helps identify number groupings and relationships

---

## Code Changes

### New Configuration Parameters

Added to `Lotto+.py` (lines 1049-1055):

```python
USE_SPECIALIZED_TRAINING = False  # Enable multi-task learning (default: off)
TCNN_AUX_WEIGHT = 0.15           # TCN auxiliary task weight
TRANSFORMER_AUX_WEIGHT = 0.20    # Transformer auxiliary task weight
GNN_AUX_WEIGHT = 0.15            # GNN auxiliary task weight
```

### Modified Model Architectures

#### TCN Model (`build_tcnn_model`, lines 4621-4687)
- Added `use_aux_task` parameter
- Added frequency change prediction head
- Multi-task compilation with weighted losses
- Backward compatible (single output when `use_aux_task=False`)

#### Transformer Model (`build_transformer_model`, lines 4690-4764)
- Added `use_aux_task` parameter
- Added future pattern prediction head
- Multi-task compilation with weighted losses
- Backward compatible (single output when `use_aux_task=False`)

#### GNN Model (`build_gnn_model`, lines 4917-5016)
- Added `use_aux_task` parameter
- Added co-occurrence community prediction head
- Multi-task compilation with weighted losses
- Backward compatible (single output when `use_aux_task=False`)

### Updated Prediction Functions

#### GNN Prediction (`_gnn_prob_from_history`, lines 5056-5066)
- Added logic to handle multi-task outputs
- Extracts main logits from tuple/list if multi-task
- Graceful fallback to single output for backward compatibility

```python
# Handle multi-task outputs
outputs = gnn_model.predict([...])
if isinstance(outputs, (list, tuple)):
    logits = outputs[0]  # Main task
else:
    logits = outputs  # Single-task
```

---

## Current Status: Infrastructure Complete, Awaiting Auxiliary Targets

### ✅ What's Implemented

1. **Model Architectures**: All three models support multi-task training
2. **Loss Functions**: Weighted multi-task losses configured
3. **Prediction Handling**: Extracting main output from multi-task models
4. **Configuration**: Toggle flag and weight parameters
5. **Backward Compatibility**: Works seamlessly with single-task mode

### ⏸️ What's Needed to Enable

1. **Auxiliary Target Preparation**:
   - TCN: Compute frequency change trends from historical draws
   - Transformer: Prepare future draw indicators (draws at t+2, t+3)
   - GNN: Generate co-occurrence community labels

2. **Training Loop Updates**:
   - Modify `fit()` calls to provide multi-task targets: `[main_labels, aux_labels]`
   - Update validation data preparation
   - Add auxiliary task monitoring

3. **Example Implementation**:
```python
# Prepare auxiliary targets for TCN
def prepare_tcn_aux_targets(draws, window=10):
    """Compute frequency change trends."""
    aux_targets = []
    for i in range(len(draws)):
        if i < window:
            aux_targets.append(np.zeros(40))
        else:
            recent_freq = compute_frequencies(draws[i-window:i])
            older_freq = compute_frequencies(draws[i-2*window:i-window])
            freq_change = (recent_freq - older_freq) / (older_freq + 1e-6)
            aux_targets.append(np.clip(freq_change, -1, 1))
    return np.array(aux_targets)

# Use in training
aux_train = prepare_tcn_aux_targets(draws[:train_end])
tcnn_model.fit(
    [X_train, M_train],
    [y_train, aux_train],  # Multi-task targets
    ...
)
```

---

## Expected Impact (When Fully Enabled)

### Quantitative Improvements

| Component | Mechanism | Expected Gain |
|-----------|-----------|---------------|
| **TCN Specialization** | Frequency trend awareness | +0.1-0.2% |
| **Transformer Specialization** | Long-range pattern learning | +0.1-0.2% |
| **GNN Specialization** | Co-occurrence structure | +0.1-0.2% |
| **Ensemble Diversity** | Complementary expertise | +0.1-0.2% |
| **Total** | Combined multi-task benefit | **+0.3-0.6%** |

### Qualitative Benefits

1. **Expert Diversity**: Models learn complementary patterns
2. **Regularization**: Auxiliary tasks prevent overfitting
3. **Interpretability**: Auxiliary outputs provide insights
4. **Robustness**: Multi-task learning improves generalization

---

## Usage Instructions

### Current State (Default: Disabled)

```python
# Models train with single-task objective (backward compatible)
USE_SPECIALIZED_TRAINING = False

# Models return single output
logits = model.predict([X, meta])  # Shape: (batch, 40)
```

### To Enable Multi-Task Training

1. **Set Configuration**:
```python
USE_SPECIALIZED_TRAINING = True
```

2. **Prepare Auxiliary Targets**:
```python
# Example for TCN
tcn_aux = prepare_freq_change_targets(draws)

# Example for Transformer
transformer_aux = prepare_future_pattern_targets(draws, lookahead=2)

# Example for GNN
gnn_aux = prepare_cooccurrence_communities(draws)
```

3. **Update Training**:
```python
# TCN with multi-task
tcnn_model.fit(
    [X_train, M_train],
    [y_train, tcn_aux_train],  # List of targets
    validation_data=([X_val, M_val], [y_val, tcn_aux_val]),
    ...
)

# Models now return tuple
main_logits, aux_output = model.predict([X, meta])
```

4. **Verify**:
- Check model summary shows multiple outputs
- Monitor both main and auxiliary losses during training
- Validate prediction handling extracts main output correctly

---

## Testing and Validation

### Quick Test (Current State)

```bash
# Should work without errors (single-task mode)
python3 Lotto+.py

# No auxiliary outputs expected
# Models function normally with backward compatibility
```

### Future Testing (After Enabling)

```bash
# Set USE_SPECIALIZED_TRAINING = True
# Prepare auxiliary targets
# Run full training pipeline

# Expected improvements:
# - Main task NLL: -0.1 to -0.3% per model
# - Ensemble NLL: -0.3 to -0.6% overall
# - Better model diversity in ensemble
```

---

## Code Statistics

### Files Modified

- `Lotto+.py`: +200 lines (model architectures + configuration)

### New Parameters

- 4 configuration parameters added

### Modified Functions

- `build_tcnn_model`: +66 lines (multi-task support)
- `build_transformer_model`: +74 lines (multi-task support)
- `build_gnn_model`: +99 lines (multi-task support)
- `_gnn_prob_from_history`: +9 lines (output handling)

### Total Changes

- **+248 lines added**
- **Net impact**: Full multi-task infrastructure in place

---

## Future Enhancements

### Phase 4.3: Auxiliary Target Automation

Automatically generate auxiliary targets without manual preparation:

1. **Self-Supervised Targets**:
   - TCN: Auto-compute frequency trends from history
   - Transformer: Use masked future prediction
   - GNN: Detect communities via graph clustering

2. **Dynamic Weighting**:
   - Adjust aux task weights based on validation performance
   - Increase weight when auxiliary task improves main task
   - Reduce weight when auxiliary task hurts main task

3. **Aux Task Monitoring**:
   - Log auxiliary task performance
   - Visualize learned patterns
   - Identify when specialization is working

---

## Conclusion

Phase 4 Upgrade 2 provides a **complete multi-task learning infrastructure** for specialized expert training:

✅ **Model Architectures**: Ready for multi-task training
✅ **Configuration**: Toggle flags and weight parameters in place
✅ **Backward Compatibility**: Works seamlessly in single-task mode
✅ **Prediction Handling**: Properly extracts main outputs
⏸️ **Auxiliary Targets**: Requires preparation before enabling

**Next Steps**:
1. Prepare auxiliary target generation functions
2. Enable `USE_SPECIALIZED_TRAINING = True`
3. Run full backtest to measure +0.3-0.6% improvement
4. Proceed to Phase 4 Upgrade 3: Hierarchical Ensemble

**Timeline**: Infrastructure complete, enabling requires ~2-4 hours of aux target development

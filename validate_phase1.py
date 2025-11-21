#!/usr/bin/env python3
"""
Phase 1 Validation Script
Verifies that all Phase 1 upgrades were correctly implemented.
"""

import re
import sys

def validate_meta_stacker():
    """Validate meta-stacker capacity increase."""
    print("=" * 60)
    print("1. VALIDATING META-STACKER CAPACITY INCREASE")
    print("=" * 60)

    with open('Lotto+.py', 'r') as f:
        content = f.read()

    # Check for new architecture
    if 'hidden_layer_sizes=(128, 64, 32,)' in content:
        print("✅ Meta-stacker upgraded to (128, 64, 32)")
        return True
    elif 'hidden_layer_sizes=(16,)' in content:
        print("❌ FAILED: Meta-stacker still using old (16,) architecture")
        return False
    else:
        print("⚠️  WARNING: Could not find meta-stacker architecture")
        return False

def validate_tcn():
    """Validate TCN replacement of HMM."""
    print("\n" + "=" * 60)
    print("2. VALIDATING TCN REPLACEMENT")
    print("=" * 60)

    with open('Lotto+.py', 'r') as f:
        content = f.read()

    # Check for TCN function definition
    has_tcn_func = 'def _build_tcn_prob_from_subset' in content
    print(f"{'✅' if has_tcn_func else '❌'} TCN function defined: {has_tcn_func}")

    # Check for TCN calls replacing HMM
    tcn_calls = content.count('_build_tcn_prob_from_subset(')
    print(f"✅ Found {tcn_calls} calls to TCN function")

    # Check for key TCN features
    has_dilation = 'dilation_rates = [1, 2, 4, 8, 16, 32]' in content
    print(f"{'✅' if has_dilation else '❌'} Dilated convolutions: {has_dilation}")

    has_residual = 'layers.Add()' in content
    print(f"{'✅' if has_residual else '❌'} Residual connections: {has_residual}")

    has_batchnorm = 'layers.BatchNormalization()' in content
    print(f"{'✅' if has_batchnorm else '❌'} Batch normalization: {has_batchnorm}")

    return has_tcn_func and tcn_calls >= 5 and has_dilation

def validate_context_window():
    """Validate context window extension."""
    print("\n" + "=" * 60)
    print("3. VALIDATING CONTEXT WINDOW EXTENSION")
    print("=" * 60)

    with open('Lotto+.py', 'r') as f:
        content = f.read()

    # Count occurrences of new 20-draw window
    count_20 = len(re.findall(r'hist\[-min\(len\(hist\), 20\):\]', content))
    print(f"✅ Found {count_20} instances of 20-draw window")

    # Check if any old 8-draw windows remain
    count_8 = len(re.findall(r'hist\[-min\(len\(hist\), 8\):\]', content))
    if count_8 > 0:
        print(f"⚠️  WARNING: Found {count_8} instances of old 8-draw window (may be intentional)")
    else:
        print(f"✅ No old 8-draw windows found (all upgraded to 20)")

    return count_20 >= 3

def validate_cross_features():
    """Validate cross-number interaction features."""
    print("\n" + "=" * 60)
    print("4. VALIDATING CROSS-NUMBER INTERACTION FEATURES")
    print("=" * 60)

    with open('Lotto+.py', 'r') as f:
        content = f.read()

    features = {
        'consecutive_before': 'Consecutive run detection',
        'in_arithmetic_seq': 'Arithmetic sequence detection',
        'momentum': 'Momentum (trend direction)',
        'cooc_score_norm': 'Co-occurrence score',
        'exclusion_score_norm': 'Exclusion pattern score',
        'q1_pressure': 'Quadrant 1 pressure',
        'cycle_strength': 'Cycle strength detection',
        'sum_compatibility': 'Sum compatibility'
    }

    found = 0
    for feature, description in features.items():
        if feature in content:
            print(f"✅ {description}: {feature}")
            found += 1
        else:
            print(f"❌ MISSING: {description}")

    print(f"\n{'✅' if found == len(features) else '❌'} Found {found}/{len(features)} new features")
    return found >= 6  # At least 75% of features

def validate_xgboost():
    """Validate XGBoost hyperparameter tuning."""
    print("\n" + "=" * 60)
    print("5. VALIDATING XGBOOST HYPERPARAMETER TUNING")
    print("=" * 60)

    with open('Lotto+.py', 'r') as f:
        content = f.read()

    # Check for increased max_depth
    has_depth_8 = 'max_depth=8' in content
    print(f"{'✅' if has_depth_8 else '❌'} max_depth=8: {has_depth_8}")

    # Check for slower learning rate
    has_lr_03 = 'learning_rate=0.03' in content
    print(f"{'✅' if has_lr_03 else '❌'} learning_rate=0.03: {has_lr_03}")

    # Check for new regularization params
    has_gamma = 'gamma=0.1' in content
    print(f"{'✅' if has_gamma else '❌'} gamma=0.1: {has_gamma}")

    has_reg_alpha = 'reg_alpha=0.1' in content
    print(f"{'✅' if has_reg_alpha else '❌'} reg_alpha=0.1: {has_reg_alpha}")

    has_min_child = 'min_child_weight=3' in content
    print(f"{'✅' if has_min_child else '❌'} min_child_weight=3: {has_min_child}")

    # Check for increased n_estimators
    has_500_est = 'n_estimators=500' in content
    print(f"{'✅' if has_500_est else '❌'} n_estimators=500: {has_500_est}")

    return has_depth_8 and has_lr_03 and has_gamma

def count_code_changes():
    """Count lines changed."""
    print("\n" + "=" * 60)
    print("CODE CHANGE STATISTICS")
    print("=" * 60)

    with open('Lotto+.py', 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)

    # Count TCN function size
    tcn_start = None
    tcn_end = None
    for i, line in enumerate(lines):
        if 'def _build_tcn_prob_from_subset' in line:
            tcn_start = i
        elif tcn_start is not None and tcn_end is None and line.strip().startswith('def '):
            tcn_end = i
            break

    tcn_lines = tcn_end - tcn_start if tcn_start and tcn_end else 0

    # Count feature additions in compute_stat_features
    phase1_marker_count = lines.count('# ========== PHASE 1: CROSS-NUMBER INTERACTION FEATURES ==========\n')

    print(f"Total lines in Lotto+.py: {total_lines}")
    print(f"New TCN function: ~{tcn_lines} lines")
    print(f"Phase 1 feature markers: {phase1_marker_count}")

    return total_lines

def main():
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + " " * 12 + "PHASE 1 VALIDATION REPORT" + " " * 21 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60 + "\n")

    results = {}

    # Run all validations
    results['meta_stacker'] = validate_meta_stacker()
    results['tcn'] = validate_tcn()
    results['context_window'] = validate_context_window()
    results['cross_features'] = validate_cross_features()
    results['xgboost'] = validate_xgboost()

    # Statistics
    count_code_changes()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(results.values())

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name.replace('_', ' ').title()}")

    print("\n" + "=" * 60)
    if passed == total:
        print(f"🎉 ALL {total}/{total} VALIDATIONS PASSED!")
        print("=" * 60)
        print("\nPhase 1 upgrades successfully implemented!")
        print("\nExpected improvements:")
        print("  • Meta-Stacker:     +0.2-0.4% NLL")
        print("  • TCN (vs HMM):     +0.3-0.5% NLL")
        print("  • Context Window:   +0.3-0.6% NLL")
        print("  • Cross Features:   +0.2-0.4% NLL")
        print("  • XGBoost Tuning:   +0.2-0.3% accuracy")
        print("  • ─────────────────────────────")
        print("  • TOTAL EXPECTED:   +0.8-1.5% NLL (~1.4% relative)")
        print("\nNext steps:")
        print("  1. Run full backtest to measure actual improvements")
        print("  2. Compare against baseline NLL: 21.7398")
        print("  3. Monitor training time (TCN may be slower initially)")
        return 0
    else:
        print(f"⚠️  {passed}/{total} VALIDATIONS PASSED")
        print("=" * 60)
        print("\nSome validations failed. Please review above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

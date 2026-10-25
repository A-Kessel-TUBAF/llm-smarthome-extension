# Classification: Gemma 4 vs ModernBERT

| Task | Model | Mode | n | Acc. | P micro | R micro | F1 micro | P macro | R macro | F1 macro | F1 macro 95% CI | Problems | Latency/item (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t1_dexa_device_types | gemma4:31b | MAS end to end (agents assigned a task) | 60 | 0.267 | 0.938 | 0.487 | 0.641 | 0.948 | 0.497 | 0.638 | 0.552–0.706 | 4 | 39.086 |
| t1_dexa_device_types | gemma4:31b | zero-shot | 60 | 0.550 | 0.940 | 0.812 | 0.871 | 0.940 | 0.834 | 0.872 | 0.825–0.906 | 0 | 1.769 |
| t1_dexa_device_types | MoritzLaurer/ModernBERT-large-zeroshot-v2.0 | zero-shot | 60 | 0.267 | 0.892 | 0.481 | 0.624 | 0.909 | 0.510 | 0.611 | 0.524–0.681 | 0 | 0.211 |
| t2_homebench_validity | gemma4:31b | zero-shot | 1500 | 0.873 | 0.873 | 0.873 | 0.873 | 0.890 | 0.873 | 0.872 | 0.856–0.888 | 0 | 8.657 |
| t2_homebench_validity | answerdotai/ModernBERT-large | finetune | 12952 | 0.946 | 0.946 | 0.946 | 0.946 | 0.929 | 0.961 | 0.940 | 0.936–0.944 | 0 | 0.026 |
| t3_simuhome_feasibility | gemma4:31b | zero-shot | 600 | 0.592 | 0.592 | 0.592 | 0.592 | 0.607 | 0.592 | 0.577 | 0.537–0.616 | 0 | 4.261 |
| t3_simuhome_feasibility | answerdotai/ModernBERT-large | cv | 600 | 0.788 | 0.788 | 0.788 | 0.788 | 0.794 | 0.788 | 0.787 | 0.755–0.820 | 0 | 0.010 |
| t4_homeassistant_domain | gemma4:31b | zero-shot | 95 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000–1.000 | 0 | 1.400 |
| t4_homeassistant_domain | MoritzLaurer/ModernBERT-large-zeroshot-v2.0 | zero-shot | 95 | 0.937 | 0.937 | 0.937 | 0.937 | 0.972 | 0.956 | 0.958 | 0.922–0.986 | 0 | 0.153 |

## Paired comparison (shared items)

| Task | ModernBERT | Other | n | F1 macro ModernBERT | F1 macro other | ΔF1 | 95% CI | Verdict |
|---|---|---|---|---|---|---|---|---|
| t1_dexa_device_types | modernbert-zero-shot | gemma4-31b-mas | 60 | 0.611 | 0.638 | -0.027 | -0.112–0.054 | no significant difference |
| t1_dexa_device_types | modernbert-zero-shot | gemma4-31b | 60 | 0.611 | 0.872 | -0.261 | -0.345–-0.196 | gemma4-31b better |
| t2_homebench_validity | modernbert-finetune | gemma4-31b | 1500 | 0.972 | 0.872 | 0.100 | 0.082–0.119 | ModernBERT better |
| t3_simuhome_feasibility | modernbert-cv | gemma4-31b | 600 | 0.787 | 0.577 | 0.211 | 0.154–0.268 | ModernBERT better |
| t4_homeassistant_domain | modernbert-zero-shot | gemma4-31b | 95 | 0.958 | 1.000 | -0.042 | -0.078–-0.014 | gemma4-31b better |

### t1_dexa_device_types – gemma4:31b (MAS end to end (agents assigned a task))

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| search | 0.786 | 0.407 | 0.537 | 27 |
| fridge | 0.938 | 0.882 | 0.909 | 17 |
| display | 0.857 | 0.462 | 0.600 | 13 |
| audio | 1.000 | 0.316 | 0.480 | 19 |
| clock | 1.000 | 0.467 | 0.636 | 15 |
| thermostat | 1.000 | 0.500 | 0.667 | 20 |
| light | 1.000 | 0.391 | 0.562 | 23 |
| calendar | 1.000 | 0.550 | 0.710 | 20 |

### t1_dexa_device_types – gemma4:31b (zero-shot)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| search | 1.000 | 0.630 | 0.773 | 27 |
| fridge | 0.850 | 1.000 | 0.919 | 17 |
| display | 0.812 | 1.000 | 0.897 | 13 |
| audio | 1.000 | 0.737 | 0.848 | 19 |
| clock | 0.933 | 0.933 | 0.933 | 15 |
| thermostat | 0.923 | 0.600 | 0.727 | 20 |
| light | 1.000 | 0.826 | 0.905 | 23 |
| calendar | 1.000 | 0.950 | 0.974 | 20 |

### t1_dexa_device_types – MoritzLaurer/ModernBERT-large-zeroshot-v2.0 (zero-shot)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| search | 1.000 | 0.037 | 0.071 | 27 |
| fridge | 0.765 | 0.765 | 0.765 | 17 |
| display | 0.857 | 0.462 | 0.600 | 13 |
| audio | 1.000 | 0.421 | 0.593 | 19 |
| clock | 0.833 | 0.667 | 0.741 | 15 |
| thermostat | 0.818 | 0.450 | 0.581 | 20 |
| light | 1.000 | 0.478 | 0.647 | 23 |
| calendar | 1.000 | 0.800 | 0.889 | 20 |

### t2_homebench_validity – gemma4:31b (zero-shot)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.755 | 0.978 | 0.852 | 500 |
| unexist_device | 1.000 | 0.940 | 0.969 | 500 |
| unexist_attribute | 0.916 | 0.700 | 0.794 | 500 |

### t2_homebench_validity – answerdotai/ModernBERT-large (finetune)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.997 | 0.891 | 0.941 | 6187 |
| unexist_device | 1.000 | 1.000 | 1.000 | 4194 |
| unexist_attribute | 0.791 | 0.993 | 0.880 | 2571 |

### t3_simuhome_feasibility – gemma4:31b (zero-shot)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| feasible | 0.567 | 0.780 | 0.656 | 300 |
| infeasible | 0.647 | 0.403 | 0.497 | 300 |

### t3_simuhome_feasibility – answerdotai/ModernBERT-large (cv)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| feasible | 0.752 | 0.860 | 0.802 | 300 |
| infeasible | 0.837 | 0.717 | 0.772 | 300 |

### t4_homeassistant_domain – gemma4:31b (zero-shot)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| cover | 1.000 | 1.000 | 1.000 | 11 |
| fan | 1.000 | 1.000 | 1.000 | 6 |
| light | 1.000 | 1.000 | 1.000 | 13 |
| lock | 1.000 | 1.000 | 1.000 | 9 |
| media-player | 1.000 | 1.000 | 1.000 | 21 |
| todo | 1.000 | 1.000 | 1.000 | 9 |
| vacuum | 1.000 | 1.000 | 1.000 | 17 |
| valve | 1.000 | 1.000 | 1.000 | 9 |

### t4_homeassistant_domain – MoritzLaurer/ModernBERT-large-zeroshot-v2.0 (zero-shot)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| cover | 1.000 | 1.000 | 1.000 | 11 |
| fan | 1.000 | 1.000 | 1.000 | 6 |
| light | 1.000 | 1.000 | 1.000 | 13 |
| lock | 1.000 | 1.000 | 1.000 | 9 |
| media-player | 0.778 | 1.000 | 0.875 | 21 |
| todo | 1.000 | 1.000 | 1.000 | 9 |
| vacuum | 1.000 | 0.647 | 0.786 | 17 |
| valve | 1.000 | 1.000 | 1.000 | 9 |

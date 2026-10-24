# Classification: Gemma 4 vs ModernBERT

| Task | Model | Mode | n | Acc. | P micro | R micro | F1 micro | P macro | R macro | F1 macro | F1 macro 95% CI | Problems | Latency/item (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t1_dexa_device_types | gemma4:31b | MAS end to end (agents assigned a task) | 100 | 0.250 | 0.955 | 0.443 | 0.605 | 0.956 | 0.452 | 0.598 | 0.526–0.660 | 9 | 43.274 |
| t1_dexa_device_types | gemma4:31b | zero-shot | 100 | 0.640 | 0.965 | 0.857 | 0.908 | 0.962 | 0.870 | 0.909 | 0.879–0.935 | 0 | 1.792 |
| t1_dexa_device_types | MoritzLaurer/ModernBERT-large-zeroshot-v2.0 | zero-shot | 100 | 0.300 | 0.935 | 0.547 | 0.690 | 0.884 | 0.574 | 0.670 | 0.616–0.718 | 0 | 0.192 |
| t2_homebench_validity | gemma4:31b | zero-shot | 1500 | 0.873 | 0.873 | 0.873 | 0.873 | 0.890 | 0.873 | 0.872 | 0.856–0.888 | 0 | 8.657 |
| t2_homebench_validity | answerdotai/ModernBERT-large | finetune | 12952 | 0.946 | 0.946 | 0.946 | 0.946 | 0.929 | 0.961 | 0.940 | 0.936–0.944 | 0 | 0.026 |
| t3_simuhome_feasibility | gemma4:31b | zero-shot | 600 | 0.592 | 0.592 | 0.592 | 0.592 | 0.607 | 0.592 | 0.577 | 0.537–0.616 | 0 | 4.261 |
| t3_simuhome_feasibility | answerdotai/ModernBERT-large | cv | 600 | 0.788 | 0.788 | 0.788 | 0.788 | 0.794 | 0.788 | 0.787 | 0.755–0.820 | 0 | 0.010 |
| t4_homeassistant_domain | gemma4:31b | zero-shot | 95 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000–1.000 | 0 | 1.400 |
| t4_homeassistant_domain | MoritzLaurer/ModernBERT-large-zeroshot-v2.0 | zero-shot | 95 | 0.937 | 0.937 | 0.937 | 0.937 | 0.972 | 0.956 | 0.958 | 0.922–0.986 | 0 | 0.153 |

## Paired comparison (shared items)

| Task | ModernBERT | Other | n | F1 macro ModernBERT | F1 macro other | ΔF1 | 95% CI | Verdict |
|---|---|---|---|---|---|---|---|---|
| t1_dexa_device_types | modernbert-zero-shot | gemma4-31b-mas | 100 | 0.670 | 0.598 | 0.072 | 0.005–0.143 | ModernBERT better |
| t1_dexa_device_types | modernbert-zero-shot | gemma4-31b | 100 | 0.670 | 0.909 | -0.239 | -0.283–-0.200 | gemma4-31b better |
| t2_homebench_validity | modernbert-finetune | gemma4-31b | 1500 | 0.972 | 0.872 | 0.100 | 0.082–0.119 | ModernBERT better |
| t3_simuhome_feasibility | modernbert-cv | gemma4-31b | 600 | 0.787 | 0.577 | 0.211 | 0.154–0.268 | ModernBERT better |
| t4_homeassistant_domain | modernbert-zero-shot | gemma4-31b | 95 | 0.958 | 1.000 | -0.042 | -0.078–-0.014 | gemma4-31b better |

### t1_dexa_device_types – gemma4:31b (MAS end to end (agents assigned a task))

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| search | 0.857 | 0.391 | 0.537 | 46 |
| fridge | 0.955 | 0.840 | 0.894 | 25 |
| display | 0.917 | 0.367 | 0.524 | 30 |
| audio | 1.000 | 0.286 | 0.444 | 35 |
| clock | 0.917 | 0.367 | 0.524 | 30 |
| thermostat | 1.000 | 0.342 | 0.510 | 38 |
| light | 1.000 | 0.591 | 0.743 | 44 |
| calendar | 1.000 | 0.436 | 0.607 | 39 |

### t1_dexa_device_types – gemma4:31b (zero-shot)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| search | 1.000 | 0.696 | 0.821 | 46 |
| fridge | 0.893 | 1.000 | 0.943 | 25 |
| display | 0.903 | 0.933 | 0.918 | 30 |
| audio | 0.967 | 0.829 | 0.892 | 35 |
| clock | 0.966 | 0.933 | 0.949 | 30 |
| thermostat | 0.966 | 0.737 | 0.836 | 38 |
| light | 1.000 | 0.886 | 0.940 | 44 |
| calendar | 1.000 | 0.949 | 0.974 | 39 |

### t1_dexa_device_types – MoritzLaurer/ModernBERT-large-zeroshot-v2.0 (zero-shot)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| search | 0.500 | 0.022 | 0.042 | 46 |
| fridge | 0.833 | 0.800 | 0.816 | 25 |
| display | 0.947 | 0.600 | 0.735 | 30 |
| audio | 1.000 | 0.429 | 0.600 | 35 |
| clock | 0.875 | 0.700 | 0.778 | 30 |
| thermostat | 0.920 | 0.605 | 0.730 | 38 |
| light | 1.000 | 0.614 | 0.761 | 44 |
| calendar | 1.000 | 0.821 | 0.901 | 39 |

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

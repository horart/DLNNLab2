# Диагностика обучения MLP

Запуск на CPU из этой папки:

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

`main.py` создаёт `results/metrics.csv` (12 запусков, эпохи 0–100), `results/gradients.csv` (A, C, D; оба seed; эпохи 0, 20, 100), `results/final_comparison.csv` (сравнительная таблица на эпохе 100), `results/summary.json` (выбор модели и test) и графики в `figures/`. Повторный запуск перезаписывает эти файлы. Готовый отчёт — `report.pdf`.

| Код | Скрытые слои | Активация | Инициализация скрытых весов | Оптимизатор | LR |
|---|---:|---|---|---|---:|
| A | 6 | Sigmoid | Xavier normal | Adam | 0.003 |
| B | 6 | Tanh | Xavier normal | Adam | 0.003 |
| C | 6 | ReLU | Xavier normal | Adam | 0.003 |
| D | 6 | ReLU | He normal | Adam | 0.003 |
| E | 6 | ReLU | He normal | SGD, momentum=0 | 0.05 |
| F | 1 | ReLU | He normal | Adam | 0.003 |

Данные, разбиение, seed и параметры обучения заданы в `main.py`. Реально использованное устройство: **cpu**.

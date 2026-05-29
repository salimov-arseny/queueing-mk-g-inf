# Выходящий поток в СМО $M^k/G/\infty$

Анализ и численная реализация для системы массового обслуживания $M^k/G/\infty$ с групповым поступлением **разнородных** и **зависимых** требований. Развитие классической работы Г.И. Фалина (1994) с фокусом на ранее не исследованный выходящий поток $D(t)$.

> Салимов А.Е. *Выходящий поток в системе массового обслуживания с бесконечным числом каналов при групповом поступлении разнородных зависимых требований.* МГУ им. М.В. Ломоносова, механико-математический факультет, кафедра теории вероятностей, 2026. Научный руководитель — проф., д.ф.-м.н. Г.И. Фалин.

> **🏅 Доклад по этой работе был отмечен грамотой на студенческой конференции.** Скан грамоты — [`awards/conference_certificate.pdf`](awards/conference_certificate.pdf).

PDF курсовой и исходник LaTeX — в [`paper/`](paper/).

## Постановка задачи

Рассматривается СМО с пуассоновским входящим потоком групп интенсивности $\lambda$, фиксированным размером группы $k$ (типы требований различны), бесконечным числом приборов и в общем случае **зависимыми** временами обслуживания внутри группы. Вектор $\tau = (\tau_1, \ldots, \tau_k)$ имеет произвольное совместное распределение.

Изучается выходящий процесс $D(t) = (D_1(t), \ldots, D_k(t))$, где $D_i(t)$ — число обслуженных требований $i$-го типа к моменту $t$.

## Основной результат (теорема о декомпозиции)

$$D(t) = B \cdot \delta(t),$$

где $\delta(t) = (\delta_y(t))_{y \subseteq S}$ — вектор размера $2^k$ с **независимыми** пуассоновскими компонентами,

$$\delta_y(t) \sim \mathrm{Pois}(\lambda g_y(t)),\qquad g_y(t) = \int_0^t \mathbb{P}\big(\{\tau_i \le u\}_{i \in y} \cap \{\tau_i > u\}_{i \notin y}\big)\,du,$$

а $B \in \{0, 1\}^{k \times 2^k}$ — матрица инцидентности $b_{iy} = \mathbb{I}(i \in y)$.

Эта декомпозиция сводит анализ многомерного зависимого процесса к независимым одномерным компонентам и заменяет громоздкий аппарат производящих функций из работ Абольникова (1968) и Choi–Park (1992) на прозрачный вероятностный аргумент.

## Ковариационный анализ

**Ковариация $N(t)$ и $D(t)$:**

$$\mathrm{Cov}(N_i(t), D_j(t)) = \lambda \sum_{\substack{y \subseteq S \\ i \in y,\, j \notin y}} g_y(t).$$

Диагональные элементы равны нулю — более того, $N_i(t)$ и $D_i(t)$ **независимы**. Внедиагональные строго положительны.

**Ковариационная функция $D(t)$** (теорема, доказанная через интервальное разбиение пуассоновского процесса):

$$\mathrm{Cov}(D_i(t_1), D_j(t_2)) = \lambda \int_0^{t_1} \mathbb{P}(u + \tau_i \le t_1,\; u + \tau_j \le t_2)\,du.$$

## Явные формулы для случая FGM ($k = 2$)

Времена обслуживания $\tau_1, \tau_2$ имеют распределение Фарли–Гумбеля–Моргенштерна с маргиналами $\mathrm{Exp}(1)$:

$$F(t_1, t_2) = (1 - e^{-t_1})(1 - e^{-t_2})(1 + \alpha e^{-t_1 - t_2}),\qquad \alpha \in [-1, 1].$$

Параметр $\alpha$ управляет внутригрупповой зависимостью: при $\alpha = 0$ — независимость, $\alpha > 0$ — положительная корреляция, $\alpha < 0$ — отрицательная.

![Линии уровня FGM-плотности](figures/fgm_density_contours.pdf)

**Функция $g_{\{1\}}(t)$:**

$$g_{\{1\}}(t) = (1 - e^{-t}) - \frac{\alpha + 1}{2}(1 - e^{-2t}) + \frac{2\alpha}{3}(1 - e^{-3t}) - \frac{\alpha}{4}(1 - e^{-4t}),$$

$$\lim_{t \to \infty} g_{\{1\}}(t) = \frac{1}{2} - \frac{\alpha}{12}.$$

![Динамика g_{1}(t)](figures/g1_plot.pdf)

**Элементы матрицы $\mathrm{Cov}(D(t_1), D(t_2))$:**

- Диагональ: $c_{ii}(t_1, t_2) = \lambda(\min(t_1, t_2) - 1 + e^{-\min(t_1, t_2)})$
- Внедиагональ — длинное выражение, реализованное в `cov_dd_offdiag_fgm` (формула (8) курсовой)

![Ковариационная матрица](figures/covariance_matrix.pdf)

## «Закон сохранения корреляции»

Главный вероятностный итог работы — асимптотическая полная корреляция выходящих потоков разных типов:

$$\lim_{t \to \infty} \rho_{12}(t) = \lim_{t \to \infty} \frac{\mathrm{Cov}(D_1(t), D_2(t))}{\sqrt{\mathrm{Var}(D_1(t))\,\mathrm{Var}(D_2(t))}} = 1.$$

Параметр $\alpha$ — лишь локальная аддитивная поправка на конечных временах. На бесконечности **сам факт группового поступления** порождает полную синхронизацию потоков обслуженных требований, независимо от внутренней корреляции времён обслуживания.

## Структура репозитория

```
queueing-mk-g-inf/
├── src/queue_mkginf.py       # FGM, g_y(t), ковариационные ядра, симулятор
├── tests/test_queue.py       # pytest: предельные значения, проверка симуляцией
├── paper/                    # курсовая (PDF + LaTeX)
├── figures/                  # графики из курсовой (FGM, g_{1}, ковариация)
├── awards/                   # грамота с конференции
├── requirements.txt
└── LICENSE
```

## Установка

```bash
git clone https://github.com/salimov-arseny/queueing-mk-g-inf.git
cd queueing-mk-g-inf
pip install -r requirements.txt
pytest tests/ -v
```

## Пример использования

```python
import numpy as np
from src.queue_mkginf import (
    g_single_fgm, g_single_fgm_limit,
    cov_dd_fgm, correlation_dd_fgm,
    simulate_mk_g_inf, fgm_service_sampler,
)

print(g_single_fgm(np.array(100.0), alpha=0.5))   # ≈ 1/2 - 0.5/12
print(g_single_fgm_limit(alpha=0.5))              # точно 1/2 - 0.5/12

print(cov_dd_fgm(t1=2.0, t2=3.0, alpha=0.5))
print(correlation_dd_fgm(t=100.0, alpha=0.5))     # ≈ 1.0

result = simulate_mk_g_inf(
    lam=1.0, horizon=5.0,
    grid=np.array([1.0, 2.0, 5.0]),
    service_sampler=fgm_service_sampler(alpha=0.0),
    n_replicas=2000,
)
```

## Литература

Полный список (10 источников) — в `paper/thesis.tex`. Ключевые работы:

- Falin G. (1994). *The $M^k/G/\infty$ batch arrival queue by heterogeneous dependent demands.* J. Appl. Prob., 31(3), 841–846.
- Daw A., Fralix B., Pender J. (2022). *Non-stationary queues with batch arrivals.* arXiv:2008.00625.
- Pang G., Whitt W. (2012). *Infinite-server queues with batch arrivals and dependent service times.* Probab. Eng. Inf. Sci., 26(2).
- Burke P.J. (1956). *The output of a queuing system.* Operations Research, 4(6).
- Gumbel E.J. (1960). *Bivariate exponential distributions.* JASA, 55(292).

## Лицензия

[MIT](LICENSE)

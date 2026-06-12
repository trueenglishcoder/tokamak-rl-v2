# План презентации: RL-управление магнитной конфигурацией токамака

Формат: доклад меньше 15 минут.  
Рекомендуемая длительность: 12-13 минут + 2 минуты на вопросы.  
Структура: 13 основных слайдов + 2 резервных.

Главная мысль доклада:

> Мы построили end-to-end pipeline для обучения RL-контроллера магнитной конфигурации в `tokamak-sim`. Первый полный эксперимент показал реальное обучение управления `Ip`, но политика пока отвергнута safety/shape gates. Это не финальный контроллер, а важный результат: теперь у нас есть проверяемая процедура, которая отличает физически полезное управление от небезопасного reward hacking.

## Тайминг

| Блок | Слайды | Время |
|---|---:|---:|
| Задача и мотивация | 1-3 | 3 мин |
| Архитектура и метод | 4-8 | 5 мин |
| Результаты первого run | 9-11 | 3 мин |
| Следующие шаги | 12-13 | 2 мин |

---

## Слайд 1. Название и цель проекта

### Текст на слайде

**Обучение управляющей политики для магнитного контроля токамака**

**Цель проекта:** получить reinforcement learning policy, которая управляет активными катушками в замкнутом контуре симулятора `tokamak-sim`.

Что должна делать политика:

- удерживать плазменную границу около заданной формы;
- отслеживать траекторию тока плазмы `Ip`;
- не нарушать ограничения по токам катушек;
- экспортироваться как обычный контроллер, а не оставаться только PyTorch-моделью.

`tokamak-sim + tokamak-rl-v2`

### Что должно быть на слайде

- Схематичный рисунок токамака: плазма, катушки, контур управления.
- Название крупно.
- Справа или снизу: 4 пункта "что должна делать политика".

### Что сказать устно

Сразу задать правильный тон: проект не про красивую кривую reward, а про полный путь от симулятора до проверяемого контроллера. В конце обучения должна быть не просто модель, а policy bundle, который можно загрузить в `tokamak-sim` и проверить в closed loop.

---

## Слайд 2. Почему задача сложная

### Текст на слайде

**Магнитное управление плазмой - это задача управления с жесткими физическими ограничениями**

Контроллер должен одновременно учитывать:

- нелинейную динамику плазмы и катушек;
- ограничения по токам и производным токов;
- реконструкцию плазменной границы;
- целевые значения формы и `Ip`;
- последствия текущего действия на будущие шаги.

**Ключевая сложность:** улучшение одной метрики может ухудшить другую. Например, можно лучше отслеживать `Ip`, но при этом нарушить current limits или потерять надежное определение boundary.

### Что должно быть на слайде

Слева схема:

```text
state + target
      ↓
RL policy
      ↓
coil-current derivative commands
      ↓
tokamak simulator
      ↓
new state
```

Справа таблица:

| Нужно улучшить | Нельзя сломать |
|---|---|
| `Ip tracking` | current limits |
| boundary shape | boundary found |
| reward | physical safety |

### Что сказать устно

Объяснить, что задача не сводится к минимизации одной ошибки. Контроллер должен балансировать несколько физических требований. Поэтому в проекте важны отдельные validation gates, а не только суммарная награда.

---

## Слайд 3. Архитектура проекта

### Текст на слайде

**Проект состоит из двух связанных частей**

`tokamak-sim`

- физическая модель плазмы и катушек;
- поиск плазменной границы;
- интерфейс для контроллеров;
- closed-loop rollout для проверки policy.

`tokamak-rl-v2`

- batched RL environment;
- генерация reference trajectories;
- reward calculation;
- MPO training;
- export deterministic actor;
- validation pipeline.

**Итоговый артефакт:** controller bundle, который загружается в `LearnedMagneticController`.

### Что должно быть на слайде

Большая блок-схема:

```text
tokamak-sim
plant dynamics + boundary finder
          ↓
tokamak-rl-v2 environment
observations + reward + replay
          ↓
MPO actor-critic training
          ↓
exported actor bundle
          ↓
LearnedMagneticController
          ↓
closed-loop validation
```

### Что сказать устно

Подчеркнуть, что это end-to-end система. Если policy обучилась, но не может загрузиться в симулятор как контроллер, результат не считается готовым.

---

## Слайд 4. Связь с TCV-подходом

### Текст на слайде

**Ориентир: TCV-style reinforcement learning control**

В проекте используется похожая обучающая структура:

- маленький actor для исполнения после обучения;
- critic используется только во время training;
- actor-critic optimization через MPO;
- много параллельных копий симулятора;
- observation содержит target references;
- после обучения экспортируется deterministic mean actor.

**Ограничение:** это не полная аппаратная копия TCV. В текущем симуляторе action - это нормированные производные токов катушек, а не реальные напряжения.

### Что должно быть на слайде

Слева:

```text
training:
actor + recurrent critic + MPO

deployment:
deterministic actor only
```

Справа:

```text
TCV-inspired structure
adapted to tokamak-sim
```

### Что сказать устно

Сказать аккуратно: мы переносим не всю физическую постановку TCV, а структуру обучения. Главная идея - маленький actor, который после обучения можно исполнять как обычный контроллер.

---

## Слайд 5. Почему нужен fixed objective

### Текст на слайде

**Проблема раннего этапа: базовая задача еще не была надежно обучаемой**

Что ломалось:

- target boundary мог не совпадать с реальной границей после reset;
- прежний reward был заменен на прямую физическую cost function;
- часть политик почти не использовала управление;
- `Q`-значения плохо различали sampled actions;
- высокий return не гарантировал улучшение физических метрик.

**Вывод:** сначала нужно доказать, что одна фиксированная физическая цель учится и проходит validation gates.

### Что должно быть на слайде

Таблица:

| Старый подход | Что происходило |
|---|---|
| частая смена reward weights | много вариантов неучебной задачи |
| sampled boundary target | mismatch на reset |
| слабый физический сигнал | critic плохо ранжирует действия |
| выбор по return | физика могла не улучшаться |

Внизу:

```text
Fixed physical objective first.
```

### Что сказать устно

Сказать, что на этом этапе нельзя постоянно менять objective и надеяться, что один вариант случайно даст контроллер. Если сама постановка содержит mismatch или слабый сигнал, смена весов reward только быстрее перебирает способы неудачи.

---

## Слайд 6. Исправление reference: hold_reset_boundary

### Текст на слайде

**Ключевое исправление: target boundary берется из реального reset**

Новый режим:

`hold_reset_boundary`

Алгоритм reset:

1. Сначала reset симулятора.
2. Boundary finder находит фактическую границу плазмы.
3. Эта граница становится статической целью на весь episode.
4. Параллельно задается segmented trajectory для `Ip`.

Sanity check:

```text
reset boundary error = 0.0 m
boundary_found = 1.0
```

### Что должно быть на слайде

Сравнение:

```text
Раньше:
sampled target boundary != actual reset plasma boundary

Теперь:
target boundary = actual reset plasma boundary
```

Можно нарисовать две границы: красную target и синюю measured. В старом варианте они не совпадают, в новом совпадают.

### Что сказать устно

Если цель неверна уже на первом шаге, агент учится компенсировать искусственную ошибку. Новый режим делает первую задачу физически понятной: удерживать фактическую начальную границу и отслеживать `Ip`.

---

## Слайд 7. Новая reward-функция

### Текст на слайде

**Dense physical reward как фиксированная objective**

Reward напрямую штрафует физические ошибки:

```text
L = w_shape * L_boundary
  + w_Ip * L_Ip
  + w_current * L_current_limit
  + w_derivative * L_derivative_usage
  + L_action_smoothness

reward = -L
```

Компоненты:

- mean/max boundary error;
- absolute `Ip` error;
- current limit violation;
- derivative usage;
- action RMS и delta-action RMS.

В диагностике логируются физические ошибки, current margin и actuator usage.

### Что должно быть на слайде

Схема:

```text
boundary error
Ip error
current safety       -> dense physical loss -> reward
derivative usage
action smoothness
```

### Что сказать устно

Dense reward нужен для первого надежного обучения. Он проще для диагностики: если политика нарушает токи, мы видим это отдельной компонентой, а не только через общий return.

---

## Слайд 8. Gated training pipeline

### Текст на слайде

**Policy принимается только если проходит физические gates**

Pipeline:

1. `reset sanity`: target boundary должен совпадать с reset boundary.
2. `no-control baseline`: измеряем поведение без управления.
3. `training`: MPO actor-critic обучение.
4. `deterministic eval`: проверяем mean actor.
5. `MPO diagnostics`: проверяем, что learning signal не вырожден.
6. `export`: сохраняем controller bundle.
7. `controller rollout`: загружаем policy в `tokamak-sim`.

Если gates не пройдены, policy rejected, даже если reward улучшился.

### Что должно быть на слайде

Горизонтальная схема:

```text
reset check -> baseline -> training -> deterministic eval -> gates -> export -> rollout
```

Список gates:

```text
boundary_found >= 0.999
current_over_limit = 0
shape_error <= 0.03 m
Ip improves vs no-control
action_rms in valid range
Q separation nonzero
```

### Что сказать устно

Это главный инженерный механизм. Мы не принимаем policy по одному return. Она должна улучшать физические метрики и быть загружаемой как контроллер.

---

## Слайд 9. Первый полный Stage 1 эксперимент

### Текст на слайде

**Stage 1 setup**

Задача:

- удерживать фактическую reset boundary;
- отслеживать segmented `Ip`;
- диапазон `Ip`: `100-160 kA`;
- длина episode: `500 steps`;
- actor outputs: normalized coil-current derivatives.

Training:

- distributed MPO training;
- 7 actor workers + 1 learner GPU;
- recurrent Q critic;
- feedforward actor;
- deterministic actor eval after training.

### Что должно быть на слайде

Таблица:

| Параметр | Значение |
|---|---|
| Reference mode | `hold_reset_boundary` |
| Ip curriculum | `100-160 kA` |
| Episode length | `500 steps` |
| Algorithm | MPO |
| Actor | feedforward Gaussian |
| Critic | recurrent Q critic |

### Что сказать устно

Stage 1 - самый простой curriculum. Его цель не финальное управление, а безопасно доказать обучаемость. Если Stage 1 не проходит gates, Stage 2 запускать рано.

---

## Слайд 10. Результат: что улучшилось

### Текст на слайде

**Первый run показал реальное обучение**

| Метрика | No control | RL actor |
|---|---:|---:|
| `Ip error` | `28.5 kA` | `9.1 kA` |
| `mean boundary error` | `5.2 cm` | `3.6 cm` |
| `action RMS` | `0.000` | `0.222` |
| `sampled_q_spread` | - | `1.22` |
| `policy_weight_max` | uniform `0.05` | `0.099` |

Интерпретация:

- actor начал использовать управление;
- critic различает sampled actions;
- MPO не застрял в uniform policy weights;
- `Ip tracking` существенно улучшился.

### Что должно быть на слайде

- Таблица метрик.
- Один график из W&B: `sampled_q_spread` или `policy_weight_max`.
- Зеленым выделить `Ip error: 28.5 -> 9.1 kA`.

### Что сказать устно

Это главный положительный результат. Раньше было непонятно, учится ли стек вообще. Теперь видно: learning signal есть, actor не нулевой, MPO не вырожден.

---

## Слайд 11. Почему политика не принята

### Текст на слайде

**Policy rejected: улучшение Ip было куплено ценой safety/shape**

Failed gates:

| Gate | Требование | Получено |
|---|---:|---:|
| `boundary_found` | `>= 0.999` | `0.988` |
| `current_over_limit` | `0 A` | `4195 A` |
| `shape_error_mean` | `<= 0.03 m` | `0.036 m` |
| controller rollout | `ok` | schema mismatch |

Вывод:

**Policy learned control, but is too aggressive. It is not deployable.**

### Что должно быть на слайде

Таблица failed gates.  
Под таблицей:

```text
Ip improved, but safety failed.
```

### Что сказать устно

Объяснить, что это не неожиданная катастрофа, а нормальная функция pipeline. Без gates можно было бы принять policy только из-за улучшения Ip, хотя физически она небезопасна.

---

## Слайд 12. Что меняем сейчас

### Текст на слайде

**Следующий Stage 1 run: safer objective**

Изменения reward:

| Параметр | Было | Стало |
|---|---:|---:|
| `shape_weight` | `4` | `8` |
| `ip_weight` | `2` | `1` |
| `current_weight` | `1` | `25` |
| `current_bad_a` | `50000` | `5000` |
| `terminal_reward` | `-5` | `-20` |

Также:

- обновляется `tokamak-sim`, чтобы `LearnedMagneticController` поддерживал новый `joint_state_v1` export;
- Stage 2 откладывается до прохождения Stage 1 gates.

Цель следующего run:

**сохранить Ip improvement, убрать current violations, улучшить boundary reliability.**

### Что должно быть на слайде

Схема итерации:

```text
Run 1: learned Ip control, failed safety
        ↓
Diagnosis
        ↓
Safer reward
        ↓
Run 2 Stage 1
```

### Что сказать устно

Это не ручное угадывание reward. Изменения следуют из физической диагностики: current violations слишком дешевые, shape недостаточно важен, Ip pressure слишком сильный.

---

## Слайд 13. Заключение

### Текст на слайде

**Итоги**

- Построен end-to-end RL pipeline для магнитного управления в `tokamak-sim`.
- Исправлена ключевая ошибка постановки: target boundary теперь совпадает с реальной boundary на reset.
- Основной путь обучения сведен к фиксированной dense physical objective.
- Первый Stage 1 run показал настоящее обучение: `Ip error` снизился с `28.5 kA` до `9.1 kA`.
- Policy пока rejected: shape/current safety gates не пройдены.
- Следующий шаг - safer Stage 1 rerun и только потом Stage 2/3 curriculum.

Финальная строка:

**Мы получили не финальный контроллер, а проверяемый pipeline, который отличает физически полезное управление от небезопасного reward hacking.**

### Что должно быть на слайде

Слева: 5-6 итоговых bullet points.  
Справа маленькая схема:

```text
learn -> validate -> reject unsafe -> improve objective
```

### Что сказать устно

Закончить честно: готовой production policy пока нет. Но теперь есть система, которая показывает, что именно учится, что ломается, и почему policy нельзя принимать.

---

## Резервный слайд A. Что значит "policy in hand"

### Текст на слайде

**Policy считается готовой только после export + validation**

Нужные артефакты:

- `best.pt`;
- `policy_weights.npz`;
- `controller_schema.json`;
- `normalization.json`;
- `metadata.json`;
- `policy_validation.json`;
- successful closed-loop rollout через `LearnedMagneticController`.

Checkpoint сам по себе не является готовым контроллером.

### Что должно быть на слайде

Checklist файлов.

### Что сказать устно

Если спросят, что значит "политика готова": готова не тогда, когда есть PyTorch checkpoint, а когда она загружается как контроллер и проходит closed-loop проверку.

---

## Резервный слайд B. Почему низкая GPU utilization

### Текст на слайде

**Почему A100 не загружены как в обычном deep learning**

Текущий training loop содержит:

- много маленьких simulator kernels;
- boundary finding на каждом шаге;
- multiprocessing actor workers;
- CPU/GPU transfer через replay;
- небольшие neural network updates по сравнению с размером GPU.

Это означает:

```text
training progresses, but is orchestration/simulator-bound,
not dense-matmul-bound.
```

### Что должно быть на слайде

Схема bottleneck:

```text
actor workers -> small GPU sim steps -> CPU queue -> learner updates
```

### Что сказать устно

Низкая GPU utilization - инженерная проблема производительности, но не главный научный блокер текущей итерации. Сначала нужно пройти Stage 1 gates, потом оптимизировать throughput.

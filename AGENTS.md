# AGENTS.md

This file defines mandatory architecture and coding rules for AI coding agents.

Agents must follow these rules unless explicitly instructed otherwise.

---

# General Philosophy

- Code must be readable, explicit, and maintainable.
- Prefer simplicity over clever abstractions.
- Follow **moderate SOLID principles**.
- Prefer composition over inheritance.
- Business architecture must remain clear from the directory structure.
- Reflect changes made in README and docs, keep it clean descriptive of the current state of repo, dont do "version control" in readme or docs

---

# Mandatory Documentation (Docstrings)

Every **class, function, and method MUST contain a docstring**.

Docstrings should explain:

- purpose
- important behavior
- parameters when relevant
- return values when not obvious

Example:

```python
class ServiceSettings(BaseModel):
    """Settings of the service runtime."""
````

Rules:

* public classes must have docstrings
* public functions must have docstrings
* public methods must have docstrings

Prefer **short and clear descriptions**.

---

# Business Module Architecture

Code must be organized around **business capabilities**, not technical layers.

Example:

```
src/
    code_wiki/
    repo_monitoring/
    adapters/
    database/
    config.py
    main.py
```

Rules:

* each capability has its own package
* package names reflect the business domain
* avoid generic folders like `services` or `core`

Good:

```
src/code_wiki/
src/repo_monitoring/
```

Bad:

```
src/services/
src/core/
```

---

# Internal Module Structure

Example agent module:

```
src/code_wiki/
    ast_node/
    cluster_node/
    generation_node/
    prompts/
    utils/
    doc_generator_graph.py
    runtime_config.py
```

Rules:

* structure must reflect responsibilities
* avoid giant modules
* prefer small focused packages

---

# LangGraph Conventions

## Graph assembly

Graph orchestration must live in:

```
*_graph.py
```

Example:

```
doc_generator_graph.py
```

Responsibilities:

* state schema
* context schema
* routing logic
* node registration
* graph compilation

Graph files must stay **readable orchestration layers**.

---

## Node organization

Nodes must represent **one step of the pipeline**.

Example:

```
ast_node/
cluster_node/
generation_node/
```

Rules:

* one node = one responsibility
* move helpers to `utils`

---

# Context and State

Runtime context must use **dataclasses**.

Example:

```python
@dataclass(frozen=True, slots=True, repr=True)
class BaseCtx:
    settings: Settings
    llm_client: ChatOpenAI
```

Rules:

* context objects should be immutable
* dependencies grouped in context

For stable schemas prefer **Pydantic models**.

Bad:

```python
Mapping[str, Any]
```

Better:

```python
class ModuleTree(BaseModel):
    nodes: list[ModuleNode]
```

---

# Central Composition

Application wiring must happen in the **composition root**.

Usually:

```
src/main.py
```

Responsibilities:

* load config
* create dependencies
* initialize graphs
* connect adapters
* wire business modules

---

# Adapter Layer

Adapters connect protocols to business logic.

Example:

```
src/adapters/
    continue_adapter/
```

Adapters may include:

* HTTP routers
* webhooks
* workers
* integration bridges

Rules:

* adapters may import business modules
* business modules must not import adapters

---

# Imports

Use **absolute imports only**.

Correct:

```python
from src.code_wiki.doc_generator_graph import build_graph
```

Wrong:

```python
from ..doc_generator_graph import build_graph
```

`__init__.py` files must stay minimal.

Rules:

* do not use `__init__.py` as an export barrel
* do not re-export symbols from package internals via `__init__.py`
* import from the concrete module where the symbol is defined
* do not use nested imports inside functions or methods; imports must be module-level
* avoid quoted forward-reference annotations; import concrete types at module level when practical
* `__init__.py` may contain only a short package docstring or package marker code when truly necessary

---

# Comments and Docstrings Language

Rules:

* comments and docstrings in application code must be written in Russian
* keep English only for external protocol names, library names, literal values, or quoted third-party messages

---

# Dataclasses

Default dataclass pattern:

```python
@dataclass(frozen=True, slots=True, repr=True)
class Foo:
    pass
```

Mutable case:

```python
@dataclass(slots=True, repr=True)
class Foo2:
    pass
```

Rules:

* prefer `frozen=True`
* always `slots=True`
* always `repr=True`

---

# Typing

Prefer **collections.abc**.

```python
from collections.abc import Iterable, Mapping, Sequence
```

Built-in generics:

```
list[str]
dict[str, int]
```

are allowed only when appropriate.

Rules:

* prefer `collections.abc`
* avoid `typing.List`, `typing.Dict`
* avoid vague container types

Bad:

```
Mapping[str, Any]
```

Better:

```
Mapping[str, ModuleInfo]
```

For structured data prefer **Pydantic models**.

---

# Static Typing (mypy)

The project uses **mypy**.

Rules:

* code should pass mypy
* fix typing issues when modifying code
* avoid `Any`
* avoid `# type: ignore`

---

# Async Rules

Prefer async functions for IO-bound operations.

Rules:

* avoid blocking IO inside async code
* prefer async clients
* avoid synchronous network libraries

---

# Configuration Architecture

Use **nested configuration models**.

Example:

```python
class Settings(BaseSettings):
    service: ServiceSettings
    llm: LLMSettings
    db: DatabaseSettings
```

Rules:

* avoid flat configs
* each section has its own model
* use Field(..., description=...)

---

# Configuration Initialization

Use cached settings factory.

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()
```

Config is injected via:

```
app.state.settings
```

---

# Paths

Prefer `pathlib`.

Good:

```
Path("data/file.txt")
```

Avoid:

```
os.path
```

---

# Database and Migrations

Use **Alembic**.

Rules:

* LLM must NOT invent migration logic
* modify SQLAlchemy models only
* migrations generated via:

```
alembic revision --autogenerate
```

PostgreSQL enum types must use **explicit, domain-specific names**.

Rules:

* do not rely on auto-derived enum type names
* do not reuse or mirror the column name as the database enum type name
* enum type names must describe the business context, for example `docgen_run_status_enum`
* when PostgreSQL schema is used, enum types must be created in that schema explicitly

---

# Logging

Use module-level logger.

```
logger = logging.getLogger(__name__)
```

Use one-line logger calls with `%s` placeholders and inline `key=value` pairs.

Rules:

* keep each logger call in one physical line
* never split a `logger.*(...)` call across multiple physical lines, even when the message is long
* use printf-style placeholders (`%s`) instead of f-strings in logger arguments
* include context in the message body as `key=value` pairs
* do not use `extra=...` for business logs
* do not use context wrappers/filters/adapters for log enrichment

Example:

```python
logger.warning("Не удалось определить service_version: repo_root=%s git_dir=%s fallback=%s", resolved_root, git_dir, fallback)
```

---

# Prompt Organization

All prompts must live in:

```
prompts/
```

Rules:

* prompts stored as `.txt`
* do not inline prompts in code
* prompts must be reusable

Example:

```
prompts/
    cluster_system.txt
    generate_docs_system.txt
```

Load prompts from files.

---

# Forbidden Patterns

Avoid:

* relative imports
* global mutable state
* giant service modules
* inline prompts
* blocking IO in async
* manual alembic migrations
* excessive Any
* excessive type ignore
* unsolicited workaround logic (do only what was explicitly requested)
* suppressing warnings/errors without explicit request
* redundant env names like `DB__DB_NAME` when `DB__NAME` is sufficient
* overengineered alias chains for config fields without clear migration requirement
* nested imports inside functions or methods
* quoted type annotations when direct imports are practical

---

# Change Minimalism

Rules:

* implement only requested behavior and required technical consequences
* if a workaround is optional, ask before adding it
* prefer one canonical config name over multiple aliases
* keep naming straightforward (`db.name`, `db.schema`) unless user asks otherwise

---

# Code Style Summary

Prefer:

* business-capability modules
* dataclasses
* pathlib
* collections.abc
* pydantic schemas
* docstrings everywhere
* class-based builders for infrastructure wiring such as Redis, checkpointers, external clients, and lifecycle resources
* SOLID structure
* safe collection access: before indexing a list/array (`items[0]`), explicitly ensure it is non-empty

---

# Agent Instruction

When generating code:

1. follow this file
2. preserve architecture
3. keep modules readable
4. add docstrings
5. respect business modules
6. keep graphs readable
7. avoid inventing migrations
8. improve typing
9. prefer collections.abc
10. store prompts in `.txt`

---

# Tokamak Project Execution Discipline

These rules are mandatory for work on `tokamak-sim` and `tokamak-rl-v2`.

## Plan discipline

- Read the project-root `PROJECT_CONTEXT.md` (`../PROJECT_CONTEXT.md` from either repository) and the repository `AGENTS.md` before changing code.
- Preserve the existing `tokamak_control` repository architecture. Do not reorganize existing `core/`, `geometry/`, or other established packages merely to satisfy generic examples earlier in this file.
- Follow the user-approved implementation plan in order. Do not silently replace it with a different architecture or expand the scope.
- One implementation step must have one concrete scientific or software objective.
- Change only the minimum canonical files required by the current step.
- Do not touch unrelated modified, deleted, or untracked files.
- Do not perform opportunistic refactors, cleanup, renaming, or architecture changes outside the current step.

## Verification discipline

- Test only the direct acceptance conditions of the change being made.
- Do not create chains of checks for checks, broad audits, duplicate preflight stages, or unrelated regression sweeps unless a direct failure requires them.
- Once the direct acceptance test passes, stop the step and report the result.
- Never claim that a test, CUDA run, replay, parity check, or benchmark was performed unless it was actually executed on the exact final file being delivered.
- Separate checks that can run in the agent environment from checks that require the user's local CUDA machine.

## Anti-Zeno execution rule

- The approved implementation plan is finite. Do not recursively insert new intermediate gates after each successful gate unless a new concrete failure or newly discovered unresolved risk makes that gate necessary.
- A verification step is justified only when its failure would change the implementation, architecture, or acceptance decision before the final target can be attempted. Extra confidence by itself is not a sufficient reason.
- Once the direct component risks relevant to the current objective have passed their acceptance checks, advance to integration or end-to-end acceptance. Do not replace forward progress with another layer of micro-tests.
- Prefer one end-to-end acceptance run that simultaneously validates already-integrated parts over a sequence of separate public-path, smoke, plumbing, wrapper, artifact-path, or "check-the-check" gates.
- Do not ask the user to repeatedly apply transfer archives and run narrowly scoped tests whose only purpose is to reconfirm behavior already established by prior accepted checks.
- Do not split required consequences of one approved objective into new numbered steps merely because they can be tested independently. Implement the remaining necessary wiring together, then validate at the meaningful acceptance boundary.
- A previously written plan may be shortened when earlier results have already closed the risks that motivated later intermediate gates. Never lengthen it automatically after success.
- Exceptions require a specific reason: a concrete failure that must be isolated, a destructive or irreversible operation, a genuinely expensive run where a cheap prerequisite prevents likely waste, or a newly discovered safety/correctness risk. State that reason explicitly.
- Treat endless subdivision of implementation and verification as a development failure mode. The default after successful direct verification is forward progress toward the user's actual target.

## File delivery and source of truth

- The current canonical repository file is the source of truth. Temporary transfer files, patches, archives, and older chat attachments are not authoritative after the canonical file changes.
- For small source changes, prefer complete replacement/addition files or deterministic exact edits over hand-written `.patch` files.
- Before handing off a changed file, inspect the final saved file and show `git diff -- <changed files>` for only the files belonging to the current step.
- Before handing off a runnable script, verify every newly added import against the canonical repository tree. When the agent environment has the required dependencies, execute the direct import/startup path of the exact final file. Never invent or guess an import module path.
- Do not overwrite or delete unrelated user work.

## Scientific-model changes

- Base new physical equations on named primary literature or an existing validated tokamak simulation/control model.
- Record the exact source equation numbers and the project's sign convention in code documentation or `PROJECT_CONTEXT.md`.
- Clearly distinguish three categories: physical constants, globally fitted model parameters, and per-shot initial hidden states.
- Do not introduce arbitrary correction terms, per-shot gains, latent forcing inputs, or fallback physics merely to improve fit unless the user explicitly approves a plan change.
- Keep shot `3864` as a closed holdout for current-plant development: do not use it for fitting, model-order selection, parameter bounds, or acceptance-threshold tuning.
- Build and accept the CPU reference formulation before implementing the GPU mirror.
- The GPU implementation must use the same mathematical model as the accepted CPU reference. No CPU fallback is allowed in the GPU hot path.

## Electromagnetic unit and passive-circuit invariants

- In `tokamak-sim`, the axisymmetric `psi = R*A_phi` contract is poloidal flux per radian (`Wb/rad`). Full loop flux linkage is `Phi = 2*pi*psi`.
- A mutual-inductance matrix used in a physical circuit must therefore use the full-flux convention `M = 2*pi*mu0*G` when `G` is `tokamak_control.core.green.green_axisymmetric`. Never use raw `G` as an inductance in henries.
- Do not regularize passive self-inductance with an arbitrary coordinate epsilon. The canonical CPU passive circuit uses finite rectangular cross-section self-inductance from Landreman et al. 2023 equations (11)-(13).
- A passive inductance matrix must represent positive magnetic energy. If the chosen discretization/formulation makes the matrix non-positive, fix the physical discretization/formulation instead of hiding the issue with a pseudo-inverse or diagonal fudge factor.
- The legacy `tokamak_control/core/wall_model.py` is not the scientific reference for the new current plant and must not be copied into the new model unchanged.
- Never silently zero wall currents after a non-finite circuit result. A non-finite result in the new canonical passive circuit is an explicit failure.
- Use Romero equation (16) for resistive voltage: `V_R = R_p * (I_p - I_nonind)`. For the current T-15MD dataset keep `I_nonind = 0` unless an independent current-drive trace is explicitly added.
- Do not infer or fit `R_p` directly from `I_p` and coil-current traces alone. In the Romero formulation `R_p` is entangled with `L_i` and equilibrium voltage; a standalone resistance fit requires an independently defined `V_R` trace or the accepted joint transformer-state identification.
- Do not invent a synthetic `V_R` target from the legacy first-order plant.

## Boundary and oracle invariants

- `boundary_found=False` is a hard physical-pipeline failure, not a reason to silently reject an oracle window.
- Invalid fixed-angle projection is a hard failure for the current 32-radius RL contract, not a silent dataset filter.
- Dense LCFS remains the physical geometry. The 32 radii remain only a derived RL representation.

## PROJECT_CONTEXT maintenance

- Update `PROJECT_CONTEXT.md` after every completed implementation step before starting the next one.
- Each update must record: the current architecture, files changed, equations/assumptions introduced, tests actually run and their results, unresolved limitations, and the single next planned step.
- When current state changes, update or mark obsolete older active-status text so the document does not contain contradictory active instructions.
- Keep historical incidents as history, but do not let historical status override the current accepted implementation.


## Current-plant identification invariants

- T-15 CSV `Ip` in the active calibrated dataset is stored as a positive magnitude, while `T15MD.toml` uses `plasma_psi_sign = -1`. The signed current state used by Romero equations must therefore be `I_transformer = plasma_psi_sign * Ip_dataset`. Never combine positive-magnitude `Ip` with boundary flux of the opposite electromagnetic sign.
- A boundary-flux trace reconstructed from measured `Ip` and measured coil currents is an identification input only. Its fit error must never be reported as independent closed-forward validation because measured `Ip` participated in constructing that input.
- Warm the passive-vessel state with the available 50 ms untrimmed T-15 prehistory before the trimmed identification interval. Do not fit one free initial current per passive filament.
- The current passive-vessel identification reference uses a 5 mm AISI 321 shell and 20 °C resistivity `0.74e-6 Ohm*m` only as a documented reference model. T-15MD literature reports both 5 mm and 8 mm shell regions, so one global positive wall-resistance scale is fitted to effective sheet resistance. Do not describe the reference thickness as the exact spatial T-15MD wall map.
- Generalized eigen coordinates of the passive circuit may be used for computational efficiency only when every physical passive mode is retained. This exact coordinate transform is not modal reduction. Do not discard modes until the later explicit modal-reduction step.
- During system identification, the passive correction to boundary flux is evaluated on the no-wall 32-point LCFS and averaged to first order. This is an identification approximation, not the final coupled forward equilibrium. Production forward validation must later recompute equilibrium with the total active + plasma + passive field.

- A successful optimizer exit and a low train error do not by themselves establish a usable physical current plant. If global physical parameters land on optimizer bounds, do not widen those bounds or open holdout shot `3864`; execute the already-planned local Jacobian/SVD identifiability step first and decide parameter reduction from that result.

## Identifiability-driven current-plant reduction

- The completed 26-parameter local sensitivity analysis is the decision point for the first reduction of the Romero fit. Its result must not be replaced by wider bounds or a new physics term.
- `T_R` (`RomeroTransformerParameters.resistive_zero_time`) is fixed to exactly zero in the reduced fit. It was the dominant least-observable singular direction, was strongly correlated with `T_B` and `boundary_weight`, and the published damped TCV fits in Romero et al. 2012 Table 3 also identified `T_R = 0`.
- `skin_state_2_initial` is not a free per-shot identification parameter in the reduced fit. The previous sensitivities of `Li0` and `skin_state_2_initial` were nearly collinear for every train shot. Keep physical `Li0` free and derive the auxiliary canonical skin state from equation (58) with the explicit initialization closure `dV_C/dt(t0) = 0`.
- The reduced fit therefore has 5 free global parameters (`omega`, `damping`, `boundary_weight`, `T_B`, wall resistance scale) and 4 free parameters per train shot (`Li0`, `Vc0`, `Rp0`, `Rp1`), for 21 free variables across the four train shots.
- Do not widen the current `omega`, `T_B`, or wall-resistance bounds in the same reduction step. First run the reduced fit with the existing bounds and inspect only its direct fit result.
- If the reduced fit again lands `omega`, `T_B`, or wall resistance scale on a bound, stop at that result. Do not immediately widen the bound or add another latent degree of freedom.
- Do not interpret or accept fitted physical parameters when `least_squares` stops only because `max_nfev` was exhausted (`success=False` at the configured evaluation limit). Complete the same fit with a larger evaluation budget first, without changing bounds or physics in that continuation.

- Увеличение `max_nfev` само по себе не является continuation least-squares fit: новый вызов `least_squares` снова стартует из переданного initial vector. Если задача формулируется как продолжение уже исчерпавшего бюджет fit, следующий запуск обязан явно загрузить сохранённую конечную точку как initial seed. Не называть повторный запуск continuation, если seed фактически не передан.

## Long-running local jobs

- Любой локальный fit, replay, benchmark или другой пользовательский запуск, который ожидаемо может длиться дольше примерно одной минуты, до передачи пользователю обязан иметь видимый runtime progress в stdout. Progress должен использовать точные счётчики самого runtime/optimizer; elapsed heartbeat допускается отдельно. ETA выводится только когда его можно получить без изменения семантики вычисления.
- Нельзя просить пользователя запускать многочасовой optimizer полностью вслепую. В локальной SciPy без `least_squares(callback=...)` использовать `verbose=2` как точный источник `Iteration/Total nfev/Cost/Optimality` и отдельный elapsed heartbeat. Не строить процент или ETA из raw residual-call count, потому что numerical-Jacobian calls не соответствуют SciPy `nfev` один-к-одному.
- Уже запущенному Python-процессу нельзя обещать ретроактивно добавить progress reporting. Не советовать прерывать длительный fit только ради индикатора, если промежуточное optimizer state не checkpoint-ится.
- Использование endpoint fit-а, остановленного по `max_nfev`, как точки для локального Jacobian/SVD анализа не означает принятие этих параметров как физически идентифицированных. Такой endpoint разрешён только для решения уже запланированного вопроса о сокращении parameterization.


## 2026-08-09 first-order flux-diffusion decision

- The 21-parameter second-order fit is not the canonical current-plant model after the reduced identifiability result. Its local Jacobian had full numerical rank but condition number `46272.8`; the least-observable singular mode was dominated by `omega_skin` (`+0.993467`) with `T_B` (`+0.113895`), while `T_B` remained on its upper bound. Do not keep fixing arbitrary second-order coefficients merely to preserve model order.
- The available T-15 shots are ordinary discharge trajectories, not dedicated fast current-modulation identification experiments. For the current development stage use the first-order Romero flux-diffusion approximation from Romero 2010 equations (41), (43)-(45), which was validated on JET, and fit only its global gain `k` and time constant `tau` together with the existing wall-resistance scale.
- Do not copy numerical `omega`, `T_B`, damping, `k`, or other TCV parameters from Romero 2012 into T-15MD as fixed constants. Literature values may motivate model structure, not substitute for T-15MD identification.
- Historical first-order stage used per-shot `Li0`, `Vc0`, `Rp0`, `Rp1`; this rule is superseded by the completed 8-shot identifiability decision below, which removes free `Vc0` using Romero equation (40).
- The second-order implementation in `romero_transformer.py` may remain as a literature/reference implementation, but the active CPU current plant and active system identification use the first-order Romero state until the project obtains data capable of identifying second-order skin dynamics.
- Transfer archives must contain source/config/document files required by the step only. Never include `__pycache__`, `.pyc`, pytest cache, run outputs, or other generated artifacts.

## Current-plant identification dataset split

- Do not reuse the historical RL training split as the identification split for the new physical current plant. The four-shot RL subset was chosen partly because the legacy current model behaved acceptably there and would bias identification toward legacy-compatible regimes.
- Active current-plant identification shots are exactly `3854`, `3855`, `3856`, `3857`, `3858`, `3859`, `3862`, `3863`. One common set of global plant parameters is fitted across all eight shots.
- Shot `3864` remains the closed current-plant holdout until the CPU plant structure and global parameters are frozen.
- The RL split remains a separate contract: RL train `3856`, `3857`, `3858`, `3863`; RL holdout `3864`. Do not conflate RL data selection with plant system identification.
- Additional identification shots may introduce new per-shot hidden initial states, but must not introduce per-shot global physics gains.

## SciPy optimizer compatibility

- Do not rely on optional `scipy.optimize.least_squares` arguments unless they are part of the user's installed SciPy API. The local T-15 development environment used on 2026-08-09 does not accept the `callback` argument.
- For long `least_squares` runs in this environment use SciPy's long-standing `verbose=2` iteration table as the canonical progress output. It reports exact optimizer iteration, total `nfev`, cost, cost reduction, step norm and optimality without changing optimizer semantics.
- Do not implement a fake percentage bar from raw residual-call counts: numerical-Jacobian calls do not map one-to-one to SciPy `nfev` and such a bar would be misleading.


## 2026-08-09 first-order 8-shot fit result

- The first-order Romero 8-shot identification fit has converged on development shots `3854`, `3855`, `3856`, `3857`, `3858`, `3859`, `3862`, `3863` with `k=0.720532`, `tau=0.14451 s` and `wall_R_scale=10`.
- This endpoint is not yet an accepted physical plant because the global wall-resistance scale is exactly on its upper optimizer bound and shot `3862` remains substantially worse than the other identification shots (`MAE=10804.9 A`).
- Historical pre-SVD gate: the wall-resistance bound was not to be widened before the first-order 8-shot Jacobian/SVD analysis. That analysis is now complete; follow the explicit post-SVD decision below.
- The completed identifiability analysis used the same 8-shot split and the exact historical 35-variable parameterization (`k`, `log(tau)`, `log(wall_R_scale)` plus `Li0/Vc0/Rp0/Rp1` per shot).


## 2026-08-09 first-order 8-shot identifiability decision

- The completed 35-parameter first-order 8-shot Jacobian has full numerical rank `35/35` with condition number `18382.6`. The dominant ambiguity is per-shot initialization, not the global first-order flux-diffusion block.
- The strongest correlations are `Li0 <-> Vc0` for every development shot (`|corr|` approximately `0.992-0.9999`). The least-observable singular mode is dominated by `shot3855.Li0` and `shot3855.Vc0`. Do not keep both as independent free initial parameters.
- In the active identification parameterization, `Vc0` must be derived from Romero equation (40), `Li*dIp/dt = V_B + V_C - 2*V_R`, using signed measured initial `dIp/dt`, fitted `Li0`, fitted `Rp0`, and current `V_B0`. This is an initialization closure only; subsequent `V_C` remains a dynamic Romero state.
- The active 8-shot first-order fit therefore has exactly 27 free variables: global `k`, `log(tau)`, `log(wall_R_scale)` and per-shot `Li0`, `Rp0`, `Rp1`. Do not reintroduce free `Vc0` without new independent data that identify it.
- The previous global wall scale `10` was on its upper bound, but `global.log_wall_R_scale` contributed only `-0.000709` to the weakest singular mode. The identifiability result therefore permits one widening of the effective wall-resistance search cap from `10` to `100`. This scale is an effective correction for the axisymmetric smooth-shell reference and must not be interpreted as literal AISI 321 material resistivity.
- T-15MD vessel literature reports both 5 mm and 8 mm shell regions and 152 horizontal/vertical ports. The current axisymmetric vessel circuit does not resolve this 3D current-path topology. If the 27-variable fit also saturates `wall_R_scale=100`, do not widen the bound again; stop and revisit the passive-vessel representation.
- Keep the existing `Rp0/Rp1`, `k`, and `tau` bounds unchanged in the same reduction step. Do not reopen second-order skin dynamics.
- Shot `3864` remains closed until the 27-variable development fit is interpreted and the CPU plant is frozen.


## 2026-08-09 post-27-variable fit decision

- The converged 27-variable 8-shot first-order fit (`k=0.446867`, `tau=0.169584 s`) no longer saturates the wall-resistance correction (`wall_R_scale=2.01933`). Do not continue widening wall bounds or attribute the remaining trajectory error to wall saturation.
- CPU development-shot acceptance is still blocked by systematic errors on `3857`, `3858`, `3859`, and especially `3862`. Do not open holdout `3864`, start the GPU mirror, or build RL artifacts from this endpoint.
- Before adding any new resistance degree of freedom, run the single teacher-forced required-`R_p(t)` diagnostic using the frozen first-order transformer parameters, fitted wall-corrected boundary voltage, and measured signed `I_p(t)`. This diagnostic is identification-only and must never be described as forward validation.
- Do not automatically replace affine `R_p(t)` with a spline, high-order polynomial, neural model, or per-timestep latent input. Increase resistance-model complexity only if the required-resistance trace is positive/smooth but demonstrates systematic non-affine structure across development shots.
- If the required-resistance trace is frequently non-positive, strongly oscillatory, or otherwise physically inconsistent, do not hide that failure with a more flexible `R_p`; revisit the upstream boundary-voltage, flux-diffusion, or passive-vessel model instead.

## 2026-08-09 post-required-resistance diagnostic decision

- The step-12 teacher-forced required-`R_p(t)` plots supersede the earlier assumption that the remaining first-order error should be addressed by immediately adding a more flexible resistance law. Across all eight development shots the inferred resistance is dominated by a pronounced fast oscillatory component around a slow approximately affine trend; the fitted affine and best affine lines are nearly coincident. Do not add a spline, polynomial, thermal latent state, or per-timestep resistance forcing from this evidence.
- The previous prohibition on reopening second-order skin dynamics was based on identifiability using only the historical four-shot subset. After the identification dataset was expanded to eight shots, that model-order decision is no longer final. It is permitted to run exactly one reduced second-order Romero model-selection fit on the eight development shots before freezing the CPU plant.
- The reduced second-order model-selection fit must keep `T_R = 0`, derive `V_C(t0)` from Romero equation (40), and derive the auxiliary second skin state from equation (58) with `dV_C/dt(t0) = 0`. Per-shot free parameters remain only `Li0`, `Rp0`, and `Rp1`.
- The eight-shot reduced second-order parameterization has exactly 29 free variables: global `omega`, `damping`, `boundary_weight`, `T_B`, `wall_R_scale`, plus `Li0/Rp0/Rp1` for each of the eight development shots. Shot `3864` remains closed.
- The existing first-order v3 fit remains the active reference until the second-order model-selection fit is interpreted. Do not silently switch production CPU/GPU/RL code to second-order solely because the fit script supports it.
- Initial second-order values (`omega`, `damping`, `boundary_weight`, `T_B`) are optimizer starting values only. They are not T-15MD constants and must not be reported as measured or literature-derived T-15MD parameters.
- Do not run another resistance-shape diagnostic or an identifiability SVD before the reduced eight-shot second-order fit result is available. The next single decision is whether the physically motivated extra flux-diffusion order materially improves the eight-shot forward fit without pathological bound saturation.
## 2026-08-09 selected second-order CPU current plant and coupled runtime

- The completed reduced second-order 8-shot model-selection fit is the selected CPU current-plant structure. It converged with `cost=0.2019746456`, `omega=51.4977505054 rad/s`, `damping=4.99991918538`, `boundary_weight=0.620833702044`, `T_B=0.0659568954170 s`, `T_R=0`, and `wall_R_scale=2.78725877506` on development shots `3854`, `3855`, `3856`, `3857`, `3858`, `3859`, `3862`, `3863`.
- This decision supersedes the earlier temporary rule that first-order Romero was the active CPU structure. Stop the model-selection cycle here. Do not run another SVD, widen the damping bound, refit `omega` separately, or add a more flexible `R_p` law before coupled forward validation produces evidence requiring a structural change.
- `damping ~= 5` is an effective control-oriented fitted parameter of the reduced model, not a measured material or machine constant of T-15MD. Do not present it, `omega`, `boundary_weight`, or `T_B` as literature-derived T-15MD constants.
- Canonical CPU runtime must evolve the selected second-order Romero state together with the full passive-vessel state and the equilibrium. Runtime `V_B` must come from the LCFS boundary flux of the total active + plasma + passive field. Measured `I_p` may not be used inside the coupled forward state update.
- Runtime keeps all physical passive modes. The eight development-shot initial states use the same 50 ms passive-current warmup as identification, fitted `Li0/Rp0/Rp1`, `V_C0` derived from Romero equation (40), and the second skin state derived from equation (58) with `dV_C/dt(t0)=0`. Do not replace these states with zero wall currents, free `V_C0`, or ad hoc initialization.
- The canonical coupled CPU runtime is causal and does not solve an algebraic `I_p[n+1] <-> LCFS[n+1]` loop. Romero advances from the already known endpoint `V_B[n]`; the resulting `I_p[n+1]` then advances the passive vessel and determines the total field and one physical LCFS endpoint for the new step. Never place the full topology-first LCFS extractor inside a scalar root/fixed-point iteration over `I_p`.
- Runtime boundary voltage remains the same physical quantity `V_B = -dPhi_B/dt`. After the one new LCFS endpoint is available, use a causal backward derivative of boundary flux: first-order backward difference on the first runtime step, then second-order BDF from the current and two previous boundary-flux samples. Do not restore the old recursive `V_B[n+1] = 2*V_B_avg - V_B[n]` closure.
- The `EquilibriumBoundary` already computed by canonical current dynamics for the accepted `psi[n+1]` must be reused by the CPU run-artifact boundary tracker. Do not perform a second full LCFS extraction for the same state merely to populate topology/radii artifacts; only the fixed-angle projection may be derived from the cached dense boundary when needed.
- In canonical CPU runtime, `boundary_found=False` or a non-finite passive/current state is an explicit failure. Do not fall back to legacy `_advance_ip`, legacy `WallModel`, measured `I_p`, or another boundary model.
- Historical pre-holdout rule: shot `3864` remained closed through development validation. It has since been used exactly once under the frozen causal initialization contract and failed that holdout replay; do not treat it as an unseen holdout for any later numerical/model change.
- GPU current dynamics is not implemented yet. A configuration selecting the canonical second-order current dynamics must fail explicitly on the GPU backend until a mathematically equivalent GPU mirror exists. Do not silently run the legacy GPU current model under the same configuration.


## 2026-08-09 coupled CPU acceptance and closed-holdout initialization

- The selected reduced second-order Romero CPU plant is frozen after the completed eight-shot coupled forward replay. Development replay used only measured coil currents during simulation and measured `Ip` only after each shot for metrics; weighted MAE was `2649.3 A`, weighted P90 `5993.0 A`, and global MAX `13209.0 A`.
- Shot `3864` must not receive per-shot fitted `Li0`, `Rp0`, or `Rp1`. Its causal initialization uses fixed priors derived only from the eight development fits: median `Li0 = 3.3555055929678565e-6 H`, median `Rp0 = 2.522930283189798e-6 Ohm`, and median affine slope `dRp/dt = -1.0225525656551376e-6 Ohm/s`.
- Holdout passive currents are warmed from zero using exactly the available 50 ms prehistory ending at `t0`, the frozen global wall model, measured prehistory coil currents, and measured prehistory `Ip`. No sample after `t0` may participate in initialization.
- Holdout `V_B(t0)` is obtained causally from the backward second-order derivative of the full coupled LCFS boundary flux at the last three prehistory endpoints. Signed `dIp/dt(t0)` is obtained from the corresponding backward derivative of measured prehistory current. `V_C(t0)` is then derived from Romero equation (40), and the second skin state from equation (58) with `dV_C/dt(t0)=0`.
- The deterministic `3864` initial-state file was generated from this frozen rule and one coupled CPU holdout replay was completed. It failed with `MAE=21500.6 A`, `P90=49013.5 A`, `MAX=63802.2 A`, `FINAL=-63514.7 A`. Do not tune global parameters or causal priors to this result. Because step 19 changes the numerical runtime after that observation, `3864` is no longer a clean independent holdout for the changed runtime; any later replay is diagnostic only unless a new independent holdout dataset is obtained.

## 2026-08-10 compact current-model reset invariants

- This section supersedes every earlier active-current-model instruction in this file. Earlier Romero/current-plant sections remain historical/reference context only where they conflict with the rules below.
- The active replacement for Romero must start from the actually available data contract: measured `Ip` for identification/evaluation and measured/applied active-coil currents as the exogenous current-model input. Do not make `V_B`, `psi`, LCFS, `R_p`, `L_i`, `V_C`, thermal diagnostics, or future endpoint parameters required inputs of the baseline compact current model.
- The baseline model class is a compact discrete-time latent state-space system `x[k+1] = A x[k] + B u[k]`, `Ip[k] = C x[k] + D u[k]`, where `u[k]` is the active-coil current vector. Latent coordinates have no assigned physical meaning unless independent evidence later supports one.
- For a held-out/new shot, reconstruct the latent initial state only from the permitted 50 ms prehistory of `Ip` and coil currents under frozen model parameters. No `Ip` sample after `t0` may participate in initialization or state updates.
- Model-order selection uses exactly development shots `3854`, `3855`, `3856`, `3857`, `3858`, `3859`, `3862`, `3863` in repeated seven-train/one-held-out full free-forward tests. Shot `3864` is not pristine for this new model because it already influenced architecture decisions.
- LCFS/field computation is downstream of predicted `Ip` for this baseline current-model experiment. A boundary-extraction failure must not be wired as a required input that prevents the current model itself from producing the next `Ip`.
- The discarded 2026-08-10 `V_B`-only state-space/order-1..4/offline-GPU branch must not be reused for model-order or input-selection evidence. Reintroducing `V_B` or any other extra current-model input requires a new explicit plan change backed by failure of the `Ip + coil currents` baseline.
- Follow the CPU-first rule for the new current model: accept and freeze the CPU mathematical reference before implementing its GPU mirror. Existing accepted GPU LCFS code is independent and remains valid.

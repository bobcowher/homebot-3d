# HomeBot-3D — Navigation Spine (v1) Design

**Date:** 2026-07-02
**Status:** Approved design, pre-implementation
**Predecessor:** `gym-homebot-2d` (top-down 2D Gymnasium env; goals/TaskManager/HER architecture reused here)

---

## 1. Purpose & Research Framing

HomeBot-3D is the 3D successor to the 2D home-robot environment, built to support **more complex reasoning chains and fine-grained control** for embodied-AI research.

The long-term vision spans four capabilities: 3D navigation, manipulation, long-horizon reasoning, and sim-to-real transfer. These do **not** ship together. They stack in dependency order, and this spec covers only the **navigation spine** — the backbone the others bolt onto later.

**Research context that shapes every decision below:** the intended algorithms are **world-model-driven, sample-efficient methods for real robots** (Dreamer lineage, TD-MPC2-style). The explicit goal is to *not need* massively parallel environments — sample efficiency per real transition is the thesis. This is a deliberate distinction from data-center-scale RL. Consequences:

- Massive parallelism is an **anti-goal**, not a forgone benefit.
- The performance axis that matters is **single-env step + render latency**, not throughput.
- Deterministic, clean episodic replay is a **correctness requirement** (world models train on sequences).
- RGB-only egocentric observation makes this a genuine **POMDP** — precisely the regime where latent world models earn their keep. Observation choice and algorithm class reinforce each other.

## 2. Substrate Decision (and the alternatives rejected)

**Chosen: MuJoCo (standard `mujoco` Python bindings — CPU physics + GPU/EGL rendering).**

Decision chain:
1. **Engine vs simulator → simulator.** Manipulation + sim-to-real (later phases) demand trustworthy contact physics. Game engines (Godot/Unity/Unreal) render well but their physics is built for game feel, not contact dynamics that transfer to hardware. Settled early even though v1 is nav-only, because the substrate must carry all four goals.
2. **Which simulator → MuJoCo.** One physics core cradle-to-grave with zero migration between nav → manipulation → sim-to-real. Python-native, conda-friendly, gold-standard contact physics and sim-to-real adoption, now free/open (DeepMind).
   - *Habitat 3.0* rejected: fastest to a nav demo, but nav-grade physics forces a likely manipulation/sim-to-real migration later — threatening two of the four goals.
   - *Isaac Lab* rejected: does all four out of the box but heaviest/steepest, most opinionated, slowest to a first loop, and Blackwell early-adopter risk on the desktop GPU.

**MJX is explicitly excluded from the design.** MJX accelerates thousands of state-based envs in lockstep — the exact axis this research deliberately does not use. Including it would add a JAX/XLA dependency that (a) serves no goal here and (b) is the weak link for the AMD hardware requirement below. One less dependency.

## 3. Hardware Constraints

- **Desktop (dev):** RTX 5090, 32 GB VRAM, 32 cores, 91 GB RAM, Ubuntu 24.04. Unconstrained.
- **Training server (now):** RTX 3090, 24 GB. May migrate to **AMD** for VRAM value.
- **AMD expansion = moderate requirement.** The stack must survive an NVIDIA→AMD move with zero rewrites.

Implications, already reflected in the design:
- MuJoCo C physics is CPU and vendor-neutral. Its render path (EGL/OSMesa/GLFW OpenGL) works on AMD.
- **No JAX on the backbone** (MJX excluded, §2). JAX's ROCm story is second-class; avoiding it removes the AMD risk.
- **Learning side is PyTorch** (mature ROCm builds), never JAX-based frameworks. The env itself is framework-agnostic; PyTorch applies only to code we write.
- **Render path must stay EGL/vendor-neutral**; verify on AMD when that box arrives.

## 4. Scope Boundary

**In (v1):**
- MuJoCo diff-drive mobile base navigating a 3D house extruded from the 2D grid maps.
- Egocentric forward-facing **RGB** observation.
- Reach-location tasks, reusing the goals/TaskManager/HER architecture.
- Gym + GoalEnv/HER API, human-viewable render.
- Deterministic, seed-reproducible resets and clean episode boundaries.

**Out (deferred; seams left where noted):**
- Arm / manipulation.
- Domain randomization.
- Depth / RGB-D as a *deployed* observation (available only via the privileged seam, §7).
- Multi-floor, multi-robot.
- The LLM / long-horizon reasoning layer.

YAGNI: build the spine, not the platform.

## 5. Architecture

Mirror the 2D package split that worked; swap the substrate.

| Module | Role |
|---|---|
| `env.py` | `_HomeBotCore` + `HomeBot3DEnv` + `HomeBot3DGoalEnv`. Ports the 2D structure. First-class **single-env**, clean Gymnasium, no vectorization assumptions. |
| `world.py` | **New — keystone.** Compiles a 2D grid map → MJCF: wall tiles → box geoms (fixed height), floor → plane, fixtures (fridge/recliner/door) → labeled bodies at the same grid coords. |
| `robot.py` | Diff-drive base: applies `[linear_vel, angular_vel]`, reads pose/collision. |
| `sensors.py` | **New.** Egocentric camera render + the privileged-info seam. |
| `tasks.py` / `goals.py` | Ported near-verbatim; "pickup" → "reach location." |
| `maps.py` | Reuse the 2D `MAP_REGISTRY` grids directly. |

Design intent: each module has one purpose and a well-defined interface; `world.py` is isolated so grid→MJCF geometry can be tested independently of physics or rendering.

## 6. The House Model — Grid Extrusion (continuity keystone)

The existing 2D tile grids compile to MJCF at reset:
- Wall tiles → box geoms of fixed height.
- Floor → plane.
- `fridge` / `recliner` / `door` → labeled bodies at the same grid coordinates as 2D.

This reuses already-authored, already-trusted maps; keeps task locations consistent between 2D and 3D; and lets 3D be validated against a world already understood. `random_start` ports directly: sample a valid floor tile → world xy (same constraints as 2D — not in a wall, robot fits, min distance from goals, seed-deterministic).

## 7. Observation & the Privileged Seam

- **Deployed observation:** egocentric forward **RGB**, 84×84 (matches 2D lineage; cheap; upgradeable to 128 later). No depth, no pose. Monocular RGB-only is the sim-to-real-honest interface.
- **Privileged seam:** the core additionally exposes **pose, depth, and goal-vector** through a `privileged` channel that is **off by default** in the returned observation and available only to training wrappers.

Rationale — this is the primary risk mitigation. Monocular RGB-only navigation from scratch is hard (no depth/pose, sparse reward, high-dim pixels). The seam preserves RGB-only *deployment* while making training tractable:
- For RL: asymmetric actor-critic (privileged critic).
- **For world models: auxiliary reconstruction targets** — pose/depth as Dreamer-style decoder heads that shape the latent without entering the deployed observation. For the intended algorithm class this is close to mandatory.

## 8. Robot & Action

- **Model:** custom MJCF diff-drive base (cylinder body + two wheels), velocity-controlled.
- **Action:** continuous `[linear_vel, angular_vel]`. Matches the 2D continuous mode and is what real mobile bases accept — the sim-to-real-honest interface. **No discrete mode in v1.**

## 9. Tasks, Reward & API

- Reuse `TaskManager` + goals registry.
- **v1 task set:** reach `{trash tile(s), fridge, recliner, door}`, gated by the same `goals=[...]` mechanism (default: all reward).
- **Reward:** sparse reach reward fires termination (mirrors the latest 2D reward commit), with HER `compute_reward` as the tight proxy.
- `GoalEnv` interface preserved so HER keeps working.
- Framework-agnostic Gymnasium API; SB3/CleanRL are possible consumers, not "the framework." Any world-model codebase consumes the env directly.

## 10. Testing

Port the 2D test *rigor* (parity of discipline, not count):
- **World compilation:** grid → MJCF geometry correctness (walls, floor, fixture placement) — testable without physics/render.
- **Robot:** diff-drive kinematics + collision.
- **Reward/termination:** reach detection, termination fires, goal gating.
- **Determinism:** same seed → identical spawn and identical episode; clean episode boundaries (world-model replay correctness).
- **GoalEnv/HER contract.**
- **Headless render:** EGL path produces frames without a display.

## 11. Top Risks

1. **RGB-only training difficulty** — the biggest risk. Mitigated by the privileged seam (§7): asymmetric critic and world-model auxiliary decoders.
2. **Single-env render latency** — the real performance axis (§1). Per-step RGB render dominates. Lever = GPU **EGL** rendering (not CPU OSMesa). Benchmark early on the 3090.
3. **AMD portability** — keep the render path EGL/vendor-neutral and the learning side PyTorch; verify on AMD when that box arrives. No JAX on the backbone (already excluded).

## 12. What v1 Delivers

A clean, low-latency **single** MuJoCo environment: a diff-drive base navigating a 3D house extruded from the trusted 2D grids, observed through an egocentric RGB camera, rewarded for reaching task locations via the ported goals/HER architecture, with a privileged seam built for latent-model supervision — vendor-neutral and portable to AMD, with no parallelism machinery and no JAX.

# ACE-Step 1.5 - GitHub Copilot Instructions

## Project Overview

ACE-Step 1.5 is a highly efficient open-source music foundation model that brings commercial-grade music generation to consumer hardware. The system combines a Language Model (LM) as an omni-capable planner with a Diffusion Transformer (DiT) for audio synthesis, supporting various music generation and editing tasks.

## Tech Stack

### Core Technologies
- **Python 3.11** (strict requirement)
- **PyTorch 2.7+** with CUDA 12.8 (Windows/Linux), MPS (macOS ARM64)
- **Transformers 4.51.0-4.57.x** for LLM inference
- **Diffusers** for diffusion models
- **Gradio 6.2.0** for web UI
- **FastAPI + Uvicorn** for REST API server
- **uv** for dependency management

### Additional Components
- **MLX** (Apple Silicon native acceleration on macOS ARM64)
- **nano-vllm** (optimized LLM inference, non-macOS ARM64)
- **PyTorch Lightning** for training
- **PEFT + LyCORIS** for LoRA training
- **torchao** for quantization and optimization

## Multi-Platform Support

**CRITICAL**: This codebase supports multiple hardware/runtime combinations:
- CUDA (NVIDIA GPUs on Windows/Linux)
- ROCm (AMD GPUs on Linux)
- Intel XPU (Intel GPUs)
- MPS (Apple Silicon on macOS)
- MLX (Apple Silicon native acceleration)
- CPU fallback

**Platform-Specific Guidelines**:
- When fixing bugs or adding features, **DO NOT alter non-target platform paths** unless explicitly required
- Changes to CUDA code should not affect MPS/XPU/CPU paths
- Use `gpu_config.py` for hardware detection and configuration
- Always test changes on the target platform and verify other platforms remain unaffected
- Document platform-specific behavior explicitly in code comments

## Code Organization

### Main Entry Points
- `acestep/acestep_v15_pipeline.py` - Main Gradio UI pipeline
- `acestep/api_server.py` - REST API server
- `cli.py` - Command-line interface
- `acestep/model_downloader.py` - Model downloading utility

### Core Modules
- `acestep/handler.py` - Main audio generation handler (AceStepHandler)
- `acestep/llm_inference.py` - LLM handler for text processing
- `acestep/inference.py` - Core generation logic and parameters
- `acestep/gpu_config.py` - Hardware detection and GPU configuration
- `acestep/audio_utils.py` - Audio processing utilities
- `acestep/constants.py` - Global constants and configuration

### UI Components
- `acestep/gradio_ui/` - Gradio interface components
- `acestep/ui/gradio/` - Additional UI modules
- `acestep/gradio_ui/i18n.py` - Internationalization (50+ languages)

### Training
- `acestep/training/` - LoRA training pipeline
- `acestep/dataset/` - Dataset handling and processing

## Coding Standards

### Python Style
- Follow **PEP 8** conventions
- Use **4 spaces** for indentation (never tabs)
- Line length: prefer ≤ 100 characters, hard limit 120
- Use **double quotes** for strings consistently
- Import order: standard library → third-party → local modules

### Type Hints
- **Required** for all new functions and methods
- Use `Optional[T]` for nullable types
- Use `Union[A, B]` or `A | B` (Python 3.10+) for type unions
- Import types from `typing` module when needed

### Docstrings
- **Mandatory** for all new or modified modules, classes, and functions
- Use Google-style or NumPy-style docstrings consistently
- Include:
  - Brief one-line summary
  - Detailed description (if needed)
  - Args/Parameters with types
  - Returns with type
  - Raises (for exceptions)

Example:
```python
def generate_music(
    prompt: str,
    duration: float = 30.0,
    *,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate music from text prompt.
    
    Args:
        prompt: Text description of desired music
        duration: Length in seconds (default: 30.0)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Audio waveform as numpy array (shape: [samples])
    
    Raises:
        ValueError: If duration is negative or exceeds limits
        RuntimeError: If GPU memory allocation fails
    """
```

### Naming Conventions
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`
- Use descriptive names that convey purpose, not implementation

### Error Handling
- **Never use bare `except:`** - always specify exception types
- Prefer specific exceptions (`ValueError`, `RuntimeError`) over generic `Exception`
- Log errors with context using `loguru` logger
- Handle GPU OOM errors gracefully with fallback strategies
- Validate user inputs early with clear error messages

### Logging
- Use `loguru` logger (already configured in the project)
- Log levels: `debug()`, `info()`, `warning()`, `error()`, `critical()`
- Avoid noisy logs in hot paths
- No `print()` statements in committed code (except CLI output)
- Include context in error logs (device, model, input shapes)

## Module Size and Decomposition

Following project guidelines in `AGENTS.md`:

- **Target module size**: ≤ 150 LOC
- **Hard cap**: 200 LOC (requires justification if exceeded)
- **Functions should do one thing**: If description contains "and", consider splitting
- **Data flow should be explicit**: `data in, data out`, minimize side effects
- **Push decisions up, push work down**: Orchestration at top, details at bottom
- Keep orchestrator/facade modules thin, move logic to focused helpers

## Testing

### Test Framework
- Use **unittest** (not pytest) for consistency
- Test file naming: `test_*.py` or `*_test.py`
- Located in `tests/` directory

### Test Requirements
- **Required** for all bug fixes and new features
- Keep tests deterministic, fast, and focused
- Use mocks/fakes for:
  - GPU operations (avoid requiring hardware)
  - File system I/O
  - Network calls
  - Model downloads

### Minimum Test Coverage
Include at least:
1. **Success path test**: Happy case with expected inputs
2. **Edge case test**: Boundary conditions, null/empty inputs
3. **Regression test**: Verify the bug/issue is fixed

### Running Tests
```bash
# Run targeted tests during development
python -m unittest tests.test_module_name

# Run all tests before submitting PR
python -m unittest discover tests/
```

## Contribution Workflow

### Before Making Changes
1. **Understand the issue scope** - Define explicit boundaries
2. **Propose a minimal plan** - Document what will change
3. **Identify affected platforms** - Note if change is platform-specific

### Making Changes
1. **Minimize blast radius** - Touch only required files/functions
2. **No drive-by refactors** - Don't mix cleanups with bug fixes
3. **Preserve interfaces** - Keep public APIs stable unless required
4. **Test incrementally** - Verify changes as you go
5. **Add focused tests** - Cover changed behavior specifically

### Review Standards
- Changes should be **easily reviewable** (small, focused diffs)
- Self-review before submitting: check for scope creep
- Document risk and validation in PR description
- Note any non-target platform impacts (or verify none)

## Dependencies

### Adding Dependencies
- Use `uv add <package>` to add to `pyproject.toml`
- Check for security vulnerabilities before adding
- Prefer well-maintained packages with active communities
- Consider package size and VRAM impact
- Document why the dependency is needed

### Platform-Specific Dependencies
- Use conditional dependencies in `pyproject.toml`:
  ```toml
  "package; sys_platform == 'darwin' and platform_machine == 'arm64'"
  ```
- Mark platform in dependency comment if not conditional

## Feature Development

### Feature Flags
- **Gate WIP features** behind explicit flags
- Do not expose non-functional UI elements as default
- Use configuration or environment variables for flags
- Keep default behavior stable

### Internationalization
- All user-facing strings must support i18n
- Use `acestep/gradio_ui/i18n.py` translation system
- Support all 50+ languages (use translation keys)
- Never hardcode English strings in UI code

## Git Workflow

### Commits
- **One logical change per commit**
- Write clear, descriptive commit messages
- Format: `<type>: <brief summary>`
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
  - Example: `fix: handle GPU OOM in CUDA inference path`

### Pull Requests
- Reference issue number in PR title/description
- Include:
  - What changed and why
  - Testing performed
  - Platform impacts (or note "no cross-platform changes")
  - Risk assessment
- Keep PRs small and focused (prefer multiple small PRs)
- Respond to review feedback promptly

## Performance Considerations

- **Memory efficiency**: ACE-Step runs on 4GB VRAM - be mindful of allocations
- **Lazy loading**: Load models only when needed
- **Batch operations**: Leverage batch generation (up to 8 songs)
- **Avoid repeated heavy operations**: Cache expensive computations
- **Profile before optimizing**: Use `profile_inference.py` for analysis

## Security

- **Never commit secrets** (API keys, tokens)
- Use `.env.example` for environment variable templates
- Validate all user inputs (audio files, prompts, parameters)
- Sanitize file paths to prevent directory traversal
- Be cautious with `eval()`, `exec()`, or dynamic imports
- Check audio file formats before processing

## Documentation

- Update relevant docs in `docs/` when changing functionality
- Keep `README.md` accurate with current features
- Document breaking changes prominently
- Include examples for new features
- Available languages: English (`en`), Chinese (`zh`), Japanese (`ja`)

## Additional Resources

- **AGENTS.md**: Detailed guidance for AI coding agents
- **CONTRIBUTING.md**: Contribution guidelines and workflow
- **SECURITY.md**: Security policy and reporting
- **docs/**: Installation guides and tutorials
- **examples/**: Usage examples and sample code

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `uv sync` |
| Launch Gradio UI | `uv run acestep` |
| Launch API server | `uv run acestep-api` |
| Download models | `uv run acestep-download` |
| Run tests | `python -m unittest discover tests/` |
| Add dependency | `uv add <package>` |
| Check code style | Use project's existing linters (if configured) |

## Key Principles

1. **Safety first**: Preserve working functionality
2. **Minimal changes**: Smallest possible diff to achieve goal
3. **Platform awareness**: Don't break other hardware configurations
4. **Test thoroughly**: Cover changed behavior with tests
5. **Document clearly**: Make your intent obvious
6. **Review carefully**: Self-review before submitting
7. **Communicate openly**: Ask when uncertain

---

For more detailed agent-specific guidance, see [`AGENTS.md`](../AGENTS.md) in the repository root.

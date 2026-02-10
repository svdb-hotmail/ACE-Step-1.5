"""LoRA management mixin for AceStepHandler."""

import os
from typing import Any, Dict

from loguru import logger
from acestep.constants import DEBUG_MODEL_LOADING
from acestep.debug_utils import debug_log


class LoraManagerMixin:
    """LoRA management behavior mixed into AceStepHandler.

    Expected host attributes:
    - model, device, dtype
    - _base_decoder
    - lora_loaded, use_lora, lora_scale
    """

    def _ensure_lora_registry(self) -> None:
        if not hasattr(self, "_lora_adapter_registry"):
            self._lora_adapter_registry = {}
        if not hasattr(self, "_lora_active_adapter"):
            self._lora_active_adapter = None

    def _debug_lora_registry_snapshot(self, max_targets_per_adapter: int = 20) -> Dict[str, Any]:
        """Return debugger-friendly snapshot of LoRA adapter registry."""
        self._ensure_lora_registry()
        adapters: Dict[str, Any] = {}
        for adapter_name, meta in self._lora_adapter_registry.items():
            targets = meta.get("targets", [])
            entries = []
            for t in targets[:max_targets_per_adapter]:
                module = t.get("module")
                entries.append(
                    {
                        "kind": t.get("kind"),
                        "module_name": t.get("module_name"),
                        "adapter": t.get("adapter"),
                        "module_class": module.__class__.__name__ if module is not None else None,
                    }
                )
            adapters[adapter_name] = {
                "path": meta.get("path"),
                "target_count": len(targets),
                "targets": entries,
                "truncated": len(targets) > max_targets_per_adapter,
            }
        return {
            "active_adapter": self._lora_active_adapter,
            "adapter_names": list(self._lora_adapter_registry.keys()),
            "adapters": adapters,
        }

    def _collect_adapter_names(self) -> list[str]:
        """Best-effort adapter name discovery across PEFT runtime variants."""
        names: list[str] = []
        decoder = getattr(self.model, "decoder", None)
        if decoder is None:
            return names

        def _append_name(value):
            if isinstance(value, str) and value and value not in names:
                names.append(value)

        def _walk(value):
            if value is None:
                return
            if isinstance(value, str):
                _append_name(value)
                return
            if isinstance(value, dict):
                for k in value.keys():
                    _append_name(k)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _walk(item)
                return
            if hasattr(value, "keys") and callable(value.keys):
                try:
                    for k in value.keys():
                        _append_name(k)
                except Exception:
                    pass
            if hasattr(value, "adapters"):
                _walk(getattr(value, "adapters"))
            if hasattr(value, "adapter_names"):
                _walk(getattr(value, "adapter_names"))

        # Common PEFT surface area
        if hasattr(decoder, "get_adapter_names") and callable(decoder.get_adapter_names):
            try:
                _walk(decoder.get_adapter_names())
            except Exception:
                pass

        if hasattr(decoder, "active_adapters"):
            try:
                active = decoder.active_adapters
                _walk(active() if callable(active) else active)
            except Exception:
                pass

        if hasattr(decoder, "active_adapter"):
            try:
                _walk(decoder.active_adapter)
            except Exception:
                pass

        if hasattr(decoder, "peft_config"):
            _walk(getattr(decoder, "peft_config"))

        return names

    @staticmethod
    def _is_lora_like_module(name: str, module) -> bool:
        """Conservative LoRA module detection for mixed PEFT implementations."""
        name_l = name.lower()
        cls_l = module.__class__.__name__.lower()
        mod_l = module.__class__.__module__.lower()
        has_lora_signals = (
            "lora" in name_l
            or "lora" in cls_l
            or ("peft" in mod_l and "lora" in mod_l)
            or hasattr(module, "lora_A")
            or hasattr(module, "lora_B")
        )
        has_scaling_api = (
            hasattr(module, "scaling")
            or hasattr(module, "set_scale")
            or hasattr(module, "scale_layer")
        )
        return has_lora_signals and has_scaling_api

    def _rebuild_lora_registry(self, lora_path: str | None = None) -> tuple[int, list[str]]:
        """Build explicit adapter->target mapping used for deterministic scaling."""
        self._ensure_lora_registry()
        self._lora_adapter_registry = {}

        adapter_names = self._collect_adapter_names()
        if not adapter_names:
            adapter_names = ["default"]
        adapter_names = [a for a in adapter_names if isinstance(a, str) and a]
        if not adapter_names:
            adapter_names = ["default"]

        for adapter in adapter_names:
            self._lora_adapter_registry[adapter] = {
                "path": lora_path,
                "targets": [],
            }

        for module_name, module in self.model.decoder.named_modules():
            if not self._is_lora_like_module(module_name, module):
                continue

            # Path 1: direct scaling dict keyed by adapter name.
            if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                for adapter in adapter_names:
                    if adapter in module.scaling:
                        self._lora_adapter_registry[adapter]["targets"].append(
                            {
                                "module": module,
                                "kind": "scaling_dict",
                                "adapter": adapter,
                                "module_name": module_name,
                            }
                        )
                continue

            # Path 2: adapter-aware method API.
            if hasattr(module, "set_scale"):
                for adapter in adapter_names:
                    self._lora_adapter_registry[adapter]["targets"].append(
                        {
                            "module": module,
                            "kind": "set_scale",
                            "adapter": adapter,
                            "module_name": module_name,
                        }
                    )
                continue

            # Path 3: adapter-agnostic scaling API (safe only for single-adapter).
            if hasattr(module, "scale_layer") and len(adapter_names) == 1:
                adapter = adapter_names[0]
                self._lora_adapter_registry[adapter]["targets"].append(
                    {
                        "module": module,
                        "kind": "scale_layer",
                        "module_name": module_name,
                    }
                )
                continue

            # Path 4: scalar scaling API (safe only for single-adapter).
            if hasattr(module, "scaling") and isinstance(module.scaling, (int, float)) and len(adapter_names) == 1:
                adapter = adapter_names[0]
                self._lora_adapter_registry[adapter]["targets"].append(
                    {
                        "module": module,
                        "kind": "scaling_scalar",
                        "module_name": module_name,
                    }
                )

        total_targets = sum(len(meta["targets"]) for meta in self._lora_adapter_registry.values())

        if self._lora_active_adapter not in self._lora_adapter_registry:
            self._lora_active_adapter = next(iter(self._lora_adapter_registry.keys()), None)

        return total_targets, list(self._lora_adapter_registry.keys())

    def _apply_scale_to_adapter(self, adapter_name: str, scale: float) -> int:
        """Apply scale to registered targets for one adapter."""
        self._ensure_lora_registry()
        meta = self._lora_adapter_registry.get(adapter_name)
        if not meta:
            return 0

        modified = 0
        for target in meta.get("targets", []):
            module = target.get("module")
            kind = target.get("kind")
            if module is None:
                continue

            try:
                if kind == "scaling_dict":
                    adapter = target.get("adapter")
                    if adapter not in module.scaling:
                        continue
                    if not hasattr(module, "_acestep_original_scaling_dict"):
                        module._acestep_original_scaling_dict = {k: v for k, v in module.scaling.items()}
                    module.scaling[adapter] = module._acestep_original_scaling_dict[adapter] * scale
                    modified += 1
                elif kind == "set_scale":
                    module.set_scale(adapter_name, scale)
                    modified += 1
                elif kind == "scale_layer":
                    if hasattr(module, "unscale_layer"):
                        module.unscale_layer()
                        module.scale_layer(scale)
                        modified += 1
                    else:
                        prev = getattr(module, "_acestep_last_scale", None)
                        if isinstance(prev, (int, float)) and prev > 0:
                            module.scale_layer(scale / prev)
                        else:
                            module.scale_layer(scale)
                        module._acestep_last_scale = float(scale)
                        modified += 1
                elif kind == "scaling_scalar":
                    if not hasattr(module, "_acestep_original_scaling_scalar"):
                        module._acestep_original_scaling_scalar = float(module.scaling)
                    module.scaling = module._acestep_original_scaling_scalar * scale
                    modified += 1
            except Exception:
                continue

        return modified

    def load_lora(self, lora_path: str) -> str:
        """Load LoRA adapter into the decoder."""
        if self.model is None:
            return "❌ Model not initialized. Please initialize service first."

        # Check if model is quantized - LoRA loading on quantized models is not supported
        # due to incompatibility between PEFT and torchao (missing get_apply_tensor_subclass argument)
        if self.quantization is not None:
            return (
                f"❌ LoRA loading is not supported on quantized models. "
                f"Current quantization: {self.quantization}. "
                "Please re-initialize the service with quantization disabled, then try loading the LoRA adapter again."
            )

        if not lora_path or not lora_path.strip():
            return "❌ Please provide a LoRA path."

        lora_path = lora_path.strip()

        # Check if path exists
        if not os.path.exists(lora_path):
            return f"❌ LoRA path not found: {lora_path}"

        # Check if it's a valid PEFT adapter directory
        config_file = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_file):
            return f"❌ Invalid LoRA adapter: adapter_config.json not found in {lora_path}"

        try:
            from peft import PeftModel, PeftConfig
        except ImportError:
            return "❌ PEFT library not installed. Please install with: pip install peft"

        try:
            import copy
            # Backup base decoder if not already backed up
            if self._base_decoder is None:
                self._base_decoder = copy.deepcopy(self.model.decoder)
                logger.info("Base decoder backed up")
            else:
                # Restore base decoder before loading new LoRA
                self.model.decoder = copy.deepcopy(self._base_decoder)
                logger.info("Restored base decoder before loading new LoRA")

            # Load PEFT adapter
            logger.info(f"Loading LoRA adapter from {lora_path}")
            self.model.decoder = PeftModel.from_pretrained(
                self.model.decoder,
                lora_path,
                is_trainable=False,
            )
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()

            self.lora_loaded = True
            self.use_lora = True  # Enable LoRA by default after loading
            self._ensure_lora_registry()
            self._lora_active_adapter = None
            target_count, adapters = self._rebuild_lora_registry(lora_path=lora_path)

            logger.info(
                f"LoRA adapter loaded successfully from {lora_path} "
                f"(adapters={adapters}, targets={target_count})"
            )
            debug_log(
                lambda: f"LoRA registry snapshot: {self._debug_lora_registry_snapshot()}",
                mode=DEBUG_MODEL_LOADING,
                prefix="lora",
            )
            return f"✅ LoRA loaded from {lora_path}"

        except Exception as e:
            logger.exception("Failed to load LoRA adapter")
            return f"❌ Failed to load LoRA: {str(e)}"

    def unload_lora(self) -> str:
        """Unload LoRA adapter and restore base decoder."""
        if not self.lora_loaded:
            return "⚠️ No LoRA adapter loaded."

        if self._base_decoder is None:
            return "❌ Base decoder backup not found. Cannot restore."

        try:
            import copy
            # Restore base decoder
            self.model.decoder = copy.deepcopy(self._base_decoder)
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()

            self.lora_loaded = False
            self.use_lora = False
            self.lora_scale = 1.0  # Reset scale to default
            self._ensure_lora_registry()
            self._lora_adapter_registry = {}
            self._lora_active_adapter = None

            logger.info("LoRA unloaded, base decoder restored")
            return "✅ LoRA unloaded, using base model"

        except Exception as e:
            logger.exception("Failed to unload LoRA")
            return f"❌ Failed to unload LoRA: {str(e)}"

    def set_use_lora(self, use_lora: bool) -> str:
        """Toggle LoRA usage for inference."""
        if use_lora and not self.lora_loaded:
            return "❌ No LoRA adapter loaded. Please load a LoRA first."

        self.use_lora = use_lora

        # Use PEFT's enable/disable methods if available
        if self.lora_loaded and hasattr(self.model.decoder, "disable_adapter_layers"):
            try:
                if use_lora:
                    if self._lora_active_adapter and hasattr(self.model.decoder, "set_adapter"):
                        try:
                            self.model.decoder.set_adapter(self._lora_active_adapter)
                        except Exception:
                            pass
                    self.model.decoder.enable_adapter_layers()
                    logger.info("LoRA adapter enabled")
                    # Apply current scale when enabling LoRA
                    if self.lora_scale != 1.0:
                        self.set_lora_scale(self.lora_scale)
                else:
                    self.model.decoder.disable_adapter_layers()
                    logger.info("LoRA adapter disabled")
            except Exception as e:
                logger.warning(f"Could not toggle adapter layers: {e}")

        status = "enabled" if use_lora else "disabled"
        return f"✅ LoRA {status}"

    def set_lora_scale(self, scale: float) -> str:
        """Set LoRA adapter scale/weight (0-1 range)."""
        if not self.lora_loaded:
            return "⚠️ No LoRA loaded"

        # Clamp scale to 0-1 range
        self.lora_scale = max(0.0, min(1.0, scale))

        # Only apply scaling if LoRA is enabled
        if not self.use_lora:
            logger.info(f"LoRA scale set to {self.lora_scale:.2f} (will apply when LoRA is enabled)")
            return f"✅ LoRA scale: {self.lora_scale:.2f} (LoRA disabled)"

        try:
            if not getattr(self, "_lora_adapter_registry", None):
                self._rebuild_lora_registry()

            active_adapter = self._lora_active_adapter
            if active_adapter is None and self._lora_adapter_registry:
                active_adapter = next(iter(self._lora_adapter_registry.keys()))
                self._lora_active_adapter = active_adapter

            debug_log(
                lambda: (
                    f"LoRA scale request: slider={self.lora_scale:.3f} "
                    f"active_adapter={active_adapter} adapters={list(self._lora_adapter_registry.keys())}"
                ),
                mode=DEBUG_MODEL_LOADING,
                prefix="lora",
            )

            modified_count = self._apply_scale_to_adapter(active_adapter, self.lora_scale) if active_adapter else 0

            if modified_count > 0 and active_adapter:
                logger.info(
                    f"LoRA scale set to {self.lora_scale:.2f} "
                    f"(adapter={active_adapter}, modified={modified_count})"
                )
                return f"✅ LoRA scale: {self.lora_scale:.2f}"
            else:
                logger.warning(
                    "No registered LoRA scaling targets found for active adapter"
                )
                return f"⚠️ Scale set to {self.lora_scale:.2f} (no modules found)"
        except Exception as e:
            logger.warning(f"Could not set LoRA scale: {e}")
            return f"⚠️ Scale set to {self.lora_scale:.2f} (partial)"

    def set_active_lora_adapter(self, adapter_name: str) -> str:
        """Set the active LoRA adapter for scaling/inference.

        This is backward compatible with single-adapter UI and is forward-compatible
        for future multi-LoRA controls.
        """
        self._ensure_lora_registry()
        if adapter_name not in self._lora_adapter_registry:
            return f"❌ Unknown adapter: {adapter_name}"
        self._lora_active_adapter = adapter_name
        debug_log(f"Selected active LoRA adapter: {adapter_name}", mode=DEBUG_MODEL_LOADING, prefix="lora")
        if self.model is not None and hasattr(self.model, "decoder") and hasattr(self.model.decoder, "set_adapter"):
            try:
                self.model.decoder.set_adapter(adapter_name)
            except Exception:
                pass
        return f"✅ Active LoRA adapter: {adapter_name}"

    def get_lora_status(self) -> Dict[str, Any]:
        """Get current LoRA status."""
        self._ensure_lora_registry()
        return {
            "loaded": self.lora_loaded,
            "active": self.use_lora,
            "scale": self.lora_scale,
            "active_adapter": self._lora_active_adapter,
            "adapters": list(self._lora_adapter_registry.keys()),
        }

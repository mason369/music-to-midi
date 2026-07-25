import logging
from unittest.mock import patch

import pytest
import torch

from src.utils.inference_precision import (
    ARIA_AMT,
    LEAP_XE,
    MIROS,
    MUSCRIPTOR,
    POLARFORMER,
    YOURMT3,
    PrecisionCapabilities,
    log_precision_plan,
    select_inference_precision,
    verify_float32_model_parameters,
)


def _capabilities(*, bf16: bool) -> PrecisionCapabilities:
    return PrecisionCapabilities(
        device="cuda:0",
        device_name="Test NVIDIA GPU",
        compute_capability="8.9" if bf16 else "7.5",
        fp32=True,
        fp16=True,
        bf16=bf16,
        tf32=bf16,
        bf16_probe_error="" if bf16 else "native BF16 unavailable",
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        (YOURMT3, "float32"),
        (MUSCRIPTOR, "float16"),
        (ARIA_AMT, "float32"),
        (MIROS, "float16"),
        (LEAP_XE, "float16"),
        (POLARFORMER, "float32"),
    ),
)
def test_turing_precision_policy(model, expected):
    with patch(
        "src.utils.inference_precision.probe_precision_capabilities",
        return_value=_capabilities(bf16=False),
    ):
        plan = select_inference_precision(model, "cuda:0")

    assert plan.parameter_dtype == "float32"
    assert plan.compute_dtype == expected
    assert plan.autocast is (expected != "float32")
    assert plan.allow_tf32 is False


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        (YOURMT3, "float32"),
        (MUSCRIPTOR, "float16"),
        (ARIA_AMT, "bfloat16"),
        (MIROS, "bfloat16"),
        (LEAP_XE, "float16"),
        (POLARFORMER, "float32"),
    ),
)
def test_ampere_ada_blackwell_precision_policy(model, expected):
    with patch(
        "src.utils.inference_precision.probe_precision_capabilities",
        return_value=_capabilities(bf16=True),
    ):
        plan = select_inference_precision(model, "cuda:0")

    assert plan.parameter_dtype == "float32"
    assert plan.compute_dtype == expected
    assert plan.autocast is (expected != "float32")
    assert plan.allow_tf32 is False


@pytest.mark.parametrize("compute_capability", ("8.0", "8.6", "8.9", "12.0"))
def test_native_bf16_cuda_generations_use_the_same_capability_driven_policy(
    compute_capability,
):
    capabilities = _capabilities(bf16=True)
    capabilities = PrecisionCapabilities(
        **{
            **capabilities.__dict__,
            "compute_capability": compute_capability,
        }
    )
    with patch(
        "src.utils.inference_precision.probe_precision_capabilities",
        return_value=capabilities,
    ):
        aria_plan = select_inference_precision(ARIA_AMT, "cuda:0")
        miros_plan = select_inference_precision(MIROS, "cuda:0")

    assert aria_plan.compute_dtype == "bfloat16"
    assert miros_plan.compute_dtype == "bfloat16"


def test_precision_log_reports_gpu_parameters_compute_and_capabilities(caplog):
    with patch(
        "src.utils.inference_precision.probe_precision_capabilities",
        return_value=_capabilities(bf16=False),
    ):
        plan = select_inference_precision(MIROS, "cuda:0")

    logger = logging.getLogger("precision-policy-test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_precision_plan(logger, plan)

    message = caplog.messages[-1]
    assert "gpu=Test NVIDIA GPU" in message
    assert "cc=7.5" in message
    assert "parameters=float32" in message
    assert "compute=float16" in message
    assert "fp32=True" in message
    assert "fp16=True" in message
    assert "bf16=False" in message
    assert "tf32_enabled=False" in message


def test_parameter_contract_rejects_whole_model_fp16_conversion():
    model = torch.nn.Linear(4, 4).half()

    with pytest.raises(RuntimeError, match="parameter dtype contract violated"):
        verify_float32_model_parameters(
            model,
            model_name="test model",
            torch_module=torch,
        )

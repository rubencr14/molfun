"""
Pytest configuration and fixtures.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Get available device for testing"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture(scope="session")
def dtype():
    """Default dtype for tests"""
    return torch.float16


@pytest.fixture
def sample_sequences():
    """Sample protein sequences for testing"""
    return [
        "MKTAYIAKQR",
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTAYIAKQRQISFPD",
    ]


@pytest.fixture
def single_sequence():
    """Single protein sequence for testing"""
    return "MKTAYIAKQR"

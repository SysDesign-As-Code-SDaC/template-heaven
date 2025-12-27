"""
Tests for topic-to-stack mapping and candidate stack inference.
"""

from templateheaven.core.template_manager import TemplateManager
from templateheaven.config.settings import Config
from templateheaven.core.models import StackCategory


def test_infer_stack_enum_common_topics(tmp_path):
    cfg = Config(config_dir=tmp_path / 'cfg')
    manager = TemplateManager(cfg)

    # Topics that should map to FULLSTACK
    assert manager._infer_stack_enum(['nextjs']) == StackCategory.FULLSTACK
    assert manager._infer_stack_enum(['next.js']) == StackCategory.FULLSTACK
    assert manager._infer_stack_enum(['trpc', 'nextjs']) == StackCategory.FULLSTACK

    # Topics that should map to FRONTEND
    assert manager._infer_stack_enum(['react']) == StackCategory.FRONTEND
    assert manager._infer_stack_enum(['vite']) == StackCategory.FRONTEND

    # Topics that should map to BACKEND
    assert manager._infer_stack_enum(['fastapi']) == StackCategory.BACKEND
    assert manager._infer_stack_enum(['django']) == StackCategory.BACKEND

    # AI/ML
    assert manager._infer_stack_enum(['pytorch']) == StackCategory.AI_ML

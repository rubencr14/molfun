"""
molfun.pipelines — composable workflows for protein ML.

Usage::

    from molfun.pipelines import Pipeline, PipelineStep
    from molfun.pipelines.steps import fetch_step, split_step, train_step

    pipeline = Pipeline([
        PipelineStep("fetch", fetch_step, config={"collection": "kinases_human"}),
        PipelineStep("split", split_step),
        PipelineStep("train", train_step, config={"strategy": "lora", "epochs": 20}),
    ])
    result = pipeline.run()

    # Or from YAML:
    pipeline = Pipeline.from_yaml("recipes/kinase_finetune.yaml")
    pipeline.run()
"""

from molfun.pipelines.pipeline import Pipeline, PipelineStep, StepResult

__all__ = ["Pipeline", "PipelineStep", "StepResult"]

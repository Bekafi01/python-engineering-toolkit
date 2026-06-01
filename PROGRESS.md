# Progress Log

## Day 1

- Project scaffolded with uv + src/ layout
- BasePreprocessor ABC with abstract transform() and validate()
- NumericPreprocessor, CategoricalPreprocessor, DatePreprocessor
- PreprocessorConfig dataclass with post_init validation

## Day 2

- Transformer Protocol (PEP 544) — duck typing with type safety
- **slots** on all preprocessors for memory efficiency
- FeaturePipeline (Strategy pattern) — accepts any Transformer
- PreprocessorFactory (Factory pattern) — central object creation with .register()
- Pipeline — chains steps, tracks timing, implements **iter** and **len**

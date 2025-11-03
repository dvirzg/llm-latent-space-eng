# LLM Latent Space Geometry and Reasoning

This repository tests whether embedding geometry causally determines reasoning style in large language models. We manipulate the spatial arrangement of category exemplars (dog, cat, hamster) in gpt-oss-20b's embedding space, creating tight vs. loose clusters, and observe/measure how this affects the model's reasoning about category boundaries (e.g., "Is a monkey a pet?"). The hypothesis: tight clusters produce rigid, rule-based categorization; loose clusters produce flexible, graded reasoning.

**Method**: Direct modification of embedding matrix weights.
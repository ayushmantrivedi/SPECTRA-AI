# .antigravity/rules.md

## Code Standards
- Language: Python 3.10+
- Type hints: Mandatory for all functions
- Docstrings: Google-style for classes and methods
- Test coverage: >85% for core modules

## Architecture Rules
- Single Responsibility: Each module has one purpose
- Async-first: Optimize for concurrent requests
- Memory hygiene: Immediately unload models after use
- Logging: Structured logging with JSON output

## VLM Integration
- Never hardcode API keys; use environment variables
- Implement exponential backoff for API rate limits
- Cache model responses when possible
- Always include structured output schemas

## Machine Learning
- All model weights must be versioned
- Training loops must be reproducible (set random seeds)
- Maintain train/val/test split ratio of 7:1.5:1.5
- Document model performance metrics (SSIM, FID, user studies)

## Testing
- Unit tests: Fast, isolated, no network calls
- Integration tests: With mock API responses
- E2E tests: Full pipeline from image to verification
- Performance tests: Track inference latency and memory

# System Architecture

## Principles
1. **No Orchestration Frameworks**: No LangChain, CrewAI, etc. The loop is handwritten.
2. **Normalized Boundaries**: The application logic only interacts with Pydantic models (`NormalizedResponse`), never SDK-specific classes.
3. **Environment Driven**: Secrets and configurations are loaded via `dotenv` and validated at startup.

## Structure
- `src/core/`: Configuration, global logging, and application-specific exceptions.
- `src/llm/`: The abstract provider interface, Groq and Gemini implementations, and Fake client for deterministic testing.

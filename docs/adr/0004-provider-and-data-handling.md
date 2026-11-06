# ADR 0004: Pluggable LLM provider and evidence handling

- Status: Accepted
- Context: CI must work without credentials, while local or hosted inference can be
  selected at runtime without changing the request/response contract.
- Decision: Keep the stub provider and implement Ollama and OpenAI-compatible hosted
  providers behind `LLMProvider`. The model receives only the question and retrieved
  passages, is instructed that passages are untrusted data, and must return the Pydantic
  `Answer` schema. Hosted credentials come only from `LLM_API_KEY` environment/secret
  configuration and are not logged.
- Consequence: Hosted mode sends the question and retrieved passages to the configured
  service; corpus owners must review that provider's data handling before enabling it.
  The stub remains available for offline tests.

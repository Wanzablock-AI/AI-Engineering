# Multi-Agent Research

A production-focused multi-agent research system.

## Responsibilities

- Decompose research requests
- Assign focused research tasks to agents
- Execute independent research tasks
- Synthesize research results
- Preserve clear separation between research and synthesis

## Architecture

Research Request
      ↓
Research Planner
      ↓
Task Allocation
      ↓
┌─────────────┬─────────────┬─────────────┐
│ Researcher 1│ Researcher 2│ Researcher 3│
└─────────────┴─────────────┴─────────────┘
      ↓
Research Results
      ↓
Synthesizer
      ↓
Final Research Result

## Development Principles

- Deterministic orchestration
- Clear agent responsibilities
- Test-driven development
- Local-first where practical
- Observable execution
- Dockerized deployment
- Explicit separation of planning, research, and synthesis

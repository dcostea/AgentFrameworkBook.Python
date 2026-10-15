# Python interview sprint for a .NET developer

**Target:** general Python backend interview. **Budget:** four-hour core, optionally four more hours. You already know programming, HTTP, SQL and OOP; spend time on Python semantics, not beginner computer science. This sprint builds working basics, not guaranteed interview readiness or production mastery.

## Before studying: use the correct interpreter

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe .\05.Agents\AgentWithStructuredOutput\main.py
```

The last command makes a paid model request. For offline practice:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agents.py -q
```

VS Code: open this folder, choose **Python: Select Interpreter**, select `.venv\Scripts\python.exe`, and set a breakpoint in `main.py`. Use **Run Python File in Terminal**. Activating `.venv` changes command lookup; it is optional when you invoke its Python executable explicitly. `.venv` isolates installed packages; `.env` is a separate, plaintext configuration file. Neither is a .NET secrets store.

## Core: exactly four hours

| Time | Learn | Hands-on exercise / exit criterion |
| --- | --- | --- |
| 0:00–0:20 | Interpreter, `venv`, `python -m pip`, modules, imports, `__main__` guard; reading a traceback bottom-up | Explain why installing into a different Python interpreter causes `ModuleNotFoundError`. Debug one sample without changing `.env`. |
| 0:20–1:10 | Indentation, `def`, loops, f-strings, `None`, truthiness; lists/dicts/sets/tuples; slicing, comprehensions, `enumerate`, `zip` | Write `summarize_moves(moves)` that validates distances, totals meters by direction and preserves order. Handle empty input. Use `dict.get`, not exception-driven lookup. |
| 1:10–1:50 | Names bind to objects; mutability, aliasing, shallow copy; `==` vs `is`; function arguments/defaults; `*args`, `**kwargs`; iterators and `yield` | Demonstrate why two names share a list, why `items=[]` is a bad default, and why a generator is consumed once. Fix all three without searching. |
| 1:50–2:25 | Classes, `self`, `__init__`, dataclasses; type hints vs runtime validation; duck typing, `Protocol`; exceptions, context managers, decorators | Read `shared\motor_tools.py` and `05.Agents\AgentWithStructuredOutput\main.py`. Explain `@tool`, `Annotated` and Pydantic. Create a small dataclass and raise/catch `ValueError`. |
| 2:25–3:00 | `async def`, coroutine vs Task, `await`, `asyncio.run`, `gather`/`TaskGroup`, timeouts/cancellation, blocking I/O | Run two `asyncio.sleep` calls sequentially then concurrently. Explain why calling an async function alone does not execute its body and why `time.sleep` blocks the event loop. |
| 3:00–3:35 | Backend basics: request/response models, validation, HTTP status codes, dependency boundaries, configuration, logging, tests | Sketch a FastAPI `POST /moves` endpoint with Pydantic request validation, a service function, and error mapping. Write three `unittest` tests for the service; no API keys/network needed. |
| 3:35–4:00 | Timed interview rehearsal | Solve a list/dictionary aggregation problem in 15 minutes. Spend 10 minutes answering the questions below aloud. Explain tradeoffs while coding. |

**Four-hour priority:** produce readable, correct code with tests. Prefer functions and standard collections over translating every C# class/interface into a Python class.

## Optional extension: reach eight hours total

| Time | Focus | Deliverable |
| --- | --- | --- |
| 4:00–5:00 | Small real HTTP API | In a separate practice project/environment, build the sketched FastAPI endpoint. Add GET and POST, positive-distance validation, 404/422 handling, and tests with a test client. Learn Uvicorn/ASGI and `/docs`. Do not add interview-only packages to this book workspace. |
| 5:00–5:45 | Persistence and API design | Use standard-library `sqlite3`: parameterized SQL, transaction boundaries and rollback. Discuss sync DB drivers in async endpoints, pooling, ORM migrations and SQLAlchemy as an EF Core analogy—not a drop-in replacement. |
| 5:45–6:30 | Testing/debugging | Learn pytest test functions/fixtures/parameterization, `unittest.mock`, `AsyncMock`, and patching where a symbol is looked up. Cover success, invalid input and dependency failure. Explain why HTTP mocks cannot prove a live provider works. |
| 6:30–7:10 | Concurrency and deployment | Discuss the usual CPython GIL, I/O threads vs CPU processes, ASGI workers, cancellation, timeouts and retry/idempotency. Review `pyproject.toml`, version constraints/lock files and import paths. |
| 7:10–8:00 | Mock interview | 25 minutes coding, 15 minutes backend design, 10 minutes reviewing weak answers. Reimplement the solution without copying the first attempt. |

## The .NET-to-Python mental model

| .NET concept | Python counterpart / important difference |
| --- | --- |
| `.csproj` / NuGet | `pyproject.toml` plus dependency management; this sample collection uses `requirements.txt`. A module import name may differ from its distribution name. |
| CLR / assemblies | Interpreter, modules and packages. Typical CPython compiles source to bytecode, but there is no C#-style compile-time type safety gate. Imports can execute top-level code. |
| Namespaces / `using` | Modules/packages and `import`; `__init__.py` is common, although namespace packages can omit it. `python -m package.module` uses package-aware execution. |
| `List<T>` / `Dictionary<K,V>` | `list` / `dict`. Collections are mutable; dict preserves insertion order; set does not promise meaningful order. List membership is O(n), dict/set membership is average O(1). |
| LINQ | Comprehensions for eager collections, generator expressions/`yield` for lazy iteration. `sorted` returns a new list; `list.sort` mutates and returns `None`. |
| Nullable / `null` | `None`; `T | None` is a type hint, not enforced validation. Use `is None`; empty strings/lists and zero are falsey but are not `None`. |
| DTO / record | `@dataclass` for data containers; Pydantic for boundary parsing/validation. Dataclasses and annotations alone do not validate incoming JSON. |
| Interface / DI | Duck typing, optional `Protocol`/ABC and explicit constructor/function injection. A DI container is not intrinsic to Python. |
| `using` / `await using` | `with` / `async with`, backed by context-manager protocols. Use them for deterministic cleanup; don't rely on garbage collection. |
| `async Task<T>` | `async def` returns a coroutine object. It starts when awaited or scheduled; a normal C# async call usually runs immediately until its first incomplete await. |
| `Task.WhenAll` | `asyncio.gather`; `TaskGroup` provides structured task lifetime and sibling cancellation on failures. Default `gather` does not automatically cancel all siblings on the first failure. |
| Attributes | Decorators can wrap/replace functions or classes; not merely metadata. `Annotated` attaches metadata to type hints. |
| Access modifiers | `_name` is a convention; `__name` triggers name mangling, not strong privacy. Avoid inheritance unless it adds clear value. |
| User Secrets / configuration | Environment variables and optional dotenv; `.env` is plaintext and must be ignored. Credential managers/vaults are separate integrations. |

## Interview traps to rehearse

1. **Mutable defaults:** defaults are evaluated once at function definition. Use `None`, create the list inside; dataclass fields use `default_factory`.
2. **Identity vs equality:** `==` compares values; `is` compares identity. Use `is None`, not `is` for string/number values.
3. **Object sharing:** arguments share object references. Mutating a passed list is visible to the caller; rebinding the local parameter is not. `copy()` is shallow.
4. **Closures:** loop-created closures capture names, not a fresh snapshot of each iteration; bind explicitly when needed.
5. **Async is not parallel CPU work:** it overlaps waiting. In ordinary GIL-enabled CPython, Python CPU-bound threads do not run bytecode in parallel; native extensions can release the GIL, and optional free-threaded builds differ. Processes are a common CPU-parallel choice.
6. **Typing is not validation:** a type checker can warn; runtime boundary validation needs explicit checks or a library such as Pydantic.
7. **Blocking calls:** use async libraries for network I/O or `asyncio.to_thread` for suitable blocking I/O; don't indiscriminately offload everything.
8. **Security:** don't unpickle untrusted input; use parameterized SQL, validate payloads and never log secrets.
9. **Exceptions:** catch specific types; preserve causes with `raise ... from exc`; use `finally`/context managers for cleanup. Handle cancellation separately and normally re-raise it.
10. **Model integration:** SDK/API capabilities and model constraints differ across languages. Check tool schemas, streaming, options and persistence—not only method-name similarity.

## Self-check questions

- What happens when you call `async def` without `await`?
- Why can `list.sort()` not be chained like LINQ `OrderBy`?
- What is the difference between iterable, iterator and generator?
- When do you use a dataclass, a dictionary, or Pydantic?
- How do you test an async function without making a network request?
- What does `if __name__ == "__main__":` protect against?
- How would you enforce an HTTP timeout and avoid duplicate writes during retries?
- Which application state belongs in a request/session rather than a process-global variable?

## Reading: use examples, not entire books

- [Python tutorial: control flow/functions](https://docs.python.org/3.13/tutorial/controlflow.html)
- [Collections and comprehensions](https://docs.python.org/3.13/tutorial/datastructures.html)
- [Names, objects and classes](https://docs.python.org/3.13/tutorial/classes.html)
- [Errors and exceptions](https://docs.python.org/3.13/tutorial/errors.html)
- [Virtual environments and packages](https://docs.python.org/3.13/tutorial/venv.html)
- [Coroutines, Tasks and TaskGroup](https://docs.python.org/3.13/library/asyncio-task.html)
- [Type hints: runtime limitations](https://docs.python.org/3.13/library/typing.html)
- [FastAPI first steps](https://fastapi.tiangolo.com/tutorial/first-steps/)

**Skip for this sprint:** advanced metaclasses/descriptors, C extensions, deep ML mathematics, Kubernetes, elaborate packaging and every orchestration chapter. Your interview is general backend Python; the agent examples are useful practice, not the entire syllabus.

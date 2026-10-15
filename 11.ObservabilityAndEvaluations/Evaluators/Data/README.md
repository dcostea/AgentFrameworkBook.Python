# Evaluator JSONL data

All twelve `.jsonl` files are byte-for-byte copies of the C# chapter fixtures (30 records). The loader accepts UTF-8 with or without a BOM, preserves tool spelling, arguments, results, references, chunks, IDs and expectations, and validates case-insensitive record fields.

Common fields: `id`, `source`, `scenario`, `agent` (name, modelProvider, modelName, instructions, tools), `userInput`, `toolCalls` (name, arguments, result), `finalResponse`, `expectedBehavior`, `shouldPass`.

Additional inputs: `groundTruth` for groundedness/equivalence/completeness/composite; `retrievedContextChunks` for retrieval. Direction records instead use `expectedChanged`, `expectedFrom`, `expectedTo`.

Expectations are assertions, never judge inputs. Fixture `modelName`/`modelProvider` are historical metadata; the configured live judge model/provider is selected separately. Tools are NOT executed during evaluation. For task/tool metrics, matching the C# tests, only assistant function calls (not fabricated final text or recorded tool results) are evaluated.

See the [chapter README](..\..\README.md) for algorithm differences, source casing correction and opt-in model scoring.

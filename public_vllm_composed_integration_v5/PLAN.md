# Native-default seven-adapter v5

Preserves v4 and its frozen pins. Only cleanup control flow changed: read-only precision drift remains HOLD, while synchronization, adapter restoration, engine shutdown, alarm cancellation and result save are independently attempted. Existing exception is preserved; cleanup-only failure raises HOLD. No numerical/API/adapter changes, no backend flag assignment. Same eager native True/True 1616-call shadow and conditional replacement study; not speed or new operation coverage.

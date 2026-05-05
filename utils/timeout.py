import asyncio
import functools
import multiprocessing
import os
import queue

from fastapi import HTTPException

TIME_OUT = 10

# On POSIX (Linux/macOS) use 'fork' – the child inherits the parent's memory
# space so no pickling is needed and startup is fast. On other platforms
# (Windows) use 'spawn', which starts a fresh interpreter and pickles arguments.
_MP_START_METHOD = "fork" if os.name == "posix" else "spawn"


def _worker(fn, args, kwargs, result_queue):
  """Runs fn(*args, **kwargs) in a child process and puts the result in result_queue."""
  try:
    result = fn(*args, **kwargs)
    result_queue.put(("ok", result))
  except Exception as e:
    result_queue.put(("error", e))


def _run_in_process(fn, args, kwargs):
  """
  Runs fn(*args, **kwargs) in a child process.
  Kills the process if it exceeds TIME_OUT seconds.
  Raises TimeoutError on timeout, or re-raises any exception from the child.
  """
  ctx = multiprocessing.get_context(_MP_START_METHOD)
  result_queue = ctx.Queue()
  process = ctx.Process(target=_worker, args=(fn, args, kwargs, result_queue))
  process.start()
  process.join(timeout=TIME_OUT)

  if process.is_alive():
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
      process.kill()
      process.join()
    raise TimeoutError()

  try:
    status, value = result_queue.get(block=False)
  except queue.Empty:
    raise RuntimeError(
      f"Worker process exited unexpectedly with code {process.exitcode}"
    )

  if status == "ok":
    return value
  raise value


async def run_with_timeout(fn, *args, **kwargs):
  """
  Execute fn(*args, **kwargs) in a subprocess with a time limit.

  ``fn`` must be a module-level function (or any other picklable callable) so
  that the 'spawn' start method (used on non-POSIX platforms) can locate it in
  the child process.  Returns the function's return value, or raises
  ``HTTPException(408)`` if the time limit is exceeded.
  """
  loop = asyncio.get_running_loop()
  try:
    return await loop.run_in_executor(
      None,
      functools.partial(_run_in_process, fn, args, kwargs),
    )
  except TimeoutError:
    raise HTTPException(
      status_code=408,
      detail=f"Optimization timed out after {TIME_OUT} seconds",
    )

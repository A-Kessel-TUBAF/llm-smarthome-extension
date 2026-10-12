/-
  Checker executable (runtime IO; not covered by the soundness theorem).

    allocation_checker --version    identity JSON on stdout
    allocation_checker --oneshot    one request on stdin, one response on stdout
    allocation_checker --serve      persistent JSONL worker: one request per line

  Diagnostics about the process go to stderr; stdout carries only JSON lines.
-/
import AllocationChecker.Codec

open AllocationChecker.Codec

def identityLine : String :=
  (Lean.Json.mkObj
    [ ("checker_release", .str checkerRelease), ("spec_version", .str specVersion),
      ("request_schema", .str schemaRequest), ("response_schema", .str schemaResponse),
      ("lean_version", .str Lean.versionString) ]).compress

partial def serveLoop (stdin : IO.FS.Stream) (stdout : IO.FS.Stream) : IO Unit := do
  let line ← stdin.getLine
  if line.isEmpty then return
  let trimmed := line.trimAsciiEnd.toString
  unless trimmed.isEmpty do
    stdout.putStrLn (← handle trimmed)
    stdout.flush
  serveLoop stdin stdout

def main (args : List String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  match args with
  | ["--version"] =>
    stdout.putStrLn identityLine
    return 0
  | ["--oneshot"] =>
    let input ← (← IO.getStdin).readToEnd
    let resp ← handle input.trimAscii.toString
    stdout.putStrLn resp
    stdout.flush
    return 0
  | ["--serve"] =>
    stderr.putStrLn s!"{checkerRelease}: serving JSONL on stdin"
    serveLoop (← IO.getStdin) stdout
    return 0
  | _ =>
    stderr.putStrLn "usage: allocation_checker (--version | --oneshot | --serve)"
    return 64

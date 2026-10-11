import SmartHome

open SmartHome SmartHome.Examples

def report (name : String) (s : State) (b : Batch) : IO Unit :=
  IO.println s!"{name}: {if check cards s b then "ACCEPTED" else "REJECTED"}"

def main : IO Unit := do
  report "goodBatch" state goodBatch
  report "conflictBatch" state conflictBatch
  report "unknownDeviceBatch" state unknownDeviceBatch
  report "outOfRangeBatch" state outOfRangeBatch
  report "missingArgBatch" state missingArgBatch
  report "wrongTypeBatch" state wrongTypeBatch
  report "wrongKindBatch" state wrongKindBatch
  report "goodBatch (lights offline)" stateLightsOffline goodBatch

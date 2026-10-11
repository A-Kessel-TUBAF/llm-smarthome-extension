import SmartHome

open SmartHome SmartHome.Examples

def report (name : String) (s : HomeState) (b : Batch) : IO Unit :=
  IO.println s!"{name}: {if check cards s b then "ACCEPTED" else "REJECTED"}"

def main : IO Unit := do
  report "goodBatch" state goodBatch
  report "twoRoomsBatch" state twoRoomsBatch
  report "conflictBatch" state conflictBatch
  report "tooHotBatch" state tooHotBatch
  report "unknownDeviceBatch" state unknownDeviceBatch
  report "missingArgBatch" state missingArgBatch
  report "wrongTypeBatch" state wrongTypeBatch
  report "wrongKindBatch" state wrongKindBatch
  report "noSuchRoomBatch" state noSuchRoomBatch
  report "goodBatch (thermostat offline)" stateThermostatOffline goodBatch

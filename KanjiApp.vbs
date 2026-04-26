Set oShell = CreateObject("WScript.Shell")
oShell.CurrentDirectory = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\") - 1)
oShell.Run """" & oShell.CurrentDirectory & "\.venv\Scripts\pythonw.exe"" """ & oShell.CurrentDirectory & "\gui.py""", 0, False

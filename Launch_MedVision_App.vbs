Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

' Set the working directory to the folder where this script is located
strPath = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = strPath

exePath = ""
' Look for the isolated virtual environment executable so it doesn't need global python
If fso.FileExists("venv_tf\Scripts\pythonw.exe") Then
    exePath = "venv_tf\Scripts\pythonw.exe"
ElseIf fso.FileExists("venv\Scripts\pythonw.exe") Then
    exePath = "venv\Scripts\pythonw.exe"
ElseIf fso.FileExists("venv_tf\Scripts\python.exe") Then
    exePath = "venv_tf\Scripts\python.exe"
ElseIf fso.FileExists("venv\Scripts\python.exe") Then
    exePath = "venv\Scripts\python.exe"
Else
    ' Fallback to system python
    exePath = "pythonw.exe"
End If

' Run the launcher script silently (0 means hide window)
WshShell.Run Chr(34) & exePath & Chr(34) & " desktop_launcher.py", 0
Set WshShell = Nothing

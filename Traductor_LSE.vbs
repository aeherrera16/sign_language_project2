' ============================================================
'  TRADUCTOR LSE - Lengua de Señas Ecuatoriana
'  Doble clic para abrir (SIN terminal visible)
'  Detecta runtime portable (CI/CD) o venv local
' ============================================================

Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
Set objEnv = WshShell.Environment("Process")

ScriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
MenuScript = ScriptDir & "\prototipo\menu.py"

' Buscar Python en orden de prioridad
PythonExe = ""

' 1. Runtime portable (descargado del CI/CD)
If FSO.FileExists(ScriptDir & "\runtime\pythonw.exe") Then
    PythonExe = ScriptDir & "\runtime\pythonw.exe"

' 2. Venv local (.venv creado por Iniciar_LSE.bat)
ElseIf FSO.FileExists(ScriptDir & "\.venv\Scripts\pythonw.exe") Then
    PythonExe = ScriptDir & "\.venv\Scripts\pythonw.exe"
End If

' Si encontramos Python -> lanzar directo (sin terminal)
If PythonExe <> "" Then
    objEnv("TF_CPP_MIN_LOG_LEVEL") = "3"
    objEnv("TF_ENABLE_ONEDNN_OPTS") = "0"
    objEnv("MEDIAPIPE_DISABLE_GPU") = "1"
    objEnv("GLOG_minloglevel") = "3"
    objEnv("ABSL_MIN_LOG_LEVEL") = "3"
    objEnv("PYTHONWARNINGS") = "ignore"

    WshShell.CurrentDirectory = ScriptDir
    WshShell.Run Chr(34) & PythonExe & Chr(34) & " " & Chr(34) & MenuScript & Chr(34), 0, False
    WScript.Quit
End If

' Si no hay Python configurado -> primera vez
Resultado = MsgBox( _
    "¡Bienvenido al Traductor LSE!" & vbCrLf & vbCrLf & _
    "Es la primera vez que abres la aplicación." & vbCrLf & _
    "Se configurará automáticamente (5-10 min)." & vbCrLf & vbCrLf & _
    "¿Deseas continuar?", _
    vbYesNo + vbInformation, _
    "Traductor LSE")

If Resultado = vbNo Then WScript.Quit

WshShell.Run "cmd /c """ & ScriptDir & "\Iniciar_LSE.bat""", 1, True

If FSO.FileExists(ScriptDir & "\.venv\Scripts\pythonw.exe") Then
    MsgBox "¡Listo! La aplicación se abrirá ahora." & vbCrLf & _
        "Las próximas veces abrirá directamente.", _
        vbInformation, "Traductor LSE"

    objEnv("TF_CPP_MIN_LOG_LEVEL") = "3"
    objEnv("PYTHONWARNINGS") = "ignore"
    WshShell.CurrentDirectory = ScriptDir
    WshShell.Run Chr(34) & ScriptDir & "\.venv\Scripts\pythonw.exe" & Chr(34) & " " & Chr(34) & MenuScript & Chr(34), 0, False
Else
    MsgBox "La configuración no se completó." & vbCrLf & _
        "Verifica tu conexión a internet e intenta de nuevo.", _
        vbCritical, "Error"
End If

Set WshShell = Nothing
Set FSO = Nothing

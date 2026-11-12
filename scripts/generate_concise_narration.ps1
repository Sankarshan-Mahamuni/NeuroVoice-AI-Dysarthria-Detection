Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice('Microsoft Zira Desktop')
$synth.Rate = 0  # natural, smooth, clear cadence

$outputPath = "E:\Dysarthia_Detection_Project\reports\demo_narration_concise.wav"
$synth.SetOutputToWaveFile($outputPath)

$text = Get-Content -Path "E:\Dysarthia_Detection_Project\scripts\narration_concise.txt" -Raw
$synth.Speak($text)
$synth.Dispose()
Write-Output "Generated concise narration at $outputPath"

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice('Microsoft Zira Desktop')
$synth.Rate = 1  # brisk, engaging, natural presentation pace

$outputPath = "E:\Dysarthia_Detection_Project\reports\demo_narration_final.wav"
$synth.SetOutputToWaveFile($outputPath)

$text = Get-Content -Path "E:\Dysarthia_Detection_Project\scripts\narration_final.txt" -Raw
$synth.Speak($text)
$synth.Dispose()
Write-Output "Generated final narration at $outputPath"

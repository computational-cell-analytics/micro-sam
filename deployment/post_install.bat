@echo off

rem This script searches for the 'freetype.dll' file within a specific directory 
rem and attempts to move it to the target directory during post-installation on Windows.
rem If not found locally, it downloads the DLL from a defined URL.


rem Define download URL (replace with the actual URL for your desired version)
set DOWNLOAD_URL=https://raw.githubusercontent.com/ubawurinna/freetype-windows-binaries/master/release%20dll/win64/freetype.dll

rem Define target directory within the installation prefix
set TARGET_DIR=%PREFIX%\Scripts

rem Create the target directory if it doesn't exist
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

rem Search for freetype.dll in specific directory (adjust path as needed)
for /F "tokens=*" %%a in ('dir /S /B "%PREFIX%\Library\freetype.dll"') do (
  if exist "%%a" (
    echo Found freetype.dll in %%a
    move "%%a" "%TARGET_DIR%"
    if exist "%TARGET_DIR%\freetype.dll" (
      echo Moved freetype.dll to %TARGET_DIR%
      exit /b 0  ; Exit script with success code if moved
    ) else (
      echo Error: Failed to move freetype.dll (check permissions)
      exit /b 1  ; Exit script with error code
    )
  )
)

rem If not found in specific location, proceed with download
powershell -ExecutionPolicy Bypass -Command Invoke-WebRequest -Uri %DOWNLOAD_URL% -OutFile "%TARGET_DIR%\freetype.dll"

if exist "%TARGET_DIR%\freetype.dll" (
  echo Downloaded freetype.dll to %TARGET_DIR%
) else (
  echo Error downloading freetype.dll
  rem Attempt to download the DLL again (optional)
  powershell -ExecutionPolicy Bypass -Command Invoke-WebRequest -Uri %DOWNLOAD_URL% -OutFile "%TARGET_DIR%\freetype.dll"
  if exist "%TARGET_DIR%\freetype.dll" (
    echo Retry: Downloaded freetype.dll to %TARGET_DIR%
  ) else (
    echo Error: Failed to download freetype.dll even after retry
    exit /b 1  ; Exit script with error code after retries
  )
)

exit /b 0  ; Exit script with success code (after download if necessary)
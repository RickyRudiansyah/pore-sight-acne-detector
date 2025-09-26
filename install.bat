@echo off

:: List of dependencies to install
set dependencies=flask ultralytics opencv-python matplotlib torch torchvision numpy Pillow PyYAML requests scipy tqdm seaborn pandas

:: Loop through each dependency and install it
for %%d in (%dependencies%) do (
    echo Installing %%d...
    pip install %%d
    if not !errorlevel! == 0 (
        echo Failed to install %%d
    )
)

:: Open app.py with Visual Studio Code after all installations are complete
code app.py

:: Ensure the correct environment and run the app.py file
call python app.py

echo Installation complete, app.py opened, and running.
pause

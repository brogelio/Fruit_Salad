## Requirements (Windows 10)
----
- Install [IntelRealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases/tag/v2.50.0)
- Install `python >= 3.7` and `pip` (e.g. through [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe))
- Install python packages in [requirements.txt](requirements.txt). See example below on how to use `pip`.
```
pip install --no-cache-dir -r requirements.txt
```

## Creating a Program Shortcut
----
1. Browse to your local copy of the source.
2. Right click [app.py](app.py). Select *Send to* then select *Desktop (create shortcut)*.
3. Right click on the newly created shortcut on the Desktop, then select *Properties*.
4. Change the `Target:` field to: `"%PYTHON_INSTALL%\pythonw.exe" "%SOURCE_FOLDER%\app.py"`.
    - %PYTHON_INSTALL%
    - %SOURCE_FOLDER% is the locat
    - Example: `"C:\Users\UserPC\Anaconda3\envs\gest\pythonw.exe" "C:\Users\UserPC\Downloads\Fruit_Salad\Release\app.py"`
5. Change the shortcut name and icon as needed.

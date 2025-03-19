@echo off
REM 获取当前路径
set "CUR_DIR=%~dp0"

REM 切换到D盘（确保路径正确）
D:

REM 结束所有正在运行的python.exe进程
taskkill /F /IM python.exe

REM 进入当前路径下的虚拟环境的Scripts目录
cd "%CUR_DIR%\.venv\Scripts"

REM 激活虚拟环境
call activate

REM 切换回项目根目录
cd "%CUR_DIR%"

REM 运行Python脚本
python wwqy.py
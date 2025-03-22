@echo off
REM 获取批处理文件所在的目录
set script_dir=%~dp0

REM 获取被拖拽的文件路径
set input_file=%1
REM 设置output_file和input_file相同
set output_file=%input_file%

REM 调用Python脚本进行简体到繁体转换
python "%script_dir%s2t.py" -i "%input_file%" -o "%output_file%"

pause

@echo off
REM ��ȡ�������ļ����ڵ�Ŀ¼
set script_dir=%~dp0

REM ��ȡ����ק���ļ�·��
set input_file=%1
REM ����output_file��input_file��ͬ
set output_file=%input_file%

REM ����Python�ű����м��嵽����ת��
python "%script_dir%s2t.py" -i "%input_file%" -o "%output_file%"

pause
